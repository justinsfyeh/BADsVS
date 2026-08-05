#!/usr/bin/env python3
"""
config.py -- single source of truth for the v4 layered MMPA pipeline.

v4 changes vs v3
----------------
1. The LINKER-inclusive index is now the ONLY database. Its cut-SMARTS is a
   superset of the old carbon-anchored one, so a single index carries all
   three design layers and `layers.assign_layer` separates them. Never pool
   results from two indexes: the same molecular pair would be counted twice
   and the global scales (z denominators) would differ.

2. `nu` (ESP charge balance) is reported as a NEUTRAL property. It is the
   quantity actually computed; mapping it to h50 via the Rice-Hare
   correlation is out of applicability domain for this chemistry, so `nu` is
   never allowed into the gated composite by default.

3. `IS` (Rice-Hare h50, cm) is retained so the with/without comparison can be
   run, but it is NOT in the default composite. Switch with --composite.

4. Chemotypes gain `hydrazo` and `azo`. Without them a bridging -NH-NH- is
   labelled `amino` and -N=N- is unlabelled, which mis-names the single most
   important linker-layer transformation (azoxy <-> hydrazo).
"""

import os
import shutil
from pathlib import Path

# --------------------------------------------------------------------------
# Properties
# --------------------------------------------------------------------------
# OBgood = -|OB|, so every directed objective is "higher is better".
# NEUTRAL properties get direction +1 (a reporting convention so z has a sign)
# but are excluded from every composite unless explicitly requested.
PROPERTIES = ["Hf", "Q", "OBgood", "Pe", "D", "P", "IS", "nu"]
MAXIMIZE   = ["Hf", "Q", "OBgood", "Pe", "D", "P", "IS"]
MINIMIZE   = []
NEUTRAL    = ["nu"]

# Which properties enter the gated composite. Pick with --composite <name>.
COMPOSITE_PRESETS = {
    "core":       ["Hf", "Q", "OBgood", "Pe", "D", "P"],
    "with_is":    ["Hf", "Q", "OBgood", "Pe", "D", "P", "IS"],
    "with_nu":    ["Hf", "Q", "OBgood", "Pe", "D", "P", "nu"],
}
DEFAULT_COMPOSITE = "core"

PROP_FAMILIES = {
    "performance": ["Q", "D", "P"],
    "energy":      ["Hf"],
    "detonation":  ["Pe"],
    "oxygen":      ["OBgood"],
    "electronic":  ["nu"],          # was "sensitivity": ["IS"]
    "sensitivity": ["IS"],          # reported, out-of-domain, see docstring
}


def directions():
    """{property: +1/-1}. NEUTRAL properties get +1 by convention."""
    d = {p: +1 for p in MAXIMIZE}
    d.update({p: -1 for p in MINIMIZE})
    d.update({p: +1 for p in NEUTRAL})
    missing = [p for p in PROPERTIES if p not in d]
    if missing:
        raise ValueError(f"No direction for properties: {missing}")
    return d


def composite_properties(name=None):
    """Property subset that enters the gated composite."""
    name = name or DEFAULT_COMPOSITE
    if name not in COMPOSITE_PRESETS:
        raise ValueError(f"Unknown composite preset {name!r}; "
                         f"choose from {list(COMPOSITE_PRESETS)}")
    return list(COMPOSITE_PRESETS[name])


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
# Override with env vars for alternate runs, e.g. support-filtered analysis:
#   export MMPA_RESULTS=/work/.../results_linker_s100
#   export MMPA_DB=/work/.../results_linker/pairs.mmpdb
ROOT      = Path(__file__).resolve().parent
DATA_DIR  = ROOT / "data"
RESULTS   = Path(os.environ["MMPA_RESULTS"]) if os.environ.get("MMPA_RESULTS") \
            else (ROOT / "results_linker")   # v4: linker index is THE index
AGG_DIR   = RESULTS / "aggregated"
FIG_DIR   = RESULTS / "figures"

MASTER_CSV     = ROOT / "master.csv"
CANDIDATES_CSV = ROOT / "candidates.csv"
DB             = Path(os.environ["MMPA_DB"]) if os.environ.get("MMPA_DB") \
            else (ROOT / "results_linker" / "pairs.mmpdb")

SMI_FILE       = DATA_DIR / "mols.smi"
FRAG_FILE      = DATA_DIR / "mols_linker.fragments"
PROP_FILE      = DATA_DIR / "properties.tsv"
CANDIDATES_TXT = DATA_DIR / "candidates.txt"
DATABASE_TXT   = DATA_DIR / "database.txt"
MANIFEST       = DATA_DIR / "manifest.json"

COL_SMILES       = "smiles"
COL_ID           = "mol_id"
COL_IS_CANDIDATE = "is_candidate"

# --------------------------------------------------------------------------
# MMP / statistical parameters
# --------------------------------------------------------------------------
RADIUS_GENERAL      = 0
MIN_RULE_SUPPORT    = 5      # mining floor; raise per layer at report time
MIN_CONTEXT_SUPPORT = 3
ALPHA               = 0.05

# Reporting thresholds per layer. The linker layer explores a much smaller
# combinatorial space than the substituent layer, so a flat threshold either
# starves the linker layer or floods the substituent layer.
MIN_SUPPORT_BY_LAYER = {"substituent": 20, "scaffold": 20, "linker": 10}

# Standard MMPA size conventions, used by the size audit (not enforced).
# Bajorath: core >= 2x each exchanged fragment; mmpdb: fragment/molecule < 0.33.
SIZE_CONVENTION = {"max_frag_ratio": 0.33, "min_core_over_frag": 2.0}

# --------------------------------------------------------------------------
# Benchmark policy
# --------------------------------------------------------------------------
BENCHMARK = {
    "source": "internal",
    "top_n": 10,
    "rank_col": "rank",
    "pool": "candidates",
    "agg": "median",
    "percentile": 90.0,
    "external_csv": None,
}

# --------------------------------------------------------------------------
# Chemotype SMARTS.
# Order matters: primary_chemotype takes the FIRST match.
#   azoxy  (N=N+-O-)  must precede n_oxide (aromatic n+ -> O-)
#   azo    (-N=N-)    must precede azole_ring, else a bridged azo reads as ring
#   hydrazo(-NH-NH-)  must precede amino,      else a bridge reads as amino
# --------------------------------------------------------------------------
CHEMOTYPE_SMARTS = [
    ("nitramine",     "[NX3][NX3+](=O)[O-]"),
    ("nitrate_ester", "[OX2][NX3+](=O)[O-]"),
    ("azide",         "[NX2]=[NX2+]=[NX1-]"),
    ("azoxy",         "[#7]=[#7+;!a][O-]"),
    ("nitro",         "[NX3+](=O)[O-]"),
    ("n_oxide",       "[n+][O-]"),
    ("azo",           "[NX2;!+;!a]=[NX2;!+;!a]"),
    ("hydrazo",       "[NX3;H1;!a][NX3;H1;!a]"),
    ("azole_ring",    "[nX2r5,nX3r5]"),
    ("amino",         "[NX3;H1,H2;!$([NX3][NX3+](=O)[O-])]"),
    ("hydroxyl",      "[OX2H]"),
    ("carbonyl",      "[CX3]=[OX1]"),
    ("fluoro",        "[F]"),
]

# --------------------------------------------------------------------------
# Environment (resolved from PATH after `conda activate mmp`)
# --------------------------------------------------------------------------
PYTHON = shutil.which("python3") or shutil.which("python") or "python3"
MMPDB  = shutil.which("mmpdb") or "mmpdb"
