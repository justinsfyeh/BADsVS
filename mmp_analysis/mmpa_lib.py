#!/usr/bin/env python3
"""
mmpa_lib.py -- shared utilities for the v2 anchored-MMPA analysis stages
(07-11). Pure-Python + numpy; RDKit is imported lazily and only where a
stage actually needs to draw a structure.

Nothing here assumes a property direction; direction (`d_P in {+1,-1}`)
is supplied by the caller wherever it is needed.
"""

import csv
import math
import sqlite3
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------
# DB access (mmpdb 3.x schema: pair.rule_environment_id -> rule_environment
# -> rule -> rule_smiles; pair.constant_id -> constant_smiles)
# --------------------------------------------------------------------------
def connect(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def property_name_ids(conn, properties):
    """Return {name: property_name_id} for properties present in the DB."""
    out = {}
    for p in properties:
        row = conn.execute(
            "SELECT id FROM property_name WHERE name = ?", (p,)
        ).fetchone()
        if row is not None:
            out[p] = row["id"]
        else:
            print(f"WARNING: property '{p}' not in DB", file=sys.stderr)
    return out


def load_property_values(conn, prop_ids):
    """
    Return {(compound_int_id, prop_name): value} for all compounds.
    `prop_ids` is {name: property_name_id}.
    """
    id_to_name = {v: k for k, v in prop_ids.items()}
    placeholders = ",".join("?" for _ in prop_ids)
    cur = conn.execute(
        f"SELECT compound_id, property_name_id, value FROM compound_property "
        f"WHERE property_name_id IN ({placeholders})",
        tuple(prop_ids.values()),
    )
    out = {}
    for row in cur:
        out[(row["compound_id"], id_to_name[row["property_name_id"]])] = row["value"]
    return out


def compound_id_maps(conn):
    """Return (public->int, int->public)."""
    pub_to_int, int_to_pub = {}, {}
    for row in conn.execute("SELECT id, public_id FROM compound"):
        pub_to_int[row["public_id"]] = row["id"]
        int_to_pub[row["id"]] = row["public_id"]
    return pub_to_int, int_to_pub


def environment_radii(conn):
    """Distinct rule_environment radii present in the DB (sorted)."""
    try:
        return sorted(r[0] for r in
                      conn.execute("SELECT DISTINCT radius FROM rule_environment"))
    except sqlite3.OperationalError:
        return []


def iter_all_pairs(conn, batch=50000, radius=0, min_support=1):
    """
    Stream every pair in the index (no candidate filter).

    A real `mmpdb index` stores environment fingerprints at several radii, so
    the SAME molecular pair (compound1, compound2, constant) appears once per
    radius. Counting all of them inflates pair counts and significance stats.
    `radius` restricts to a single environment radius (default 0 = the most
    general, environment-agnostic rule) so each molecular pair is counted once.
    Pass radius=None to disable filtering (only safe if the DB is single-radius).

    `min_support` keeps only rules with at least that many pairs at the chosen
    radius (or across all radii if radius is None).

    Yields dicts:
      c1, c2                 compound int ids (rule goes from c1's frag -> c2's frag)
      from_smiles, to_smiles fragment SMILES on each side of the rule
      context                shared constant SMILES (may be None)

    mmpdb convention: pair.compound1 carries rule.from_smiles,
    pair.compound2 carries rule.to_smiles.
    """
    params = {}
    where = []
    if radius is not None:
        where.append("re.radius = :radius")
        params["radius"] = radius
    support_join = ""
    if min_support is not None and min_support > 1:
        radius_filter = "WHERE re2.radius = :radius" if radius is not None else ""
        support_join = f"""
        JOIN (
            SELECT re2.rule_id AS rule_id
            FROM pair p2
            JOIN rule_environment re2 ON p2.rule_environment_id = re2.id
            {radius_filter}
            GROUP BY re2.rule_id
            HAVING COUNT(*) >= :min_support
        ) AS supported ON re.rule_id = supported.rule_id
        """
        params["min_support"] = int(min_support)
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""
    query = f"""
    SELECT pair.compound1_id AS c1,
           pair.compound2_id AS c2,
           rs_from.smiles    AS from_smiles,
           rs_to.smiles      AS to_smiles,
           cs.smiles         AS context
    FROM pair
    JOIN rule_environment re ON pair.rule_environment_id = re.id
    JOIN rule              ON re.rule_id = rule.id
    JOIN rule_smiles rs_from ON rule.from_smiles_id = rs_from.id
    JOIN rule_smiles rs_to   ON rule.to_smiles_id   = rs_to.id
    LEFT JOIN constant_smiles cs ON pair.constant_id = cs.id
    {support_join}
    {where_sql}
    """
    cur = conn.execute(query, params)
    while True:
        rows = cur.fetchmany(batch)
        if not rows:
            break
        for r in rows:
            yield {
                "c1": r["c1"], "c2": r["c2"],
                "from_smiles": r["from_smiles"], "to_smiles": r["to_smiles"],
                "context": r["context"],
            }


# --------------------------------------------------------------------------
# Streaming moment accumulator (for means / std / correlation at scale)
# --------------------------------------------------------------------------
class Moments:
    """Univariate running moments (n, sum, sumsq, n_positive)."""
    __slots__ = ("n", "s", "ss", "npos")

    def __init__(self):
        self.n = 0; self.s = 0.0; self.ss = 0.0; self.npos = 0

    def add(self, x):
        self.n += 1; self.s += x; self.ss += x * x
        if x > 0:
            self.npos += 1

    @property
    def mean(self):
        return self.s / self.n if self.n else float("nan")

    @property
    def std(self):
        if self.n < 2:
            return 0.0
        var = (self.ss - self.s * self.s / self.n) / (self.n - 1)
        return math.sqrt(max(var, 0.0))

    @property
    def frac_pos(self):
        return self.npos / self.n if self.n else float("nan")

    @property
    def sem(self):
        return self.std / math.sqrt(self.n) if self.n else float("nan")

    @property
    def tstat(self):
        se = self.sem
        return self.mean / se if se else float("nan")


class Bivariate:
    """Running bivariate moments for Pearson correlation."""
    __slots__ = ("n", "sx", "sy", "sxx", "syy", "sxy")

    def __init__(self):
        self.n = 0; self.sx = self.sy = 0.0
        self.sxx = self.syy = self.sxy = 0.0

    def add(self, x, y):
        self.n += 1
        self.sx += x; self.sy += y
        self.sxx += x * x; self.syy += y * y; self.sxy += x * y

    @property
    def corr(self):
        n = self.n
        if n < 3:
            return float("nan")
        cov = self.sxy - self.sx * self.sy / n
        vx = self.sxx - self.sx * self.sx / n
        vy = self.syy - self.sy * self.sy / n
        if vx <= 0 or vy <= 0:
            return float("nan")
        return cov / math.sqrt(vx * vy)


# --------------------------------------------------------------------------
# Multi-objective helpers
# --------------------------------------------------------------------------
def directed(values, directions):
    """Map raw property vector to a 'higher-is-better' vector via d_P."""
    return np.array([directions[p] * v for p, v in values.items()])


def dominates(a, b, eps=0.0):
    """
    True if a weakly dominates b and strictly beats it on >=1 axis,
    where both a,b are 'higher-is-better' vectors (np arrays). NaNs are
    treated as 'no information' and ignored pairwise.
    """
    mask = ~(np.isnan(a) | np.isnan(b))
    if not mask.any():
        return False
    aa, bb = a[mask], b[mask]
    return np.all(aa >= bb - eps) and np.any(aa > bb + eps)


def pareto_front(points):
    """
    points: list of 'higher-is-better' np arrays.
    Returns a boolean list: True if the point is non-dominated.
    """
    n = len(points)
    nd = [True] * n
    for i in range(n):
        if not nd[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if dominates(points[j], points[i]):
                nd[i] = False
                break
    return nd


# --------------------------------------------------------------------------
# IO helpers
# --------------------------------------------------------------------------
def fnum(v, fmt="{:.6g}"):
    if v is None or v == "" or (isinstance(v, float) and math.isnan(v)):
        return ""
    return fmt.format(v)


def write_rows(path, rows, fieldnames, float_cols=()):
    float_cols = set(float_cols)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            out = {}
            for k in fieldnames:
                v = r.get(k, "")
                out[k] = fnum(v) if k in float_cols else v
            w.writerow(out)


def read_rows(path, numeric=()):
    numeric = set(numeric)
    out = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            d = dict(r)
            for k in numeric:
                v = d.get(k, "")
                d[k] = float(v) if v not in ("", None) else None
            out.append(d)
    return out


def parse_directions(properties, maximize, minimize):
    d = {}
    for p in maximize:
        d[p] = +1
    for p in minimize:
        d[p] = -1
    missing = [p for p in properties if p not in d]
    if missing:
        sys.exit(f"ERROR: no direction for: {missing} (use --maximize/--minimize)")
    return d


def have_rdkit():
    try:
        import rdkit  # noqa
        return True
    except ImportError:
        return False


# --------------------------------------------------------------------------
# Chemotype tagging (shared by transformation mining, endpoints, figures)
# --------------------------------------------------------------------------
import re as _re

_ATTACH_RE = _re.compile(r"\[\*(?::\d+)?\]")
_CHEMO_PATTERNS = None      # lazily compiled {name: Mol}


def strip_attachment(frag_smiles):
    """Remove mmpdb attachment atoms ([*:1], [*], ...) from a fragment SMILES."""
    return _ATTACH_RE.sub("", str(frag_smiles)).strip()


_STAR_RE = _re.compile(r"\[\*(?::\d+)?\]|\*")


@lru_cache(maxsize=200000)
def canon_fragment(frag_smiles):
    """Canonical matching key for a fragment across attachment conventions.

    The fragments file writes attachment points as bare `*` while the mmpdb
    rule_smiles table uses `[*:1]`. Both are normalized to an unlabeled dummy
    `[*]` and RDKit-canonicalized, so a candidate's fragment inventory can be
    matched to mined-rule endpoints regardless of notation or atom ordering.
    Attachment-point NUMBERING is intentionally ignored (fine for endpoint
    membership). Returns the RDKit canonical SMILES, or the raw string on
    failure.
    """
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    s = _STAR_RE.sub("[*]", str(frag_smiles))
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return str(frag_smiles)
    return Chem.MolToSmiles(mol)


def _compile_chemotypes():
    global _CHEMO_PATTERNS
    if _CHEMO_PATTERNS is not None:
        return _CHEMO_PATTERNS
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    import config
    _CHEMO_PATTERNS = []
    for name, smarts in config.CHEMOTYPE_SMARTS:
        patt = Chem.MolFromSmarts(smarts)
        if patt is not None:
            _CHEMO_PATTERNS.append((name, patt))
    return _CHEMO_PATTERNS


@lru_cache(maxsize=100000)
def chemotype_tags(frag_smiles):
    """Return every chemotype label matching this fragment (may be empty).

    Parses the fragment WITH its [*:n] dummy atom preserved so that patterns
    keyed on real atoms (e.g. nitramine's linking N) only match when that atom
    is genuinely inside the fragment.
    """
    from rdkit import Chem
    patterns = _compile_chemotypes()
    mol = Chem.MolFromSmiles(str(frag_smiles))
    if mol is None:
        mol = Chem.MolFromSmiles(strip_attachment(frag_smiles))
    if mol is None:
        return []
    return [name for name, patt in patterns if mol.HasSubstructMatch(patt)]


def primary_chemotype(frag_smiles):
    """First matching chemotype label, or 'other'."""
    tags = chemotype_tags(frag_smiles)
    return tags[0] if tags else "other"


# --------------------------------------------------------------------------
# Benchmark policy (shared by candidate profiles + figures)
# --------------------------------------------------------------------------
def load_benchmark(master_df, properties, directions, spec, candidates_csv=None):
    """
    Build a benchmark reference vector from a spec dict (see config.BENCHMARK).

    Returns (ref_vector, benchmark_ids, description):
      ref_vector    {property: float}   the bar a candidate must beat
      benchmark_ids list of mol ids in the reference set ([] for external)
      description   human-readable string for captions
    """
    import numpy as np
    import pandas as pd

    source = spec.get("source", "internal")
    agg = spec.get("agg", "median")

    def aggregate(df):
        vec = {}
        for p in properties:
            vals = pd.to_numeric(df[p], errors="coerce").dropna().values
            if len(vals) == 0:
                vec[p] = float("nan"); continue
            if agg == "mean":
                vec[p] = float(np.mean(vals))
            elif agg == "best":
                vec[p] = float(np.max(vals) if directions[p] > 0 else np.min(vals))
            elif agg == "percentile":
                q = spec.get("percentile", 90.0)
                q = q if directions[p] > 0 else (100.0 - q)
                vec[p] = float(np.percentile(vals, q))
            else:  # median
                vec[p] = float(np.median(vals))
        return vec

    if source == "external":
        path = spec.get("external_csv")
        if not path:
            raise ValueError("benchmark source=external but external_csv not set")
        ext = pd.read_csv(path)
        ref = aggregate(ext)
        ids = ext[ext.columns[0]].astype(str).tolist()
        return ref, ids, f"external:{path} (agg={agg}, n={len(ext)})"

    # internal
    pool = spec.get("pool", "candidates")
    if pool == "candidates":
        sub = master_df[master_df[_ISCAND(master_df)] == 1].copy()
    elif pool == "database":
        sub = master_df[master_df[_ISCAND(master_df)] == 0].copy()
    else:
        sub = master_df.copy()

    top_n = int(spec.get("top_n", 10))
    rank_col = spec.get("rank_col", "rank")
    ranked_ids = None
    if candidates_csv is not None and Path(candidates_csv).exists():
        cc = pd.read_csv(candidates_csv)
        if rank_col in cc.columns and "label" in cc.columns:
            cc = cc.sort_values(rank_col).head(top_n)
            ranked_ids = cc["label"].astype(str).tolist()

    if ranked_ids is not None:
        idcol = _MOLID(sub)
        chosen = sub[sub[idcol].astype(str).isin(ranked_ids)]
        if len(chosen) == 0:
            chosen = sub.head(top_n)
        desc = f"internal top-{top_n} by {rank_col} (agg={agg})"
    else:
        # fall back: top-N by directed standardized composite over the pool
        comp = _composite(sub, properties, directions)
        chosen = sub.loc[comp.sort_values(ascending=False).head(top_n).index]
        desc = f"internal top-{top_n} by composite (agg={agg})"

    ref = aggregate(chosen)
    ids = chosen[_MOLID(chosen)].astype(str).tolist()
    return ref, ids, desc


def _ISCAND(df):
    import config
    return config.COL_IS_CANDIDATE if config.COL_IS_CANDIDATE in df.columns else "is_candidate"


def _MOLID(df):
    import config
    return config.COL_ID if config.COL_ID in df.columns else "mol_id"


def _composite(df, properties, directions):
    import numpy as np
    import pandas as pd
    z = pd.DataFrame(index=df.index)
    for p in properties:
        v = pd.to_numeric(df[p], errors="coerce")
        sd = v.std() or 1.0
        z[p] = directions[p] * (v - v.mean()) / sd
    return z.mean(axis=1)


def beats_benchmark(cand_values, ref_vector, properties, directions, eps=0.0):
    """True if candidate weakly dominates ref on all props and strictly on >=1."""
    import math
    ge_all, gt_any = True, False
    for p in properties:
        cv, rv = cand_values.get(p), ref_vector.get(p)
        if cv is None or rv is None or (isinstance(cv, float) and math.isnan(cv)) \
                or (isinstance(rv, float) and math.isnan(rv)):
            continue
        diff = directions[p] * (cv - rv)
        if diff < -eps:
            ge_all = False
        if diff > eps:
            gt_any = True
    return ge_all and gt_any


def benchmark_scorecard(cand_values, ref_vector, s_global, properties, directions, eps=0.0):
    """Partial-dominance view of a candidate vs the benchmark vector.

    Returns (n_obj_beat, composite_vs_benchmark):
      n_obj_beat             number of objectives where candidate is favorable
      composite_vs_benchmark sum over props of directed, std-scaled (cand - ref)
    (Strict all-objective domination is rare under EM tradeoffs; these give an
    honest graded picture for tables/figures.)
    """
    import math
    n_beat = 0
    comp = 0.0
    for p in properties:
        cv, rv = cand_values.get(p), ref_vector.get(p)
        if cv is None or rv is None or (isinstance(cv, float) and math.isnan(cv)) \
                or (isinstance(rv, float) and math.isnan(rv)):
            continue
        diff = directions[p] * (cv - rv)
        if diff > eps:
            n_beat += 1
        sd = s_global.get(p, 1.0) or 1.0
        comp += diff / sd
    return n_beat, comp
