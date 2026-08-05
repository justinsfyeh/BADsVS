#!/usr/bin/env python3
"""
layers.py -- structural stratification for the MMPA pipeline.

Assigns every transformation rule A -> B to one of three design layers that
mirror the enumeration axes of the library (bridge x ring x functional group):

    linker       fragment spans >= 2 attachment points  (bridge chemistry)
    scaffold     single attachment, fragment contains a ring (azole swap)
    substituent  single attachment, acyclic fragment (functional group)

Publication figures (15_axis_figures) use a finer axis split via assign_axis():
pure acyclic bridges stay "bridge", while multi-attachment fragments that also
contain a ring are labelled "confounded" and excluded from single-axis panels.

Also provides:
  * undirected_key()      -> dedup mirror rows before ranking / FDR / plotting
  * frag_heavy()          -> fragment size, for the core:fragment ratio audit
  * recompute_composite() -> re-derive gated composite over a property SUBSET
                             without re-streaming the database (h50 on/off)
  * global_bh()           -> Benjamini-Hochberg across ALL rules x properties

Depends only on rdkit + numpy.
"""
from __future__ import annotations

import re
from collections import OrderedDict

import numpy as np

try:
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    _HAVE_RDKIT = True
except ImportError:                                    # pragma: no cover
    _HAVE_RDKIT = False

# layer ordering: higher = more structural
LAYER_RANK = OrderedDict([("substituent", 0), ("scaffold", 1), ("linker", 2)])
LAYERS = list(LAYER_RANK)

# axis ordering used by 15_axis_figures (publication naming)
AXIS_RANK = OrderedDict([
    ("substituent", 0), ("ring", 1), ("bridge", 2), ("confounded", 3),
])
AXES = list(AXIS_RANK)

_ATTACH_RE = re.compile(r"\[\*(?::\d+)?\]")
_RING_RE = re.compile(r"[a-zA-Z]\d|%\d\d")             # fallback ring detector


def n_attachments(frag: str) -> int:
    """Number of attachment points ([*:1], [*:2], ... or bare [*])."""
    if frag is None:
        return 0
    return len(_ATTACH_RE.findall(str(frag)))


def _mol(frag: str):
    if not _HAVE_RDKIT or frag is None:
        return None
    return Chem.MolFromSmiles(str(frag), sanitize=True)


def has_ring(frag: str) -> bool:
    m = _mol(frag)
    if m is not None:
        return m.GetRingInfo().NumRings() > 0
    return bool(_RING_RE.search(str(frag or "")))       # fallback


def frag_heavy(frag: str) -> int:
    """Heavy atoms in the fragment, EXCLUDING attachment dummies."""
    m = _mol(frag)
    if m is not None:
        return sum(1 for a in m.GetAtoms() if a.GetAtomicNum() > 1)
    s = _ATTACH_RE.sub("", str(frag or ""))
    return len(re.findall(r"[A-Z][a-z]?|[cnops]", s))


def frag_layer(frag: str) -> str:
    """Layer of a single fragment."""
    na = n_attachments(frag)
    if na >= 2:
        return "linker"
    return "scaffold" if has_ring(frag) else "substituent"


def assign_layer(frag_from: str, frag_to: str):
    """Layer of a rule. Returns (layer, cross_layer_flag).

    When the two endpoints disagree the more structural layer wins and
    cross_layer=True, so these rules can be reported separately (their effect
    sizes are not comparable with within-layer rules).
    """
    la, lb = frag_layer(frag_from), frag_layer(frag_to)
    if la == lb:
        return la, False
    return (la if LAYER_RANK[la] > LAYER_RANK[lb] else lb), True


def frag_axis(frag: str) -> str:
    """Publication axis of a single fragment.

    Splits the linker layer into pure bridges (>=2 attachments, acyclic) and
    confounded fragments that carry both a bridge and a ring.
    """
    na = n_attachments(frag)
    ring = has_ring(frag)
    if na >= 2 and ring:
        return "confounded"
    if na >= 2:
        return "bridge"
    if ring:
        return "ring"
    return "substituent"


def assign_axis(frag_from: str, frag_to: str):
    """Axis of a rule. Returns (axis, axis_clean).

    Confounded = either endpoint is simultaneously a bridge and a ring.
    Mixed-axis endpoint pairs take the more structural axis but are marked
    unclean (axis_clean=False), matching assign_layer's cross_layer semantics
    with inverted polarity on the flag.
    """
    a, b = frag_axis(frag_from), frag_axis(frag_to)
    if a == b:
        return a, a != "confounded"
    if a == "confounded" or b == "confounded":
        return "confounded", False
    winner = a if AXIS_RANK[a] > AXIS_RANK[b] else b
    return winner, False


def undirected_key(frag_from: str, frag_to: str, canon=None) -> str:
    """Direction-free identifier so mirror rows collapse to one rule.

    Pass canon=mmpa_lib.canon_fragment so the key uses the SAME normalisation
    as stage 07's endpoint matching (bare `*` -> `[*]`, attachment numbering
    ignored, RDKit-canonical). Without it, keys built here will not line up
    with candidate fragment inventories.

    NOTE for 2-cut linker fragments: canon_fragment deliberately ignores
    attachment NUMBERING, so [*:1]X[*:2] and [*:2]X[*:1] collapse. That is
    correct for endpoint membership but means a rule whose only difference is
    which ring sits on which side is not distinguished. Check this on the
    linker layer before ranking.
    """
    f = canon or (lambda x: str(x))
    return "||".join(sorted([f(frag_from), f(frag_to)]))


def size_audit(frag_from, frag_to, core_heavy=None):
    """Fragment-size descriptors for the MMPA size-convention audit.

    core_heavy: heavy atoms of the constant part (from constant_smiles).
    Returns dict; frag_ratio is the fraction of the molecule being exchanged,
    which standard MMPA practice keeps below ~0.33.
    """
    hf, ht = frag_heavy(frag_from), frag_heavy(frag_to)
    out = {"frag_heavy_from": hf, "frag_heavy_to": ht,
           "frag_heavy_max": max(hf, ht), "frag_heavy_diff": abs(hf - ht)}
    if core_heavy is not None and core_heavy > 0:
        out["core_heavy"] = core_heavy
        out["frag_ratio"] = max(hf, ht) / (max(hf, ht) + core_heavy)
        out["core_over_frag"] = core_heavy / max(max(hf, ht), 1)
    else:
        out["core_heavy"] = ""
        out["frag_ratio"] = ""
        out["core_over_frag"] = ""
    return out


# ---------------------------------------------------------------------------
# multiplicity + property-subset composites
# ---------------------------------------------------------------------------
def global_bh(pvals, alpha=0.05):
    """Benjamini-Hochberg over a FLAT list of p-values (all rules x props).

    Stage 04 currently applies BH within each rule (m = n_properties), which
    does not control the false discovery rate across the ~1e5 mined rules.
    Feed this the pooled p-values of the UNDIRECTED rule set instead.
    """
    p = np.asarray(pvals, dtype=float)
    sig = np.zeros_like(p, dtype=bool)
    idx = np.where(np.isfinite(p))[0]
    if idx.size == 0:
        return sig
    order = idx[np.argsort(p[idx])]
    m = order.size
    thresh = 0
    for rank, i in enumerate(order, start=1):
        if p[i] <= alpha * rank / m:
            thresh = rank
    for rank, i in enumerate(order, start=1):
        if rank <= thresh:
            sig[i] = True
    return sig


def recompute_composite(row, props, weights=None):
    """Re-derive gated/ungated composite over an arbitrary property SUBSET.

    Uses the per-property z_* and sig_* columns already present in
    transformations.csv, so switching h50/nu in or out needs no DB re-stream.
    Returns dict with composite_gated, composite_z, n_improve, n_worsen,
    has_tradeoff.
    """
    w = weights or {p: 1.0 for p in props}
    gated = 0.0
    total = 0.0
    n_imp = n_wor = 0
    for p in props:
        z = row.get(f"z_{p}")
        try:
            z = float(z)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(z):
            continue
        total += w[p] * z
        if str(row.get(f"sig_{p}")).lower() == "true":
            gated += w[p] * z
            if z > 0:
                n_imp += 1
            elif z < 0:
                n_wor += 1
    return {"composite_gated": gated, "composite_z": total,
            "n_improve": n_imp, "n_worsen": n_wor,
            "has_tradeoff": (n_imp > 0 and n_wor > 0)}


def annotate_transformations(df, props=None, weights=None, dedup=False, canon=None):
    """Add layer / cross_layer / undirected_key / size columns to a
    transformations.csv DataFrame. Optionally recompute composites over
    `props` and collapse mirror rows."""
    lay = df.apply(lambda r: assign_layer(r["frag_from"], r["frag_to"]),
                   axis=1, result_type="expand")
    df = df.copy()
    df["layer"] = lay[0]
    df["cross_layer"] = lay[1]
    df["undirected_key"] = df.apply(
        lambda r: undirected_key(r["frag_from"], r["frag_to"], canon), axis=1)
    sz = df.apply(lambda r: size_audit(r["frag_from"], r["frag_to"]),
                  axis=1, result_type="expand")
    for c in sz.columns:
        df[c] = sz[c]
    if props:
        rec = df.apply(lambda r: recompute_composite(r, props, weights),
                       axis=1, result_type="expand")
        for c in rec.columns:
            df[c] = rec[c]
    if dedup:
        df = (df.sort_values("composite_gated", ascending=False)
                .drop_duplicates("undirected_key", keep="first"))
    return df
