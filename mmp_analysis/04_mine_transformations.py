#!/usr/bin/env python3
"""
04_mine_transformations.py -- PRIMARY discovery stage (v4 stage 4).

Mines every well-supported transformation rule A -> B from the linker-inclusive
index at radius = config.RADIUS_GENERAL and characterizes each rule by effect,
transferability, standardized z, and chemistry.

v4 changes vs v3
----------------
1. LAYER. Every rule is tagged linker / scaffold / substituent (layers.py), so
   bridge chemistry, ring swaps and substituent edits are never ranked against
   each other. Rules whose two endpoints sit in different layers are flagged
   cross_layer=True; their effect sizes are not comparable with within-layer
   rules and should be reported separately.

2. GLOBAL FDR. v3 ran Benjamini-Hochberg inside each rule (m = n_properties),
   which does not control the false discovery rate across the ~1e5 mined
   rules. v4 pools every (rule, property) p-value from the UNDIRECTED rule set
   and applies BH once. Expect the count of significant rules to drop sharply
   -- that drop is the point. Use --local-bh to reproduce the v3 behaviour.

3. SIZE AUDIT. The SQL now joins constant_smiles so each rule records the mean
   heavy-atom count of its constant part. frag_ratio = frag/(frag+core) is the
   quantity standard MMPA practice keeps below ~0.33; reporting it is how the
   "your transformations are too large" objection gets answered with data
   rather than assertion.

4. UNDIRECTED KEY. Both directions are still emitted (stage 07 needs the
   reverse edit for candidates carrying B), but every row now carries
   undirected_key so ranking, plotting and FDR can deduplicate. Any statistic
   computed over transformations.csv WITHOUT deduplicating is double counted.

5. COMPOSITE SUBSET. Per-property z_* and sig_* are written for ALL properties;
   the gated composite is summed over --composite only. Switching h50/nu in or
   out therefore needs no database re-stream -- see 13_layer_report.py.

Output (config.RESULTS):
  transformations.csv   one row per DIRECTED rule (>= MIN_RULE_SUPPORT pairs)

Usage:
  <tartarus python> 04_mine_transformations.py
  <tartarus python> 04_mine_transformations.py --composite with_is
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import config
import layers as L
import mmpa_lib as ml


def stream_rules(conn, props, pv, radius, min_support=1):
    """rule_id -> {"from","to","n","mom":{P:Moments},"core":Moments}.

    If min_support > 1, only rules with at least that many pairs at `radius`
    are streamed (same filter semantics as stage 03).
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
    SELECT re.rule_id AS rule_id,
           pair.compound1_id AS c1,
           pair.compound2_id AS c2,
           rs_from.smiles AS ffrom,
           rs_to.smiles   AS fto,
           cs.smiles      AS ctx
    FROM pair
    JOIN rule_environment re ON pair.rule_environment_id = re.id
    JOIN rule ON re.rule_id = rule.id
    JOIN rule_smiles rs_from ON rule.from_smiles_id = rs_from.id
    JOIN rule_smiles rs_to   ON rule.to_smiles_id   = rs_to.id
    LEFT JOIN constant_smiles cs ON pair.constant_id = cs.id
    {support_join}
    {where_sql}
    """
    cur = conn.execute(query, params)
    stats_map = {}
    n_pairs = 0
    while True:
        rows = cur.fetchmany(50000)
        if not rows:
            break
        for r in rows:
            n_pairs += 1
            if n_pairs % 500000 == 0:
                print(f"  ...{n_pairs} pairs")
            rid = r["rule_id"]
            st = stats_map.get(rid)
            if st is None:
                st = {"from": r["ffrom"], "to": r["fto"], "n": 0,
                      "mom": {p: ml.Moments() for p in props},
                      "core": ml.Moments()}
                stats_map[rid] = st
            st["n"] += 1
            if r["ctx"] is not None:
                st["core"].add(L.frag_heavy(r["ctx"]))
            c1, c2 = r["c1"], r["c2"]
            for p in props:
                a = pv.get((c1, p)); b = pv.get((c2, p))
                if a is None or b is None:
                    continue
                st["mom"][p].add(b - a)      # dP = P(to) - P(from)
    print(f"Streamed {n_pairs} pairs across {len(stats_map)} rules "
          f"(min_support={min_support})")
    return stats_map


def build_raw(st, props, direction):
    """Per-rule statistics in the CANONICAL (stored) direction, no significance
    yet -- significance needs the pooled p-values of every rule."""
    ffrom, fto = st["from"], st["to"]
    lay, cross = L.assign_layer(ffrom, fto)
    core_h = st["core"].mean if st["core"].n else None
    rec = {
        "_ffrom": ffrom, "_fto": fto,
        "n_pairs": st["n"], "n_contexts": st["n"],   # == n at radius 0
        "layer": lay, "cross_layer": cross,
        "undirected_key": L.undirected_key(ffrom, fto, ml.canon_fragment),
    }
    rec.update(L.size_audit(ffrom, fto, core_heavy=core_h))
    for p in props:
        m = st["mom"][p]
        if m.n >= 2:
            mean_d, sem, std_d = m.mean, m.sem, m.std
            t = mean_d / sem if sem else float("nan")
            pval = float(2 * stats.t.sf(abs(t), df=m.n - 1)) if sem else float("nan")
            fpos = m.frac_pos
        elif m.n == 1:
            mean_d, sem, std_d = m.mean, float("nan"), 0.0
            t = pval = fpos = float("nan")
        else:
            mean_d = sem = std_d = t = pval = fpos = float("nan")
        rec[f"_mean_{p}"] = mean_d
        rec[f"_fpos_{p}"] = fpos
        rec[f"std_d_{p}"] = std_d
        rec[f"p_{p}"] = pval
        rec[f"n_{p}"] = m.n
    return rec


def finalize(raw, props, comp_props, direction, s_global, weights, flip):
    """Emit one directed row. flip=True negates the canonical deltas."""
    sign = -1.0 if flip else 1.0
    ffrom = raw["_fto"] if flip else raw["_ffrom"]
    fto = raw["_ffrom"] if flip else raw["_fto"]
    rec = {
        "frag_from": ffrom, "frag_to": fto,
        "chemotype_from": ml.primary_chemotype(ffrom),
        "chemotype_to": ml.primary_chemotype(fto),
        "layer": raw["layer"], "cross_layer": raw["cross_layer"],
        "undirected_key": raw["undirected_key"],
        "direction": "reverse" if flip else "canonical",
        "n_pairs": raw["n_pairs"], "n_contexts": raw["n_contexts"],
        "core_heavy": raw["core_heavy"],
        "frag_heavy_max": raw["frag_heavy_max"],
        "frag_heavy_diff": raw["frag_heavy_diff"],
        "frag_ratio": raw["frag_ratio"],
        "core_over_frag": raw["core_over_frag"],
    }
    n_imp = n_wor = 0
    comp_gated = 0.0
    comp_all = 0.0
    for p in props:
        mean_d = sign * raw[f"_mean_{p}"]
        fpos = raw[f"_fpos_{p}"]
        favorable = fpos if direction[p] > 0 else (1 - fpos)
        if flip and not np.isnan(favorable):
            favorable = 1 - favorable
        zv = (direction[p] * mean_d / s_global[p]
              if np.isfinite(mean_d) else float("nan"))
        rec[f"mean_d_{p}"] = mean_d
        rec[f"std_d_{p}"] = raw[f"std_d_{p}"]
        rec[f"z_{p}"] = zv
        rec[f"p_{p}"] = raw[f"p_{p}"]
        rec[f"sign_consistency_{p}"] = favorable
        rec[f"sig_{p}"] = bool(raw[f"sig_{p}"])
        if p in comp_props and np.isfinite(zv):
            comp_all += weights[p] * zv
            if rec[f"sig_{p}"]:
                comp_gated += weights[p] * zv
                if zv > 0:
                    n_imp += 1
                elif zv < 0:
                    n_wor += 1
    rec["composite_z"] = comp_all
    rec["composite_gated"] = comp_gated
    rec["n_improve"] = n_imp
    rec["n_worsen"] = n_wor
    rec["has_tradeoff"] = (n_imp > 0 and n_wor > 0)
    rec["composite_props"] = "|".join(comp_props)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(config.DB))
    ap.add_argument("--properties", nargs="+", default=config.PROPERTIES)
    ap.add_argument("--composite", default=config.DEFAULT_COMPOSITE,
                    choices=list(config.COMPOSITE_PRESETS),
                    help="Property subset entering the gated composite.")
    ap.add_argument("--global-scales", default=str(config.RESULTS / "global_scales.csv"))
    ap.add_argument("--min-support", type=int, default=config.MIN_RULE_SUPPORT)
    ap.add_argument("--alpha", type=float, default=config.ALPHA)
    ap.add_argument("--radius", type=int, default=config.RADIUS_GENERAL)
    ap.add_argument("--local-bh", action="store_true",
                    help="v3 behaviour: BH within each rule (NOT recommended).")
    ap.add_argument("--output", default=str(config.RESULTS / "transformations.csv"))
    args = ap.parse_args()

    props = args.properties
    direction = config.directions()
    comp_props = config.composite_properties(args.composite)
    weights = {p: 1.0 for p in props}
    print(f"Composite preset '{args.composite}': {comp_props}")

    s_global = {p: 1.0 for p in props}
    gs_path = Path(args.global_scales)
    if gs_path.exists():
        gs = pd.read_csv(gs_path).set_index("property")
        for p in props:
            if p in gs.index and gs.loc[p, "std"] > 0:
                s_global[p] = float(gs.loc[p, "std"])
    else:
        print(f"WARNING: {gs_path} missing; z-scores use std=1 "
              f"(run 03_fragment_goodness.py first)")

    conn = ml.connect(args.db)
    radii = ml.environment_radii(conn)
    radius = None if args.radius == -1 else args.radius
    if radius is not None and radii and radius not in radii:
        radius = radii[0]
    print(f"Mining transformations at radius={radius} (DB radii={radii}); "
          f"min_support={args.min_support}")
    prop_ids = ml.property_name_ids(conn, props)
    props = list(prop_ids.keys())
    comp_props = [p for p in comp_props if p in props]
    weights = {p: 1.0 for p in props}
    pv = ml.load_property_values(conn, prop_ids)
    print(f"Loaded {len(pv)} property values")

    stats_map = stream_rules(conn, props, pv, radius,
                             min_support=args.min_support)

    # ---- pass 1: canonical-direction statistics -------------------------
    raw = [build_raw(st, props, direction)
           for st in stats_map.values() if st["n"] >= args.min_support]
    print(f"{len(raw)} undirected rules with >= {args.min_support} pairs")

    # ---- global BH over the UNDIRECTED rule set -------------------------
    if args.local_bh:
        for r in raw:
            sig = L.global_bh([r[f"p_{p}"] for p in props], args.alpha)
            for j, p in enumerate(props):
                r[f"sig_{p}"] = bool(sig[j])
        print("BH applied WITHIN each rule (v3 behaviour, m=n_properties)")
    else:
        flat = [r[f"p_{p}"] for r in raw for p in props]
        sig_flat = L.global_bh(flat, args.alpha)
        for i, r in enumerate(raw):
            for j, p in enumerate(props):
                r[f"sig_{p}"] = bool(sig_flat[i * len(props) + j])
        n_tests = int(np.isfinite(np.asarray(flat, dtype=float)).sum())
        print(f"Global BH over {n_tests} (rule x property) tests at "
              f"alpha={args.alpha}: {int(sig_flat.sum())} significant")

    # ---- pass 2: emit both directions -----------------------------------
    out_rows = []
    for r in raw:
        out_rows.append(finalize(r, props, comp_props, direction,
                                 s_global, weights, flip=False))
        out_rows.append(finalize(r, props, comp_props, direction,
                                 s_global, weights, flip=True))
    out_rows.sort(key=lambda r: -r["composite_gated"])

    fields = ["frag_from", "frag_to", "chemotype_from", "chemotype_to",
              "layer", "cross_layer", "undirected_key", "direction",
              "n_pairs", "n_contexts", "core_heavy", "frag_heavy_max",
              "frag_heavy_diff", "frag_ratio", "core_over_frag"]
    floats = ["core_heavy", "frag_ratio", "core_over_frag"]
    for p in props:
        fields += [f"mean_d_{p}", f"std_d_{p}", f"z_{p}", f"p_{p}",
                   f"sign_consistency_{p}", f"sig_{p}"]
        floats += [f"mean_d_{p}", f"std_d_{p}", f"z_{p}", f"p_{p}",
                   f"sign_consistency_{p}"]
    fields += ["composite_z", "composite_gated", "n_improve", "n_worsen",
               "has_tradeoff", "composite_props"]
    floats += ["composite_z", "composite_gated"]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    ml.write_rows(args.output, out_rows, fields, floats)
    print(f"Wrote {len(out_rows)} directed rows "
          f"({len(raw)} undirected rules) -> {args.output}")

    # ---- layer summary ---------------------------------------------------
    df = pd.DataFrame(out_rows)
    print("\nRules per layer (undirected):")
    for lay, sub in df[df.direction == "canonical"].groupby("layer"):
        n_sig = int((sub.n_improve + sub.n_worsen > 0).sum())
        ratio = pd.to_numeric(sub.frag_ratio, errors="coerce").mean()
        print(f"  {lay:12s} n={len(sub):6d}  with>=1 sig prop={n_sig:6d}  "
              f"mean frag_ratio={ratio:.3f}")
    n_cross = int(df[df.direction == "canonical"].cross_layer.sum())
    if n_cross:
        print(f"  cross-layer rules (report separately): {n_cross}")


if __name__ == "__main__":
    main()
