#!/usr/bin/env python3
"""
03_fragment_goodness.py -- global evidence layer (v4 stage 3).

Unchanged definition:
    A_P(v) = mean over every pair where v appears of
             P(mol_with_v) - P(mol_with_partner_fragment)

v4 additions
------------
* every fragment carries its LAYER (linker / scaffold / substituent)
* the Delta-Delta correlation matrix is accumulated PER LAYER as well as
  pooled, so stage 09 can ask whether the population->edit decoupling differs
  between bridge chemistry, ring swaps and substituent edits
* per-layer constant (core) size distribution is recorded, which is the raw
  material for the fragment/core size audit

Run this ONCE on the linker-inclusive index. The global_scales.csv it writes
is the z denominator for every later stage; never mix scales from two indexes.

Outputs (config.RESULTS):
  fragment_goodness.csv   per-fragment A, s, n, frac_pos, t, layer, n_contexts
  global_scales.csv       per property: mean, std, n over all pair-deltas
  delta_corr.csv          long-form corr(dP,dQ) PER LAYER (+ layer='all')
  layer_sizes.csv         per-layer pair counts and core/fragment size stats

Usage:
  <tartarus python> 03_fragment_goodness.py
"""
import argparse
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import config
import layers as L
import mmpa_lib as ml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(config.DB))
    ap.add_argument("--properties", nargs="+", default=config.PROPERTIES)
    ap.add_argument("--output-dir", default=str(config.RESULTS))
    ap.add_argument("--radius", type=int, default=config.RADIUS_GENERAL,
                    help="Environment radius to keep. -1 = all.")
    ap.add_argument("--min-support", type=int, default=1,
                    help="Keep only rules with at least this many pairs "
                         "at the chosen radius.")
    ap.add_argument("--track-contexts", action="store_true", default=True)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    conn = ml.connect(args.db)
    radii = ml.environment_radii(conn)
    radius = None if args.radius == -1 else args.radius
    if radius is not None and radii and radius not in radii:
        print(f"WARNING: radius {radius} absent (DB radii={radii}); "
              f"using min radius {radii[0]} instead")
        radius = radii[0]
    print(f"DB environment radii present: {radii}; using radius={radius}; "
          f"min_support={args.min_support}")
    prop_ids = ml.property_name_ids(conn, args.properties)
    props = list(prop_ids.keys())
    missing = [p for p in args.properties if p not in props]
    if missing:
        print(f"WARNING: properties absent from DB and skipped: {missing}")
    pv = ml.load_property_values(conn, prop_ids)
    print(f"Loaded {len(pv)} property values for {len(props)} properties")

    buckets = ("all",) + tuple(L.LAYERS)

    frag_mom = defaultdict(lambda: {p: ml.Moments() for p in props})
    frag_pairs = defaultdict(int)
    frag_ctx = defaultdict(set) if args.track_contexts else None
    global_mom = {p: ml.Moments() for p in props}
    biv = {b: {(a, c): ml.Bivariate() for a, c in combinations(props, 2)}
           for b in buckets}
    layer_pairs = defaultdict(int)
    core_mom = {b: ml.Moments() for b in buckets}     # core heavy atoms
    fragsz_mom = {b: ml.Moments() for b in buckets}   # exchanged fragment size
    ratio_mom = {b: ml.Moments() for b in buckets}    # frag / (frag + core)

    n_pairs = 0
    for rec in ml.iter_all_pairs(conn, radius=radius,
                                 min_support=args.min_support):
        n_pairs += 1
        if n_pairs % 500000 == 0:
            print(f"  ...{n_pairs} pairs")
        c1, c2 = rec["c1"], rec["c2"]
        vfrom, vto = rec["from_smiles"], rec["to_smiles"]
        ctx = rec["context"]

        lay, _cross = L.assign_layer(vfrom, vto)
        layer_pairs[lay] += 1

        frag_pairs[vfrom] += 1
        frag_pairs[vto] += 1
        if frag_ctx is not None and ctx is not None:
            frag_ctx[vfrom].add(ctx)
            frag_ctx[vto].add(ctx)

        # ---- size audit -----------------------------------------------
        if ctx is not None:
            core_h = L.frag_heavy(ctx)
            frag_h = max(L.frag_heavy(vfrom), L.frag_heavy(vto))
            for b in ("all", lay):
                core_mom[b].add(core_h)
                fragsz_mom[b].add(frag_h)
                if core_h + frag_h > 0:
                    ratio_mom[b].add(frag_h / (frag_h + core_h))

        # ---- property deltas ------------------------------------------
        deltas = {}
        for p in props:
            a = pv.get((c1, p))
            b = pv.get((c2, p))
            if a is None or b is None:
                continue
            dP = b - a                       # P(to) - P(from)
            deltas[p] = dP
            global_mom[p].add(dP)
            frag_mom[vfrom][p].add(-dP)
            frag_mom[vto][p].add(+dP)

        for (a, c) in biv["all"]:
            if a in deltas and c in deltas:
                biv["all"][(a, c)].add(deltas[a], deltas[c])
                biv[lay][(a, c)].add(deltas[a], deltas[c])

    print(f"Streamed {n_pairs} pairs; {len(frag_pairs)} distinct fragments")
    print("Pairs per layer: " +
          ", ".join(f"{k}={v}" for k, v in sorted(layer_pairs.items())))

    # ---- fragment_goodness.csv ----
    fields = ["fragment", "chemotype", "layer", "n_attach", "frag_heavy",
              "n_total_pairs", "n_contexts"]
    for p in props:
        fields += [f"A_{p}", f"s_{p}", f"n_{p}", f"fpos_{p}", f"t_{p}"]
    float_cols = [c for c in fields if c.split("_")[0] in ("A", "s", "fpos", "t")]

    rows = []
    for frag, mom in frag_mom.items():
        r = {"fragment": frag,
             "chemotype": ml.primary_chemotype(frag),
             "layer": L.frag_layer(frag),
             "n_attach": L.n_attachments(frag),
             "frag_heavy": L.frag_heavy(frag),
             "n_total_pairs": frag_pairs[frag],
             "n_contexts": len(frag_ctx[frag]) if frag_ctx is not None else ""}
        for p in props:
            m = mom[p]
            r[f"A_{p}"] = m.mean if m.n else None
            r[f"s_{p}"] = m.std if m.n else None
            r[f"n_{p}"] = m.n
            r[f"fpos_{p}"] = m.frac_pos if m.n else None
            r[f"t_{p}"] = m.tstat if m.n > 1 else None
        rows.append(r)
    rows.sort(key=lambda r: -r["n_total_pairs"])
    ml.write_rows(out / "fragment_goodness.csv", rows, fields, float_cols)
    print(f"  -> {out/'fragment_goodness.csv'} ({len(rows)} fragments)")

    # ---- global_scales.csv ----
    gs_rows = [{"property": p, "mean": global_mom[p].mean,
                "std": global_mom[p].std, "n": global_mom[p].n} for p in props]
    ml.write_rows(out / "global_scales.csv", gs_rows,
                  ["property", "mean", "std", "n"], ["mean", "std"])
    print(f"  -> {out/'global_scales.csv'}")

    # ---- delta_corr.csv (per layer) ----
    dc_rows = []
    for b in buckets:
        for (a, c), bv in biv[b].items():
            dc_rows.append({"layer": b, "p1": a, "p2": c,
                            "corr": bv.corr, "n": bv.n})
    ml.write_rows(out / "delta_corr.csv", dc_rows,
                  ["layer", "p1", "p2", "corr", "n"], ["corr"])
    print(f"  -> {out/'delta_corr.csv'} ({len(dc_rows)} rows over {len(buckets)} layers)")

    # ---- layer_sizes.csv ----
    ls_rows = []
    for b in buckets:
        ls_rows.append({
            "layer": b,
            "n_pairs": (n_pairs if b == "all" else layer_pairs.get(b, 0)),
            "core_heavy_mean": core_mom[b].mean if core_mom[b].n else "",
            "core_heavy_std": core_mom[b].std if core_mom[b].n else "",
            "frag_heavy_mean": fragsz_mom[b].mean if fragsz_mom[b].n else "",
            "frag_heavy_std": fragsz_mom[b].std if fragsz_mom[b].n else "",
            "frag_ratio_mean": ratio_mom[b].mean if ratio_mom[b].n else "",
            "frag_ratio_std": ratio_mom[b].std if ratio_mom[b].n else "",
            "frac_over_convention": "",
        })
    ml.write_rows(out / "layer_sizes.csv", ls_rows,
                  ["layer", "n_pairs", "core_heavy_mean", "core_heavy_std",
                   "frag_heavy_mean", "frag_heavy_std",
                   "frag_ratio_mean", "frag_ratio_std", "frac_over_convention"],
                  ["core_heavy_mean", "core_heavy_std", "frag_heavy_mean",
                   "frag_heavy_std", "frag_ratio_mean", "frag_ratio_std"])
    print(f"  -> {out/'layer_sizes.csv'}")
    for r in ls_rows:
        if r["frag_ratio_mean"] != "":
            flag = ("  <-- exceeds MMPA convention"
                    if r["frag_ratio_mean"] > config.SIZE_CONVENTION["max_frag_ratio"]
                    else "")
            print(f"     {r['layer']:12s} mean frag_ratio={r['frag_ratio_mean']:.3f}{flag}")


if __name__ == "__main__":
    main()
