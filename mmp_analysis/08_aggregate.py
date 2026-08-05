#!/usr/bin/env python3
"""
08_aggregate.py -- direction-free aggregations of the anchored pairs (v3 stage 8).

Supporting/coverage layer: per-candidate neighborhood statistics and local
tradeoff structure from stage 06's anchored_pairs.csv. All outputs land under
config.AGG_DIR (results/aggregated/) -- fixing the v2 path drift where some
tables were written to results/ and broke downstream figures.

Tables (none assume a "better" direction):
  headline.csv       one row per candidate: n, mean/median/std/frac_pos of each
                     Δ; plus pair breadth (unique neighbors, cand-frags)
  per_fragment.csv   one row per (candidate, candidate_fragment)
  tradeoffs.csv      per-candidate Pearson r of (Δ_P1, Δ_P2)
  cand_vs_cand.csv   anchored pairs whose neighbor is also a candidate

Usage:
  <tartarus python> 08_aggregate.py
"""
import argparse
from pathlib import Path
from collections import defaultdict
from itertools import combinations

import numpy as np

import config
import mmpa_lib as ml


def load_pairs(path, properties):
    rows = ml.read_rows(path, numeric=[f"d_{p}" for p in properties])
    for r in rows:
        r["neighbor_is_candidate"] = str(r.get("neighbor_is_candidate", "")).lower() == "true"
    return rows


def stats(values):
    arr = np.array([v for v in values if v is not None], dtype=float)
    n = len(arr)
    if n == 0:
        return 0, "", "", "", ""
    return (n, float(arr.mean()), float(np.median(arr)),
            float(arr.std(ddof=1)) if n > 1 else 0.0, float((arr > 0).mean()))


def build_headline(pairs, properties):
    by = defaultdict(list)
    for p in pairs:
        by[p["candidate_id"]].append(p)
    rows = []
    for cand, plist in by.items():
        row = {"candidate_id": cand, "n_pairs_total": len(plist),
               "n_pairs_vs_candidate": sum(1 for x in plist if x["neighbor_is_candidate"]),
               "n_pairs_vs_database": sum(1 for x in plist if not x["neighbor_is_candidate"]),
               "n_unique_neighbors": len(set(x["neighbor_id"] for x in plist)),
               "n_unique_cand_frags": len(set(x["candidate_frag"] for x in plist))}
        for p in properties:
            n, mean, median, std, fpos = stats([x[f"d_{p}"] for x in plist])
            row[f"n_{p}"], row[f"mean_d_{p}"], row[f"median_d_{p}"] = n, mean, median
            row[f"std_d_{p}"], row[f"frac_pos_d_{p}"] = std, fpos
        rows.append(row)
    rows.sort(key=lambda r: r["candidate_id"])
    return rows


def build_per_fragment(pairs, properties):
    by = defaultdict(list)
    for p in pairs:
        by[(p["candidate_id"], p["candidate_frag"])].append(p)
    rows = []
    for (cand, frag), plist in by.items():
        row = {"candidate_id": cand, "candidate_frag": frag,
               "chemotype": ml.primary_chemotype(frag), "n_pairs": len(plist),
               "n_distinct_neighbor_frags": len(set(x["neighbor_frag"] for x in plist))}
        for p in properties:
            n, mean, median, std, fpos = stats([x[f"d_{p}"] for x in plist])
            row[f"n_{p}"], row[f"mean_d_{p}"], row[f"median_d_{p}"] = n, mean, median
            row[f"std_d_{p}"], row[f"frac_pos_d_{p}"] = std, fpos
        rows.append(row)
    rows.sort(key=lambda r: (r["candidate_id"], -r["n_pairs"]))
    return rows


def build_tradeoffs(pairs, properties, min_pairs):
    by = defaultdict(list)
    for p in pairs:
        by[p["candidate_id"]].append(p)
    rows = []
    prop_pairs = list(combinations(properties, 2))
    for cand, plist in by.items():
        if len(plist) < min_pairs:
            continue
        row = {"candidate_id": cand, "n_pairs": len(plist)}
        for p1, p2 in prop_pairs:
            xs, ys = [], []
            for x in plist:
                v1, v2 = x[f"d_{p1}"], x[f"d_{p2}"]
                if v1 is None or v2 is None:
                    continue
                xs.append(v1); ys.append(v2)
            if len(xs) < min_pairs or np.std(xs) == 0 or np.std(ys) == 0:
                row[f"corr_{p1}_{p2}"] = ""
            else:
                row[f"corr_{p1}_{p2}"] = float(np.corrcoef(xs, ys)[0, 1])
            row[f"n_{p1}_{p2}"] = len(xs)
        rows.append(row)
    rows.sort(key=lambda r: r["candidate_id"])
    return rows, prop_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=str(config.RESULTS / "anchored_pairs.csv"))
    ap.add_argument("--properties", nargs="+", default=config.PROPERTIES)
    ap.add_argument("--min-tradeoff-pairs", type=int, default=10)
    ap.add_argument("--output-dir", default=str(config.AGG_DIR))
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    props = args.properties

    print(f"Loading anchored pairs from {args.pairs}...")
    pairs = load_pairs(args.pairs, props)
    print(f"  {len(pairs)} pairs")

    rows = build_headline(pairs, props)
    fns = ["candidate_id", "n_pairs_total", "n_pairs_vs_candidate",
           "n_pairs_vs_database", "n_unique_neighbors", "n_unique_cand_frags"]
    floats = []
    for p in props:
        fns += [f"n_{p}", f"mean_d_{p}", f"median_d_{p}", f"std_d_{p}", f"frac_pos_d_{p}"]
        floats += [f"mean_d_{p}", f"median_d_{p}", f"std_d_{p}", f"frac_pos_d_{p}"]
    ml.write_rows(out / "headline.csv", rows, fns, floats)
    print(f"  -> {out/'headline.csv'} ({len(rows)} candidates)")

    rows = build_per_fragment(pairs, props)
    fns = ["candidate_id", "candidate_frag", "chemotype", "n_pairs",
           "n_distinct_neighbor_frags"]
    floats = []
    for p in props:
        fns += [f"n_{p}", f"mean_d_{p}", f"median_d_{p}", f"std_d_{p}", f"frac_pos_d_{p}"]
        floats += [f"mean_d_{p}", f"median_d_{p}", f"std_d_{p}", f"frac_pos_d_{p}"]
    ml.write_rows(out / "per_fragment.csv", rows, fns, floats)
    print(f"  -> {out/'per_fragment.csv'} ({len(rows)} rows)")

    rows, prop_pairs = build_tradeoffs(pairs, props, args.min_tradeoff_pairs)
    fns = ["candidate_id", "n_pairs"]
    floats = []
    for p1, p2 in prop_pairs:
        fns += [f"corr_{p1}_{p2}", f"n_{p1}_{p2}"]
        floats += [f"corr_{p1}_{p2}"]
    ml.write_rows(out / "tradeoffs.csv", rows, fns, floats)
    print(f"  -> {out/'tradeoffs.csv'} ({len(rows)} candidates >= {args.min_tradeoff_pairs} pairs)")

    cc = [p for p in pairs if p["neighbor_is_candidate"]]
    fns = ["pair_id", "candidate_id", "neighbor_id", "context_smiles",
           "candidate_frag", "neighbor_frag"] + [f"d_{p}" for p in props]
    ml.write_rows(out / "cand_vs_cand.csv", cc, fns, [f"d_{p}" for p in props])
    print(f"  -> {out/'cand_vs_cand.csv'} ({len(cc)} pairs)")
    print(f"\nAll aggregates in {out}/")


if __name__ == "__main__":
    main()
