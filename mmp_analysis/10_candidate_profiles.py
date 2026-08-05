#!/usr/bin/env python3
"""
10_candidate_profiles.py -- candidate decomposition & benchmarking (v3 stage 10).

For every candidate:
  * fragment inventory F(c) joined to global fragment-goodness -> standardized
    z_P(v) = d_P * A_P(v) / s_P^global
  * generalist/specialist scalars (secondary, fragment-goodness view)
  * subset membership: on the candidate Pareto front, and beats the BENCHMARK
    on every objective
  * regression-to-the-mean table (guards against reading neighborhood means as
    causal superiority)

Benchmark policy comes from config.BENCHMARK (via mmpa_lib.load_benchmark):
  - external: a CSV of reference-molecule property values
  - internal: top-N molecules (default top-10 candidates by curated `rank`)
  aggregated (median/mean/best/percentile) into a reference vector.

Outputs (config.RESULTS):
  candidate_fragment_profiles.csv, candidate_scalars.csv, regression_to_mean.csv

Usage:
  <tartarus python> 10_candidate_profiles.py
  <tartarus python> 10_candidate_profiles.py --benchmark-source external \\
      --benchmark-csv refs.csv
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import config
import mmpa_lib as ml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default=str(config.MASTER_CSV))
    ap.add_argument("--candidates-csv", default=str(config.CANDIDATES_CSV))
    ap.add_argument("--anchored", default=str(config.RESULTS / "anchored_pairs.csv"))
    ap.add_argument("--fragment-goodness", default=str(config.RESULTS / "fragment_goodness.csv"))
    ap.add_argument("--global-scales", default=str(config.RESULTS / "global_scales.csv"))
    ap.add_argument("--headline", default=str(config.AGG_DIR / "headline.csv"))
    ap.add_argument("--properties", nargs="+", default=config.PROPERTIES)
    ap.add_argument("--min-frag-pairs", type=int, default=3)
    # benchmark overrides (else config.BENCHMARK)
    ap.add_argument("--benchmark-source", choices=["internal", "external"], default=None)
    ap.add_argument("--benchmark-csv", default=None)
    ap.add_argument("--benchmark-top-n", type=int, default=None)
    ap.add_argument("--benchmark-agg", default=None)
    ap.add_argument("--output-dir", default=str(config.RESULTS))
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    master = pd.read_csv(args.master)
    master[config.COL_ID] = master[config.COL_ID].astype(str)

    gs = pd.read_csv(args.global_scales).set_index("property")
    fg = pd.read_csv(args.fragment_goodness)
    fg["fragment"] = fg["fragment"].astype(str)

    # Drop properties absent from DB-derived tables (e.g. nu not indexed).
    props = []
    for p in args.properties:
        if f"A_{p}" not in fg.columns:
            print(f"WARNING: property '{p}' absent from fragment_goodness; skipped")
            continue
        if p not in gs.index:
            print(f"WARNING: property '{p}' absent from global_scales; skipped")
            continue
        if p not in master.columns:
            print(f"WARNING: property '{p}' absent from master.csv; skipped")
            continue
        props.append(p)
    if not props:
        raise SystemExit("ERROR: no usable properties left after filtering")
    print(f"Using properties: {props}")

    direction = config.directions()
    s_global = {p: (gs.loc[p, "std"] if gs.loc[p, "std"] > 0 else 1.0)
                for p in props}

    A, npairs_frag = {}, {}
    for _, r in fg.iterrows():
        f = r["fragment"]
        A[f] = {p: (r[f"A_{p}"] if not pd.isna(r[f"A_{p}"]) else None) for p in props}
        npairs_frag[f] = int(r["n_total_pairs"]) if not pd.isna(r["n_total_pairs"]) else 0

    def zf(frag, p):
        a = A.get(frag, {}).get(p)
        return None if a is None else direction[p] * a / s_global[p]

    anchored = ml.read_rows(args.anchored)
    Fc = defaultdict(set)
    for r in anchored:
        Fc[r["candidate_id"]].add(r["candidate_frag"])

    # ---- candidate_fragment_profiles.csv ----
    prof_fields = ["candidate_id", "fragment", "chemotype", "n_frag_pairs"] + [f"z_{p}" for p in props]
    prof_rows = []
    for cand in sorted(Fc):
        for frag in Fc[cand]:
            if npairs_frag.get(frag, 0) < args.min_frag_pairs:
                continue
            row = {"candidate_id": cand, "fragment": frag,
                   "chemotype": ml.primary_chemotype(frag),
                   "n_frag_pairs": npairs_frag.get(frag, 0)}
            for p in props:
                row[f"z_{p}"] = zf(frag, p)
            prof_rows.append(row)
    ml.write_rows(out / "candidate_fragment_profiles.csv", prof_rows, prof_fields,
                  [f"z_{p}" for p in props])
    print(f"  -> {out/'candidate_fragment_profiles.csv'} ({len(prof_rows)} rows)")

    by_cand = defaultdict(list)
    for r in prof_rows:
        by_cand[r["candidate_id"]].append(r)

    # ---- benchmark ----
    spec = dict(config.BENCHMARK)
    if args.benchmark_source: spec["source"] = args.benchmark_source
    if args.benchmark_csv: spec["source"], spec["external_csv"] = "external", args.benchmark_csv
    if args.benchmark_top_n: spec["top_n"] = args.benchmark_top_n
    if args.benchmark_agg: spec["agg"] = args.benchmark_agg
    bench, bench_ids, bench_desc = ml.load_benchmark(
        master, props, direction, spec, args.candidates_csv)
    print(f"Benchmark: {bench_desc} -> " +
          ", ".join(f"{p}={bench[p]:.3g}" for p in props))

    cand_master = master[master[config.COL_IS_CANDIDATE] == 1].copy()
    cand_ids, cand_vecs, cand_valmap = [], [], {}
    for _, r in cand_master.iterrows():
        cid = str(r[config.COL_ID])
        cand_ids.append(cid)
        cand_vecs.append(np.array([direction[p] * float(r[p]) for p in props]))
        cand_valmap[cid] = {p: float(r[p]) for p in props}
    nd = ml.pareto_front(cand_vecs)
    on_pareto = {cid: bool(f) for cid, f in zip(cand_ids, nd)}
    beats = {cid: ml.beats_benchmark(cand_valmap[cid], bench, props, direction)
             for cid in cand_ids}
    scorecard = {cid: ml.benchmark_scorecard(cand_valmap[cid], bench, s_global,
                                             props, direction) for cid in cand_ids}

    scal_fields = (["candidate_id", "generalist", "specialist", "n_fragments",
                    "on_pareto", "beats_benchmark", "n_obj_beat_benchmark",
                    "composite_vs_benchmark"] + [f"val_{p}" for p in props])
    scal_rows = []
    for cid in cand_ids:
        frags = by_cand.get(cid, [])
        gen = spec_s = None
        if frags:
            zmat = np.array([[r[f"z_{p}"] if r[f"z_{p}"] is not None else np.nan
                              for p in props] for r in frags], dtype=float)
            if np.isfinite(zmat).any():
                gen = float(np.nanmin(np.nanmin(zmat, axis=1)))
                spec_s = float(np.nanmin(np.nanmax(zmat, axis=0)))
        n_beat, comp = scorecard[cid]
        row = {"candidate_id": cid, "generalist": gen, "specialist": spec_s,
               "n_fragments": len(frags), "on_pareto": on_pareto.get(cid, False),
               "beats_benchmark": beats.get(cid, False),
               "n_obj_beat_benchmark": n_beat, "composite_vs_benchmark": comp}
        for p in props:
            row[f"val_{p}"] = cand_valmap[cid][p]
        scal_rows.append(row)
    ml.write_rows(out / "candidate_scalars.csv", scal_rows, scal_fields,
                  ["generalist", "specialist", "composite_vs_benchmark"]
                  + [f"val_{p}" for p in props])
    n_partial = sum(1 for cid in cand_ids if scorecard[cid][0] >= len(props) - 1)
    print(f"  -> {out/'candidate_scalars.csv'} (pareto={sum(on_pareto.values())}, "
          f"beats_all={sum(beats.values())}, beats>={len(props)-1}obj: {n_partial})")

    # ---- regression-to-mean ----
    headline = ml.read_rows(args.headline, numeric=[f"mean_d_{p}" for p in props])
    head_by = {r["candidate_id"]: r for r in headline}
    db = master[master[config.COL_IS_CANDIDATE] == 0]
    db_mean = {p: float(db[p].mean()) for p in props}
    rtm_rows = []
    for cid in cand_ids:
        h = head_by.get(cid)
        for p in props:
            cv = cand_valmap[cid][p]
            md = h.get(f"mean_d_{p}") if h else None
            rtm_rows.append({"candidate_id": cid, "property": p, "cand_value": cv,
                             "db_mean": db_mean[p], "centered": cv - db_mean[p],
                             "mean_delta": md})
    ml.write_rows(out / "regression_to_mean.csv", rtm_rows,
                  ["candidate_id", "property", "cand_value", "db_mean",
                   "centered", "mean_delta"],
                  ["cand_value", "db_mean", "centered", "mean_delta"])
    print(f"  -> {out/'regression_to_mean.csv'}")


if __name__ == "__main__":
    main()
