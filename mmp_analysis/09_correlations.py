#!/usr/bin/env python3
"""
09_correlations.py -- Figure 1 data (v4 stage 9).

Pairs the population (static) correlations computed from the raw 108k property
values against the edit-level (delta) correlations from stage 03.

v4 change: stage 03 now emits delta correlations PER LAYER, so this stage emits
one comparison block per layer plus the pooled 'all'. The scientific question
it opens: does the population -> edit decoupling come from bridge chemistry,
ring swaps, or substituent edits? Those three have different physical causes
and there is no reason to expect one number to describe all of them.

Caveat to carry into the figure: delta correlations are attenuated by
measurement noise in a way static correlations are not (regression dilution),
and Var(dP) = 2 Var(P) under independent errors. A positive divergence is
therefore the null expectation, not evidence on its own. Compare against a
permutation null or an explicit noise-floor estimate before claiming
decoupling. Pairs whose properties are near a hard bound (e.g. nu -> 0.25) are
additionally heteroscedastic; consider excluding them as a robustness check.

Outputs:
  static_corr.csv   long-form Pearson + Spearman over raw property values
  corr_compare.csv  layer,p1,p2,static_pearson,static_spearman,delta,n_delta,
                    divergence

Usage:
  python 09_correlations.py --scope all
  python 09_correlations.py --properties Hf Q OBgood Pe D P nu
"""
import argparse
from itertools import combinations
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr, spearmanr

import config
import layers as L
import mmpa_lib as ml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default=str(config.MASTER_CSV))
    ap.add_argument("--delta-corr", default=str(config.RESULTS / "delta_corr.csv"))
    ap.add_argument("--properties", nargs="+", default=config.PROPERTIES)
    ap.add_argument("--scope", choices=["all", "database", "candidates"],
                    default="all")
    ap.add_argument("--output-dir", default=str(config.RESULTS))
    args = ap.parse_args()

    props = args.properties
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    m = pd.read_csv(args.master)
    if args.scope == "database":
        m = m[m[config.COL_IS_CANDIDATE] == 0]
    elif args.scope == "candidates":
        m = m[m[config.COL_IS_CANDIDATE] == 1]

    present = [p for p in props if p in m.columns]
    missing = [p for p in props if p not in m.columns]
    if missing:
        print(f"WARNING: not in master.csv, skipped: {missing}")
    props = present

    static_rows = []
    static_p, static_s = {}, {}
    for a, b in combinations(props, 2):
        sub = m[[a, b]].dropna()
        if len(sub) >= 3:
            pr = pearsonr(sub[a], sub[b])[0]
            sp = spearmanr(sub[a], sub[b])[0]
        else:
            pr = sp = float("nan")
        static_rows.append({"p1": a, "p2": b, "pearson": pr,
                            "spearman": sp, "n": len(sub)})
        static_p[(a, b)] = pr
        static_s[(a, b)] = sp
    ml.write_rows(out / "static_corr.csv", static_rows,
                  ["p1", "p2", "pearson", "spearman", "n"],
                  ["pearson", "spearman"])
    print(f"  -> {out/'static_corr.csv'} (scope={args.scope}, n_mols={len(m)})")

    delta = pd.read_csv(args.delta_corr)
    if "layer" not in delta.columns:
        delta["layer"] = "all"          # tolerate a v3 delta_corr.csv
    buckets = [b for b in ("all",) + tuple(L.LAYERS)
               if b in set(delta["layer"])]

    cmp_rows = []
    for lay in buckets:
        d = delta[delta["layer"] == lay]
        dl = {(r["p1"], r["p2"]): (r["corr"], r["n"]) for _, r in d.iterrows()}
        for a, b in combinations(props, 2):
            s = static_p.get((a, b))
            dv, dn = dl.get((a, b), (None, None))
            div = (dv - s) if (s is not None and dv is not None
                               and pd.notna(s) and pd.notna(dv)) else None
            cmp_rows.append({"layer": lay, "p1": a, "p2": b,
                             "static_pearson": s,
                             "static_spearman": static_s.get((a, b)),
                             "delta": dv, "n_delta": dn, "divergence": div})
    ml.write_rows(out / "corr_compare.csv", cmp_rows,
                  ["layer", "p1", "p2", "static_pearson", "static_spearman",
                   "delta", "n_delta", "divergence"],
                  ["static_pearson", "static_spearman", "delta", "divergence"])
    print(f"  -> {out/'corr_compare.csv'} ({len(buckets)} layers)")

    cc = pd.DataFrame(cmp_rows)
    for lay in buckets:
        sub = cc[(cc.layer == lay)].dropna(subset=["divergence"])
        if not len(sub):
            continue
        top = sub.reindex(sub.divergence.abs().sort_values(ascending=False).index).head(3)
        s = "; ".join(f"{r.p1}-{r.p2} {r.divergence:+.3f}" for r in top.itertuples())
        print(f"  {lay:12s} largest |divergence|: {s}")


if __name__ == "__main__":
    main()
