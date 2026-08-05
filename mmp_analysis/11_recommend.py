#!/usr/bin/env python3
"""
11_recommend.py -- direction-aware design recommendations (v4 stage 11).

v4: recommendations are ranked WITHIN each design layer (linker / scaffold /
substituent). A flat ranking is dominated by scaffold swaps simply because they
exchange more atoms; that is a size effect, not a chemistry result. Mirror rows
are collapsed on undirected_key before ranking, and support floors come from
config.MIN_SUPPORT_BY_LAYER.

Replaces the v2 recommenders (old 06_recommend / 10_composite), which grouped
by (candidate, frag_from, frag_to) and always found n=1 -> empty output. v3
instead ranks the GLOBALLY supported transformations that a candidate can
actually apply (role = actionable in the stage-07 rule x endpoint matrix), so
support and significance come from the 108k index, not a single local pair.

For each candidate we emit its best actionable edits A -> B with:
  n_pairs (global support), per-property z + BH-significance, gated composite,
  tradeoff flag, and local anchored confirmation where it exists.

Outputs (config.RESULTS):
  recommendations.csv        one row per (candidate, recommended edit)

Usage:
  <tartarus python> 11_recommend.py
  <tartarus python> 11_recommend.py --top-per-candidate 5 --allow-tradeoffs
"""
import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd

import config
import layers as L
import mmpa_lib as ml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default=str(config.RESULTS / "candidate_rule_matrix.csv"))
    ap.add_argument("--properties", nargs="+", default=config.PROPERTIES)
    ap.add_argument("--min-support", type=int, default=config.MIN_RULE_SUPPORT)
    ap.add_argument("--min-significant", type=int, default=1,
                    help="Require at least this many BH-significant improved props.")
    ap.add_argument("--allow-tradeoffs", action="store_true",
                    help="Include edits that also significantly worsen a property.")
    ap.add_argument("--top-per-candidate", type=int, default=5,
                    help="Top edits per candidate PER LAYER.")
    ap.add_argument("--per-layer-support", action="store_true", default=True,
                    help="Use config.MIN_SUPPORT_BY_LAYER instead of a flat floor.")
    ap.add_argument("--dedup", action="store_true", default=True,
                    help="Collapse mirror rows on undirected_key before ranking.")
    ap.add_argument("--output", default=str(config.RESULTS / "recommendations.csv"))
    args = ap.parse_args()

    m = pd.read_csv(args.matrix)
    m = m[m["role"] == "actionable"].copy()
    props = [p for p in args.properties if f"z_{p}" in m.columns]
    missing = [p for p in args.properties if p not in props]
    if missing:
        print(f"WARNING: properties absent from matrix, skipped: {missing}")
    if not props:
        raise SystemExit("ERROR: no usable properties in matrix")
    print(f"Using properties: {props}")
    if "layer" not in m.columns:
        m["layer"] = [L.assign_layer(a, b)[0]
                      for a, b in zip(m["frag_from"], m["frag_to"])]
    if args.per_layer_support:
        floor = m["layer"].map(config.MIN_SUPPORT_BY_LAYER).fillna(args.min_support)
        m = m[m["n_pairs"] >= floor]
        print("Per-layer support floors: " +
              ", ".join(f"{k}>={v}" for k, v in config.MIN_SUPPORT_BY_LAYER.items()))
    else:
        m = m[m["n_pairs"] >= args.min_support]
    if args.dedup and "undirected_key" in m.columns:
        before = len(m)
        m = (m.sort_values("composite_gated", ascending=False)
               .drop_duplicates(["candidate_id", "undirected_key"], keep="first"))
        print(f"Mirror dedup on undirected_key: {before} -> {len(m)} rows")
    print(f"Actionable (candidate, rule) rows after support gating: {len(m)}")

    # significance count = number of sig_* True with positive z
    def n_sig_improved(row):
        c = 0
        for p in props:
            if str(row.get(f"sig_{p}")).lower() == "true" and float(row.get(f"z_{p}", 0) or 0) > 0:
                c += 1
        return c

    m["n_sig_improved"] = m.apply(n_sig_improved, axis=1)
    m = m[m["n_sig_improved"] >= args.min_significant]
    if not args.allow_tradeoffs:
        m = m[m["n_worsen"] == 0]
    print(f"After significance/tradeoff gating: {len(m)}")

    m = m.sort_values(["candidate_id", "layer", "composite_gated"],
                      ascending=[True, True, False])

    out_rows = []
    for (cid, lay), sub in m.groupby(["candidate_id", "layer"]):
        for _, r in sub.head(args.top_per_candidate).iterrows():
            edit = (f"{ml.strip_attachment(r['frag_from']) or 'H'} -> "
                    f"{ml.strip_attachment(r['frag_to']) or 'H'}")
            rec = {
                "candidate_id": cid,
                "layer": lay,
                "edit": edit,
                "frag_from": r["frag_from"], "frag_to": r["frag_to"],
                "chemotype_edit": f"{r['chemotype_from']} -> {r['chemotype_to']}",
                "global_n_support": int(r["n_pairs"]),
                "composite_gated": r["composite_gated"],
                "n_improve": int(r["n_improve"]), "n_worsen": int(r["n_worsen"]),
                "n_sig_improved": int(r["n_sig_improved"]),
                "has_tradeoff": bool(r["has_tradeoff"]),
                "has_local_evidence": bool(r["has_local_evidence"]),
            }
            for p in props:
                rec[f"z_{p}"] = r.get(f"z_{p}", "")
                rec[f"sig_{p}"] = r.get(f"sig_{p}", "")
                rec[f"local_mean_d_{p}"] = r.get(f"local_mean_d_{p}", "")
            out_rows.append(rec)

    fields = ["candidate_id", "layer", "edit", "frag_from", "frag_to", "chemotype_edit",
              "global_n_support", "composite_gated", "n_improve", "n_worsen",
              "n_sig_improved", "has_tradeoff", "has_local_evidence"]
    floats = ["composite_gated"]
    for p in props:
        fields += [f"z_{p}", f"sig_{p}", f"local_mean_d_{p}"]
        floats += [f"z_{p}", f"local_mean_d_{p}"]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    ml.write_rows(args.output, out_rows, fields, floats)
    n_cand = len({r["candidate_id"] for r in out_rows})
    print(f"Wrote {len(out_rows)} recommendations for {n_cand} candidates "
          f"-> {args.output}")
    if out_rows:
        print("\nExample (top composite):")
        top = sorted(out_rows, key=lambda r: -r["composite_gated"])[:5]
        for r in top:
            print(f"  {r['candidate_id']} [{r['layer']:11s}]: {r['chemotype_edit']:24s} "
                  f"gated={r['composite_gated']:+.2f} n={r['global_n_support']} "
                  f"nimp={r['n_improve']} local={r['has_local_evidence']}")


if __name__ == "__main__":
    main()
