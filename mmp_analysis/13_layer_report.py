#!/usr/bin/env python3
"""
13_layer_report.py -- layered summary + property-subset sensitivity (v4 stage 13).

Does two jobs, both purely from transformations.csv (no database access):

A. LAYER REPORT
   Per-layer counts, mirror-dedup counts, significance counts, effect-size
   distributions, and the fragment/core size audit against standard MMPA
   conventions. This is the table that answers "are your transformations too
   large to be matched pairs?" with measured numbers.

B. COMPOSITE SENSITIVITY
   Recomputes the gated composite under every preset in config.COMPOSITE_PRESETS
   (core / with_is / with_nu) and reports how much the top-rule ranking moves.
   Because per-property z_* and sig_* are stored per rule, this needs no
   re-stream. If the ranking is stable across presets, the conclusions do not
   depend on the out-of-domain h50 mapping; if it is not, that instability is
   itself a result and belongs in the SI.

Outputs (config.RESULTS):
  layer_report.csv          one row per layer
  composite_sensitivity.csv one row per (layer, preset)
  top_rules_by_layer.csv    deduplicated top rules within each layer

Usage:
  <tartarus python> 13_layer_report.py
  <tartarus python> 13_layer_report.py --top 30
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import config
import layers as L
import mmpa_lib as ml


def dedup(df):
    """Collapse mirror rows; keep the canonical direction if present."""
    if "undirected_key" not in df.columns:
        df = df.copy()
        df["undirected_key"] = [L.undirected_key(a, b, ml.canon_fragment)
                                for a, b in zip(df.frag_from, df.frag_to)]
    if "direction" in df.columns and (df.direction == "canonical").any():
        return df[df.direction == "canonical"].copy()
    return (df.sort_values("composite_gated", ascending=False)
              .drop_duplicates("undirected_key", keep="first").copy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transformations",
                    default=str(config.RESULTS / "transformations.csv"))
    ap.add_argument("--properties", nargs="+", default=config.PROPERTIES)
    ap.add_argument("--top", type=int, default=25,
                    help="Top rules per layer to write out / compare rankings on.")
    ap.add_argument("--output-dir", default=str(config.RESULTS))
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.transformations)
    props = [p for p in args.properties if f"z_{p}" in df.columns]
    print(f"Loaded {len(df)} directed rows; properties present: {props}")

    if "layer" not in df.columns:
        lay = [L.assign_layer(a, b) for a, b in zip(df.frag_from, df.frag_to)]
        df["layer"] = [x[0] for x in lay]
        df["cross_layer"] = [x[1] for x in lay]

    und = dedup(df)
    print(f"Undirected rules after mirror dedup: {len(und)}")

    conv = config.SIZE_CONVENTION
    layers_present = [l for l in L.LAYERS if (und.layer == l).any()]

    # ---------------- A. layer report ------------------------------------
    rep = []
    for lay in layers_present:
        sub = und[und.layer == lay]
        sig_any = np.zeros(len(sub), dtype=bool)
        for p in props:
            if f"sig_{p}" in sub.columns:
                sig_any |= sub[f"sig_{p}"].astype(str).str.lower().eq("true").values
        ratio = pd.to_numeric(sub.get("frag_ratio"), errors="coerce")
        cof = pd.to_numeric(sub.get("core_over_frag"), errors="coerce")
        gated = pd.to_numeric(sub.composite_gated, errors="coerce")
        rep.append({
            "layer": lay,
            "n_rules_undirected": len(sub),
            "n_rules_with_sig_prop": int(sig_any.sum()),
            "frac_sig": float(sig_any.mean()) if len(sub) else "",
            "n_cross_layer": int(sub.get("cross_layer", pd.Series(dtype=bool))
                                 .astype(str).str.lower().eq("true").sum()),
            "median_n_pairs": float(sub.n_pairs.median()) if len(sub) else "",
            "median_frag_heavy": float(pd.to_numeric(
                sub.get("frag_heavy_max"), errors="coerce").median())
                if "frag_heavy_max" in sub else "",
            "median_core_heavy": float(pd.to_numeric(
                sub.get("core_heavy"), errors="coerce").median())
                if "core_heavy" in sub else "",
            "median_frag_ratio": float(ratio.median()) if ratio.notna().any() else "",
            "frac_over_ratio_conv": float((ratio > conv["max_frag_ratio"]).mean())
                if ratio.notna().any() else "",
            "frac_under_core_conv": float((cof < conv["min_core_over_frag"]).mean())
                if cof.notna().any() else "",
            "gated_p50": float(gated.median()) if gated.notna().any() else "",
            "gated_p95": float(gated.quantile(0.95)) if gated.notna().any() else "",
            "gated_max": float(gated.max()) if gated.notna().any() else "",
        })
    fields = list(rep[0]) if rep else []
    ml.write_rows(out / "layer_report.csv", rep, fields,
                  [f for f in fields if f.startswith(("frac", "median", "gated"))])
    print(f"  -> {out/'layer_report.csv'}")
    for r in rep:
        warn = ""
        if r["frac_over_ratio_conv"] != "" and r["frac_over_ratio_conv"] > 0.5:
            warn = f"  <-- {r['frac_over_ratio_conv']:.0%} exceed frag_ratio {conv['max_frag_ratio']}"
        print(f"  {r['layer']:12s} n={r['n_rules_undirected']:6d} "
              f"sig={r['frac_sig'] if r['frac_sig']=='' else format(r['frac_sig'],'.1%'):>6} "
              f"median_ratio={r['median_frag_ratio']}{warn}")

    # ---------------- B. composite sensitivity ---------------------------
    presets = {name: [p for p in plist if p in props]
               for name, plist in config.COMPOSITE_PRESETS.items()}
    sens_rows = []
    ranks = {}
    for lay in layers_present:
        sub = und[und.layer == lay].copy()
        if not len(sub):
            continue
        comp = {}
        for name, plist in presets.items():
            vals = sub.apply(lambda r: L.recompute_composite(r, plist)["composite_gated"],
                             axis=1)
            comp[name] = vals.values
            sub[f"gated_{name}"] = vals.values
        base = config.DEFAULT_COMPOSITE
        for name in presets:
            v, b = comp[name], comp[base]
            ok = np.isfinite(v) & np.isfinite(b)
            rho = spearmanr(v[ok], b[ok])[0] if ok.sum() > 2 else float("nan")
            topn_base = set(sub.nlargest(args.top, f"gated_{base}").undirected_key)
            topn_this = set(sub.nlargest(args.top, f"gated_{name}").undirected_key)
            jac = (len(topn_base & topn_this) / len(topn_base | topn_this)
                   if topn_base | topn_this else float("nan"))
            sens_rows.append({
                "layer": lay, "preset": name,
                "properties": "|".join(presets[name]),
                "n_rules": len(sub),
                "spearman_vs_default": rho,
                f"top{args.top}_jaccard_vs_default": jac,
                "gated_mean": float(np.nanmean(v)),
                "gated_max": float(np.nanmax(v)) if np.isfinite(v).any() else "",
            })
        ranks[lay] = sub

    jf = f"top{args.top}_jaccard_vs_default"
    ml.write_rows(out / "composite_sensitivity.csv", sens_rows,
                  ["layer", "preset", "properties", "n_rules",
                   "spearman_vs_default", jf, "gated_mean", "gated_max"],
                  ["spearman_vs_default", jf, "gated_mean", "gated_max"])
    print(f"  -> {out/'composite_sensitivity.csv'}")
    print(f"\nRanking stability vs '{config.DEFAULT_COMPOSITE}' preset:")
    for r in sens_rows:
        if r["preset"] == config.DEFAULT_COMPOSITE:
            continue
        print(f"  {r['layer']:12s} {r['preset']:10s} "
              f"spearman={r['spearman_vs_default']:.3f}  "
              f"top{args.top} overlap={r[jf]:.2f}")

    # ---------------- top rules per layer --------------------------------
    keep = ["layer", "frag_from", "frag_to", "chemotype_from", "chemotype_to",
            "n_pairs", "frag_ratio", "composite_gated", "n_improve", "n_worsen",
            "has_tradeoff"] + [f"z_{p}" for p in props] + \
           [f"gated_{n}" for n in presets]
    top_rows = []
    for lay, sub in ranks.items():
        for _, r in sub.nlargest(args.top, "composite_gated").iterrows():
            top_rows.append({k: r.get(k, "") for k in keep})
    ml.write_rows(out / "top_rules_by_layer.csv", top_rows, keep,
                  [k for k in keep if k.startswith(("z_", "gated_", "composite",
                                                    "frag_ratio"))])
    print(f"  -> {out/'top_rules_by_layer.csv'} ({len(top_rows)} rows)")


if __name__ == "__main__":
    main()
