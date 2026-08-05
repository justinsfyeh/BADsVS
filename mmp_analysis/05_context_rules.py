#!/usr/bin/env python3
"""
05_context_rules.py -- OPTIONAL context-conditional rules (v3 stage 5).

Stage 04 measures each rule A->B at radius 0 (environment-agnostic = the
transferable effect). This stage asks the complementary question:

    "Is A->B favorable everywhere, or only in certain local scaffolds?"

Mechanics: mmpdb precomputes every rule at environment radii 0..5. At radius R
a single rule splits into several `rule_environment` rows, one per distinct
local environment. We re-examine the top rules at radius R, compute the effect
per environment (gated by MIN_CONTEXT_SUPPORT pairs), and flag environments
whose directed effect departs from the rule's global (radius-0) mean -- e.g.
sign flips. Because the environment fingerprint is an opaque hash, each context
is NAMED by the chemotypes of its member constant (scaffold) SMILES.

Support decays fast with radius, so single-pair environments are treated as
anecdotes, never rules.

Output (config.RESULTS):
  context_conditional_rules.csv

Usage:
  <tartarus python> 05_context_rules.py                 # config defaults
  <tartarus python> 05_context_rules.py --radius 2 --top-rules 150
"""
import argparse
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import pandas as pd

import config
import mmpa_lib as ml


def top_rule_ids(conn, radius0, top_rules, min_support):
    """Return list of (rule_id, from_smiles, to_smiles, n0) for the highest-
    support rules at radius 0 (candidates for context conditioning)."""
    q = f"""
    SELECT re.rule_id AS rid, rs_from.smiles AS ffrom, rs_to.smiles AS fto,
           count(*) AS n0
    FROM pair p
    JOIN rule_environment re ON p.rule_environment_id = re.id
    JOIN rule r ON re.rule_id = r.id
    JOIN rule_smiles rs_from ON r.from_smiles_id = rs_from.id
    JOIN rule_smiles rs_to   ON r.to_smiles_id   = rs_to.id
    WHERE re.radius = {radius0}
    GROUP BY re.rule_id
    HAVING n0 >= {min_support}
    ORDER BY n0 DESC
    LIMIT {top_rules}
    """
    return [(r["rid"], r["ffrom"], r["fto"], r["n0"]) for r in conn.execute(q)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(config.DB))
    ap.add_argument("--properties", nargs="+", default=config.PROPERTIES)
    ap.add_argument("--radius", type=int, default=2,
                    help="Environment radius at which to condition (default 2).")
    ap.add_argument("--top-rules", type=int, default=150,
                    help="Number of highest-support radius-0 rules to examine.")
    ap.add_argument("--min-context-support", type=int,
                    default=config.MIN_CONTEXT_SUPPORT)
    ap.add_argument("--global-scales", default=str(config.RESULTS / "global_scales.csv"))
    ap.add_argument("--output", default=str(config.RESULTS / "context_conditional_rules.csv"))
    args = ap.parse_args()

    props = args.properties
    direction = config.directions()

    s_global = {p: 1.0 for p in props}
    gs_path = Path(args.global_scales)
    if gs_path.exists():
        gs = pd.read_csv(gs_path).set_index("property")
        for p in props:
            if p in gs.index and gs.loc[p, "std"] > 0:
                s_global[p] = float(gs.loc[p, "std"])

    conn = ml.connect(args.db)
    prop_ids = ml.property_name_ids(conn, props)
    props = list(prop_ids.keys())
    pv = ml.load_property_values(conn, prop_ids)

    rules = top_rule_ids(conn, config.RADIUS_GENERAL, args.top_rules,
                         config.MIN_RULE_SUPPORT)
    print(f"Examining {len(rules)} top rules at radius {args.radius}")

    # global (radius-0) directed mean per rule/property for divergence reference
    out_rows = []
    for rid, ffrom, fto, n0 in rules:
        # radius-0 global means
        g_mean = {p: ml.Moments() for p in props}
        for r in conn.execute(
                "SELECT p.compound1_id c1, p.compound2_id c2 "
                "FROM pair p JOIN rule_environment re ON p.rule_environment_id=re.id "
                "WHERE re.rule_id=? AND re.radius=?", (rid, config.RADIUS_GENERAL)):
            for p in props:
                a = pv.get((r["c1"], p)); b = pv.get((r["c2"], p))
                if a is not None and b is not None:
                    g_mean[p].add(b - a)

        # per-environment at radius R
        env_pairs = defaultdict(list)   # env_id -> [(c1,c2), ...]
        env_ctx = defaultdict(list)     # env_id -> [constant_smiles,...]
        for r in conn.execute(
                "SELECT p.compound1_id c1, p.compound2_id c2, re.id env, cs.smiles ctx "
                "FROM pair p JOIN rule_environment re ON p.rule_environment_id=re.id "
                "LEFT JOIN constant_smiles cs ON p.constant_id=cs.id "
                "WHERE re.rule_id=? AND re.radius=?", (rid, args.radius)):
            env_pairs[r["env"]].append((r["c1"], r["c2"]))
            if r["ctx"] is not None:
                env_ctx[r["env"]].append(r["ctx"])

        for env_id, pairs in env_pairs.items():
            if len(pairs) < args.min_context_support:
                continue
            rec = {"rule_id": rid, "frag_from": ffrom, "frag_to": fto,
                   "chemotype_from": ml.primary_chemotype(ffrom),
                   "chemotype_to": ml.primary_chemotype(fto),
                   "radius": args.radius, "env_id": env_id,
                   "n_pairs_env": len(pairs), "n_pairs_global": n0}
            # characterize context by chemotypes of member scaffolds
            tags = Counter()
            for ctx in env_ctx.get(env_id, [])[:50]:
                for t in ml.chemotype_tags(ctx):
                    tags[t] += 1
            rec["context_chemotypes"] = ";".join(
                f"{t}:{c}" for t, c in tags.most_common(4)) or "unclassified"
            rec["context_example"] = (env_ctx.get(env_id, [""])[0])[:60]

            n_flip = 0
            for p in props:
                m = ml.Moments()
                for c1, c2 in pairs:
                    a = pv.get((c1, p)); b = pv.get((c2, p))
                    if a is not None and b is not None:
                        m.add(b - a)
                env_mean = m.mean if m.n else float("nan")
                glob_mean = g_mean[p].mean if g_mean[p].n else float("nan")
                rec[f"mean_d_{p}"] = env_mean
                rec[f"z_{p}"] = (direction[p] * env_mean / s_global[p]
                                 if not np.isnan(env_mean) else float("nan"))
                # sign flip vs global directed effect
                if (not np.isnan(env_mean) and not np.isnan(glob_mean)
                        and env_mean != 0 and glob_mean != 0
                        and np.sign(direction[p]*env_mean) != np.sign(direction[p]*glob_mean)):
                    n_flip += 1
            rec["n_props_sign_flip_vs_global"] = n_flip
            out_rows.append(rec)

    out_rows.sort(key=lambda r: (-r["n_props_sign_flip_vs_global"], -r["n_pairs_env"]))

    fields = ["rule_id", "frag_from", "frag_to", "chemotype_from", "chemotype_to",
              "radius", "env_id", "n_pairs_env", "n_pairs_global",
              "context_chemotypes", "context_example",
              "n_props_sign_flip_vs_global"]
    floats = []
    for p in props:
        fields += [f"mean_d_{p}", f"z_{p}"]
        floats += [f"mean_d_{p}", f"z_{p}"]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    ml.write_rows(args.output, out_rows, fields, floats)
    print(f"Wrote {len(out_rows)} context-conditional rows -> {args.output}")
    n_flip = sum(1 for r in out_rows if r["n_props_sign_flip_vs_global"] > 0)
    print(f"  ({n_flip} environments show >=1 sign flip vs the global rule effect)")


if __name__ == "__main__":
    main()
