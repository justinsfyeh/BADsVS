#!/usr/bin/env python3
"""
06_extract_pairs.py -- candidate-anchored pairs (v3 stage 6).

For each candidate c, finds every matched pair (c, m) in the index and writes
one row per anchored pair with deltas in the CANDIDATE -> NEIGHBOR direction:

    delta_P = P_neighbor - P_candidate

In v3 this is the candidate-LOCAL evidence layer (case studies + coverage), not
the discovery engine. Each row now also carries the rule it instantiates and
that rule's GLOBAL support, so a candidate observation can be read against the
108k-level evidence for the same edit.

Fixes carried over from the v2 bug review:
  * correct mmpdb 3.x SQL (JOIN rule_environment; radius = RADIUS_GENERAL)
  * candidate-vs-candidate pairs emit a row anchored on EACH candidate

Output (config.RESULTS):
  anchored_pairs.csv

Usage:
  <tartarus python> 06_extract_pairs.py
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import config
import mmpa_lib as ml


def load_id_list(path):
    with open(path) as f:
        return set(line.strip() for line in f if line.strip())


def rule_support_map(conn, radius):
    """rule_id -> number of pairs at `radius` (the global transferable support)."""
    out = {}
    for r in conn.execute(
            "SELECT re.rule_id AS rid, count(*) AS n "
            "FROM pair p JOIN rule_environment re ON p.rule_environment_id=re.id "
            "WHERE re.radius=? GROUP BY re.rule_id", (radius,)):
        out[r["rid"]] = r["n"]
    return out


def fetch_anchored_pairs(conn, candidate_ids, properties, radius):
    prop_ids = ml.property_name_ids(conn, properties)
    if not prop_ids:
        sys.exit("ERROR: no requested properties found in DB")
    props = list(prop_ids.keys())
    pv = ml.load_property_values(conn, prop_ids)

    pub_to_int, int_to_pub = ml.compound_id_maps(conn)
    candidate_int_ids = set()
    missing = []
    for c in candidate_ids:
        if c in pub_to_int:
            candidate_int_ids.add(pub_to_int[c])
        else:
            missing.append(c)
    if missing:
        print(f"WARNING: {len(missing)} candidate IDs not in DB: "
              f"{missing[:5]}{'...' if len(missing) > 5 else ''}", file=sys.stderr)
    print(f"Looking up pairs for {len(candidate_int_ids)} candidates in DB...")

    print("Building global rule-support map...")
    supp = rule_support_map(conn, radius)

    cand_ph = ",".join("?" for _ in candidate_int_ids)
    if not cand_ph:
        sys.exit("No candidates in DB; nothing to extract.")

    query = f"""
    SELECT re.rule_id AS rule_id,
           pair.id AS pair_id,
           pair.compound1_id AS c1_id,
           pair.compound2_id AS c2_id,
           rs_from.smiles AS frag_from,
           rs_to.smiles AS frag_to,
           cs.smiles AS context_smiles
    FROM pair
    JOIN rule_environment AS re ON pair.rule_environment_id = re.id
    JOIN rule ON re.rule_id = rule.id
    JOIN rule_smiles AS rs_from ON rule.from_smiles_id = rs_from.id
    JOIN rule_smiles AS rs_to   ON rule.to_smiles_id   = rs_to.id
    JOIN constant_smiles AS cs  ON pair.constant_id     = cs.id
    WHERE re.radius = ?
      AND ( pair.compound1_id IN ({cand_ph})
            OR pair.compound2_id IN ({cand_ph}) )
    """
    params = [radius] + list(candidate_int_ids) + list(candidate_int_ids)

    rows_out = []
    n_seen = 0
    for row in conn.execute(query, params):
        n_seen += 1
        c1, c2 = row["c1_id"], row["c2_id"]
        # candidate-candidate pairs anchor on BOTH sides
        anchors = []
        if c1 in candidate_int_ids:
            anchors.append((c1, c2, row["frag_from"], row["frag_to"]))
        if c2 in candidate_int_ids:
            anchors.append((c2, c1, row["frag_to"], row["frag_from"]))

        for cand_int, neigh_int, cand_frag, neigh_frag in anchors:
            rec = {
                "pair_id": row["pair_id"],
                "rule_id": row["rule_id"],
                "global_n_support": supp.get(row["rule_id"], 0),
                "candidate_id": int_to_pub.get(cand_int, f"INT_{cand_int}"),
                "neighbor_id": int_to_pub.get(neigh_int, f"INT_{neigh_int}"),
                "neighbor_is_candidate": (neigh_int in candidate_int_ids),
                "context_smiles": row["context_smiles"],
                "candidate_frag": cand_frag,
                "neighbor_frag": neigh_frag,
            }
            n_props = 0
            for p in props:
                vc = pv.get((cand_int, p)); vn = pv.get((neigh_int, p))
                if vc is None or vn is None:
                    rec[f"d_{p}"] = ""
                else:
                    rec[f"d_{p}"] = vn - vc
                    n_props += 1
            rec["n_props_available"] = n_props
            if n_props == 0:
                continue
            rows_out.append(rec)

    print(f"  saw {n_seen} pairs touching candidates, emitted {len(rows_out)} rows")
    return rows_out, props


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(config.DB))
    ap.add_argument("--candidates", default=str(config.CANDIDATES_TXT))
    ap.add_argument("--properties", nargs="+", default=config.PROPERTIES)
    ap.add_argument("--radius", type=int, default=config.RADIUS_GENERAL)
    ap.add_argument("--output", default=str(config.RESULTS / "anchored_pairs.csv"))
    args = ap.parse_args()

    candidate_ids = load_id_list(args.candidates)
    print(f"Loaded {len(candidate_ids)} candidate IDs")

    conn = ml.connect(args.db)
    rows, props = fetch_anchored_pairs(conn, candidate_ids, args.properties, args.radius)
    if not rows:
        sys.exit("No anchored pairs found.")

    fields = ["pair_id", "rule_id", "global_n_support", "candidate_id",
              "neighbor_id", "neighbor_is_candidate", "context_smiles",
              "candidate_frag", "neighbor_frag", "n_props_available"]
    fields += [f"d_{p}" for p in props]
    floats = [f"d_{p}" for p in props]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    ml.write_rows(args.output, rows, fields, floats)
    print(f"Wrote {len(rows)} anchored pairs to {args.output}")

    pair_counts = Counter(r["candidate_id"] for r in rows)
    print("Top candidates by pair count:")
    for cid, n in pair_counts.most_common(5):
        print(f"  {cid:<15} {n}")
    print(f"Candidates with zero pairs: {len(candidate_ids) - len(pair_counts)}")


if __name__ == "__main__":
    main()
