#!/usr/bin/env python3
"""
07_candidate_endpoints.py -- the bridge from rules to candidates (v4 stage 7).

v4: carries layer / cross_layer / undirected_key / frag_ratio through from
stage 04 so downstream ranking can stratify and deduplicate.

Places every candidate into the context of the mined transformation rules.
For each directed rule A -> B (stage 04) and each candidate, the candidate is:

  actionable        it contains fragment A  -> applying A->B is a proposed edit
  already_optimized it contains fragment B  -> it already sits on the good side
  (neither)         orthogonal to this rule -> not reported

Candidate fragment inventories come from data/mols.fragments (the fragmentation
of ALL 62 candidates, including the 5 that never paired), NOT from anchored
pairs -- so coverage is complete. Local anchored evidence (stage 06) is joined
where it exists.

Outputs (config.RESULTS):
  candidate_rule_matrix.csv   one row per (candidate, rule, role)
  actionable_shortlist.csv    best actionable edits per candidate (gated)

Usage:
  <tartarus python> 07_candidate_endpoints.py
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

import config
import layers as L
import mmpa_lib as ml


def candidate_inventories(frag_file, candidate_ids, anchored_path=None):
    """candidate -> set(canonical fragment keys).

    Union of two sources for maximal, DB-consistent coverage:
      (a) mols.fragments variable fragments (all fragmentable candidates)
      (b) anchored candidate_frag from stage 06 -- captures H-position edits
          (candidate_frag == [H]) and any fragment the file parser missed
          (e.g. candidates mmpdb recorded with 0 cuts but still paired).
    """
    cand = set(candidate_ids)
    inv = defaultdict(set)
    with open(frag_file) as f:
        for line in f:
            if not line.startswith('["RECORD"'):
                continue
            rec = json.loads(line)
            cid = rec[1]
            if cid not in cand:
                continue
            for cut in rec[5]:
                inv[cid].add(ml.canon_fragment(cut[4]))   # variable fragment
    if anchored_path and Path(anchored_path).exists():
        a = pd.read_csv(anchored_path)
        for cid, sub in a.groupby("candidate_id"):
            if cid in cand:
                for frag in sub["candidate_frag"].unique():
                    inv[cid].add(ml.canon_fragment(frag))
    return inv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transformations", default=str(config.RESULTS / "transformations.csv"))
    ap.add_argument("--anchored", default=str(config.RESULTS / "anchored_pairs.csv"))
    ap.add_argument("--fragments", default=str(config.FRAG_FILE))
    ap.add_argument("--candidates", default=str(config.CANDIDATES_TXT))
    ap.add_argument("--properties", nargs="+", default=config.PROPERTIES)
    ap.add_argument("--min-support", type=int, default=config.MIN_RULE_SUPPORT)
    ap.add_argument("--matrix-out", default=str(config.RESULTS / "candidate_rule_matrix.csv"))
    ap.add_argument("--shortlist-out", default=str(config.RESULTS / "actionable_shortlist.csv"))
    args = ap.parse_args()

    props = args.properties
    candidate_ids = [l.strip() for l in open(args.candidates) if l.strip()]
    print(f"Loaded {len(candidate_ids)} candidates")

    inv = candidate_inventories(args.fragments, candidate_ids, args.anchored)
    print(f"Built fragment inventories for {len(inv)} candidates "
          f"(median {sorted(len(v) for v in inv.values())[len(inv)//2]} frags)")

    # canonical-key -> candidates having it
    frag_to_cands = defaultdict(set)
    for cid, frags in inv.items():
        for fk in frags:
            frag_to_cands[fk].add(cid)

    t = pd.read_csv(args.transformations)
    t = t[t["n_pairs"] >= args.min_support].copy()
    # canonical keys for rule endpoints
    t["key_from"] = t["frag_from"].map(ml.canon_fragment)
    t["key_to"] = t["frag_to"].map(ml.canon_fragment)
    print(f"Loaded {len(t)} directed rules (>= {args.min_support} pairs)")

    # local anchored evidence: (candidate, canon frag_from, canon frag_to) -> mean deltas
    local = {}
    if Path(args.anchored).exists():
        a = pd.read_csv(args.anchored)
        a["kc"] = a["candidate_frag"].map(ml.canon_fragment)
        a["kn"] = a["neighbor_frag"].map(ml.canon_fragment)
        grp = a.groupby(["candidate_id", "kc", "kn"])
        for (cid, kc, kn), sub in grp:
            local[(cid, kc, kn)] = {p: sub[f"d_{p}"].mean() for p in props
                                    if f"d_{p}" in sub}

    matrix_rows = []
    for _, r in t.iterrows():
        kf, kt = r["key_from"], r["key_to"]
        # actionable: candidate has A (frag_from) -> edit toward B
        for cid in frag_to_cands.get(kf, ()):
            row = _mkrow(cid, r, "actionable", props, local, kf, kt)
            matrix_rows.append(row)
        # already optimized: candidate has B (frag_to)
        for cid in frag_to_cands.get(kt, ()):
            row = _mkrow(cid, r, "already_optimized", props, local, kf, kt)
            matrix_rows.append(row)

    fields = ["candidate_id", "role", "frag_from", "frag_to",
              "chemotype_from", "chemotype_to", "layer", "cross_layer",
              "undirected_key", "frag_ratio",
              "n_pairs", "global_n_support",
              "composite_gated", "composite_z", "n_improve", "n_worsen",
              "has_tradeoff", "has_local_evidence"]
    floats = ["composite_gated", "composite_z", "frag_ratio"]
    for p in props:
        fields += [f"z_{p}", f"sig_{p}", f"local_mean_d_{p}"]
        floats += [f"z_{p}", f"local_mean_d_{p}"]

    ml.write_rows(args.matrix_out, matrix_rows, fields, floats)
    print(f"Wrote {len(matrix_rows)} (candidate, rule, role) rows -> {args.matrix_out}")

    # shortlist: actionable, positive gated composite, best per candidate
    act = [r for r in matrix_rows
           if r["role"] == "actionable" and r["composite_gated"] > 0
           and r["n_worsen"] == 0]
    act.sort(key=lambda r: (r["candidate_id"], -r["composite_gated"]))
    ml.write_rows(args.shortlist_out, act, fields, floats)
    n_cand_act = len({r["candidate_id"] for r in act})
    print(f"Wrote {len(act)} clean actionable edits for {n_cand_act} candidates "
          f"-> {args.shortlist_out}")

    # coverage summary
    covered = {r["candidate_id"] for r in matrix_rows}
    print(f"Candidates appearing in >=1 rule: {len(covered)}/{len(candidate_ids)}")
    orphan = sorted(set(candidate_ids) - covered)
    if orphan:
        print(f"Candidates in NO mined rule: {orphan}")


def _mkrow(cid, r, role, props, local, kf, kt):
    row = {
        "candidate_id": cid, "role": role,
        "frag_from": r["frag_from"], "frag_to": r["frag_to"],
        "chemotype_from": r["chemotype_from"], "chemotype_to": r["chemotype_to"],
        "layer": r.get("layer", L.assign_layer(r["frag_from"], r["frag_to"])[0]),
        "cross_layer": r.get("cross_layer", ""),
        "undirected_key": r.get("undirected_key", ""),
        "frag_ratio": r.get("frag_ratio", ""),
        "n_pairs": int(r["n_pairs"]), "global_n_support": int(r["n_pairs"]),
        "composite_gated": r["composite_gated"], "composite_z": r["composite_z"],
        "n_improve": int(r["n_improve"]), "n_worsen": int(r["n_worsen"]),
        "has_tradeoff": bool(r["has_tradeoff"]),
    }
    lk = local.get((cid, kf, kt))
    row["has_local_evidence"] = lk is not None
    for p in props:
        row[f"z_{p}"] = r.get(f"z_{p}", "")
        row[f"sig_{p}"] = r.get(f"sig_{p}", "")
        row[f"local_mean_d_{p}"] = (lk.get(p) if lk else "")
    return row


if __name__ == "__main__":
    main()
