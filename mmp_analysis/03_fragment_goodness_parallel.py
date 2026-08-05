#!/usr/bin/env python3
"""
03_fragment_goodness_parallel.py -- parallel version of stage 03.

Same statistics as 03_fragment_goodness.py, but shards the pair stream across
worker processes (by pair.id % n_workers) and merges Moments / Bivariate /
counts at the end.

Usage:
  python 03_fragment_goodness_parallel.py --workers 8
  python 03_fragment_goodness_parallel.py --workers 8 --pair-id-mod 20   # ~5% sample
"""
from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import config
import layers as L
import mmpa_lib as ml


# --------------------------------------------------------------------------
# Serializable accumulator blobs (plain tuples / dicts for pickle)
# --------------------------------------------------------------------------
def _mom_new():
    return [0, 0.0, 0.0, 0]  # n, s, ss, npos


def _mom_add(m, x):
    m[0] += 1
    m[1] += x
    m[2] += x * x
    if x > 0:
        m[3] += 1


def _mom_merge(a, b):
    a[0] += b[0]
    a[1] += b[1]
    a[2] += b[2]
    a[3] += b[3]


def _mom_to_obj(m):
    o = ml.Moments()
    o.n, o.s, o.ss, o.npos = m[0], m[1], m[2], m[3]
    return o


def _biv_new():
    return [0, 0.0, 0.0, 0.0, 0.0, 0.0]  # n, sx, sy, sxx, syy, sxy


def _biv_add(b, x, y):
    b[0] += 1
    b[1] += x
    b[2] += y
    b[3] += x * x
    b[4] += y * y
    b[5] += x * y


def _biv_merge(a, b):
    for i in range(6):
        a[i] += b[i]


def _biv_to_obj(b):
    o = ml.Bivariate()
    o.n, o.sx, o.sy, o.sxx, o.syy, o.sxy = b
    return o


def _process_shard(args):
    """Worker: stream pairs where pair.id % n_workers == worker_id."""
    (db, props, prop_ids, radius, track_contexts, worker_id, n_workers,
     pair_id_mod, progress_every) = args

    conn = ml.connect(db)
    pv = ml.load_property_values(conn, prop_ids)
    buckets = ("all",) + tuple(L.LAYERS)
    prop_pairs = list(combinations(props, 2))

    frag_mom = {}          # frag -> {p: mom_list}
    frag_pairs = defaultdict(int)
    frag_ctx = defaultdict(set) if track_contexts else None
    global_mom = {p: _mom_new() for p in props}
    biv = {b: {pc: _biv_new() for pc in prop_pairs} for b in buckets}
    layer_pairs = defaultdict(int)
    core_mom = {b: _mom_new() for b in buckets}
    fragsz_mom = {b: _mom_new() for b in buckets}
    ratio_mom = {b: _mom_new() for b in buckets}

    where = ["(pair.id % :n_workers) = :worker_id"]
    params = {"n_workers": n_workers, "worker_id": worker_id}
    if radius is not None:
        where.append("re.radius = :radius")
        params["radius"] = radius
    if pair_id_mod is not None and pair_id_mod > 1:
        # Thinning independent of n_workers so 1-worker and N-worker
        # runs see the same pair set: id % pair_id_mod == 0.
        where.append("(pair.id % :pair_id_mod) = 0")
        params["pair_id_mod"] = pair_id_mod

    query = f"""
    SELECT pair.compound1_id AS c1,
           pair.compound2_id AS c2,
           rs_from.smiles    AS from_smiles,
           rs_to.smiles      AS to_smiles,
           cs.smiles         AS context
    FROM pair
    JOIN rule_environment re ON pair.rule_environment_id = re.id
    JOIN rule              ON re.rule_id = rule.id
    JOIN rule_smiles rs_from ON rule.from_smiles_id = rs_from.id
    JOIN rule_smiles rs_to   ON rule.to_smiles_id   = rs_to.id
    LEFT JOIN constant_smiles cs ON pair.constant_id = cs.id
    WHERE {' AND '.join(where)}
    """
    cur = conn.execute(query, params)
    n_pairs = 0
    t0 = time.time()
    while True:
        rows = cur.fetchmany(5000)
        if not rows:
            break
        for r in rows:
            n_pairs += 1
            if progress_every and n_pairs % progress_every == 0:
                dt = time.time() - t0
                print(f"  [worker {worker_id}] ...{n_pairs} pairs "
                      f"({n_pairs / max(dt, 1e-6):.0f}/s)", flush=True)

            c1, c2 = r["c1"], r["c2"]
            vfrom, vto = r["from_smiles"], r["to_smiles"]
            ctx = r["context"]

            lay, _cross = L.assign_layer(vfrom, vto)
            layer_pairs[lay] += 1
            frag_pairs[vfrom] += 1
            frag_pairs[vto] += 1
            if frag_ctx is not None and ctx is not None:
                frag_ctx[vfrom].add(ctx)
                frag_ctx[vto].add(ctx)

            if ctx is not None:
                core_h = L.frag_heavy(ctx)
                frag_h = max(L.frag_heavy(vfrom), L.frag_heavy(vto))
                for b in ("all", lay):
                    _mom_add(core_mom[b], core_h)
                    _mom_add(fragsz_mom[b], frag_h)
                    if core_h + frag_h > 0:
                        _mom_add(ratio_mom[b], frag_h / (frag_h + core_h))

            deltas = {}
            for p in props:
                a = pv.get((c1, p))
                b = pv.get((c2, p))
                if a is None or b is None:
                    continue
                dP = b - a
                deltas[p] = dP
                _mom_add(global_mom[p], dP)
                if vfrom not in frag_mom:
                    frag_mom[vfrom] = {pp: _mom_new() for pp in props}
                if vto not in frag_mom:
                    frag_mom[vto] = {pp: _mom_new() for pp in props}
                _mom_add(frag_mom[vfrom][p], -dP)
                _mom_add(frag_mom[vto][p], +dP)

            for pc in prop_pairs:
                a, c = pc
                if a in deltas and c in deltas:
                    _biv_add(biv["all"][pc], deltas[a], deltas[c])
                    _biv_add(biv[lay][pc], deltas[a], deltas[c])

    conn.close()
    print(f"  [worker {worker_id}] done: {n_pairs} pairs in "
          f"{time.time() - t0:.1f}s", flush=True)

    # sets -> lists for pickle size / speed
    frag_ctx_out = None
    if frag_ctx is not None:
        frag_ctx_out = {k: list(v) for k, v in frag_ctx.items()}

    return {
        "n_pairs": n_pairs,
        "frag_mom": frag_mom,
        "frag_pairs": dict(frag_pairs),
        "frag_ctx": frag_ctx_out,
        "global_mom": global_mom,
        "biv": biv,
        "layer_pairs": dict(layer_pairs),
        "core_mom": core_mom,
        "fragsz_mom": fragsz_mom,
        "ratio_mom": ratio_mom,
    }


def _merge_results(parts, props):
    buckets = ("all",) + tuple(L.LAYERS)
    prop_pairs = list(combinations(props, 2))

    frag_mom = {}
    frag_pairs = defaultdict(int)
    frag_ctx = defaultdict(set)
    global_mom = {p: _mom_new() for p in props}
    biv = {b: {pc: _biv_new() for pc in prop_pairs} for b in buckets}
    layer_pairs = defaultdict(int)
    core_mom = {b: _mom_new() for b in buckets}
    fragsz_mom = {b: _mom_new() for b in buckets}
    ratio_mom = {b: _mom_new() for b in buckets}
    n_pairs = 0
    track_ctx = False

    for part in parts:
        n_pairs += part["n_pairs"]
        for frag, n in part["frag_pairs"].items():
            frag_pairs[frag] += n
        for p in props:
            _mom_merge(global_mom[p], part["global_mom"][p])
        for b in buckets:
            _mom_merge(core_mom[b], part["core_mom"][b])
            _mom_merge(fragsz_mom[b], part["fragsz_mom"][b])
            _mom_merge(ratio_mom[b], part["ratio_mom"][b])
            for pc in prop_pairs:
                _biv_merge(biv[b][pc], part["biv"][b][pc])
        for lay, n in part["layer_pairs"].items():
            layer_pairs[lay] += n

        for frag, moms in part["frag_mom"].items():
            if frag not in frag_mom:
                frag_mom[frag] = {p: _mom_new() for p in props}
            for p in props:
                _mom_merge(frag_mom[frag][p], moms[p])

        if part["frag_ctx"] is not None:
            track_ctx = True
            for frag, ctxs in part["frag_ctx"].items():
                frag_ctx[frag].update(ctxs)

    return {
        "n_pairs": n_pairs,
        "frag_mom": frag_mom,
        "frag_pairs": frag_pairs,
        "frag_ctx": frag_ctx if track_ctx else None,
        "global_mom": {p: _mom_to_obj(global_mom[p]) for p in props},
        "biv": {b: {pc: _biv_to_obj(biv[b][pc]) for pc in prop_pairs}
                for b in buckets},
        "layer_pairs": layer_pairs,
        "core_mom": {b: _mom_to_obj(core_mom[b]) for b in buckets},
        "fragsz_mom": {b: _mom_to_obj(fragsz_mom[b]) for b in buckets},
        "ratio_mom": {b: _mom_to_obj(ratio_mom[b]) for b in buckets},
    }


def _write_outputs(merged, props, out: Path):
    buckets = ("all",) + tuple(L.LAYERS)
    frag_mom = merged["frag_mom"]
    frag_pairs = merged["frag_pairs"]
    frag_ctx = merged["frag_ctx"]
    global_mom = merged["global_mom"]
    biv = merged["biv"]
    layer_pairs = merged["layer_pairs"]
    core_mom = merged["core_mom"]
    fragsz_mom = merged["fragsz_mom"]
    ratio_mom = merged["ratio_mom"]
    n_pairs = merged["n_pairs"]

    print(f"Streamed {n_pairs} pairs; {len(frag_pairs)} distinct fragments")
    print("Pairs per layer: " +
          ", ".join(f"{k}={v}" for k, v in sorted(layer_pairs.items())))

    fields = ["fragment", "chemotype", "layer", "n_attach", "frag_heavy",
              "n_total_pairs", "n_contexts"]
    for p in props:
        fields += [f"A_{p}", f"s_{p}", f"n_{p}", f"fpos_{p}", f"t_{p}"]
    float_cols = [c for c in fields if c.split("_")[0] in ("A", "s", "fpos", "t")]

    rows = []
    for frag, moms in frag_mom.items():
        r = {"fragment": frag,
             "chemotype": ml.primary_chemotype(frag),
             "layer": L.frag_layer(frag),
             "n_attach": L.n_attachments(frag),
             "frag_heavy": L.frag_heavy(frag),
             "n_total_pairs": frag_pairs[frag],
             "n_contexts": len(frag_ctx[frag]) if frag_ctx is not None else ""}
        for p in props:
            m = _mom_to_obj(moms[p])
            r[f"A_{p}"] = m.mean if m.n else None
            r[f"s_{p}"] = m.std if m.n else None
            r[f"n_{p}"] = m.n
            r[f"fpos_{p}"] = m.frac_pos if m.n else None
            r[f"t_{p}"] = m.tstat if m.n > 1 else None
        rows.append(r)
    rows.sort(key=lambda r: -r["n_total_pairs"])
    ml.write_rows(out / "fragment_goodness.csv", rows, fields, float_cols)
    print(f"  -> {out / 'fragment_goodness.csv'} ({len(rows)} fragments)")

    gs_rows = [{"property": p, "mean": global_mom[p].mean,
                "std": global_mom[p].std, "n": global_mom[p].n} for p in props]
    ml.write_rows(out / "global_scales.csv", gs_rows,
                  ["property", "mean", "std", "n"], ["mean", "std"])
    print(f"  -> {out / 'global_scales.csv'}")

    dc_rows = []
    for b in buckets:
        for (a, c), bv in biv[b].items():
            dc_rows.append({"layer": b, "p1": a, "p2": c,
                            "corr": bv.corr, "n": bv.n})
    ml.write_rows(out / "delta_corr.csv", dc_rows,
                  ["layer", "p1", "p2", "corr", "n"], ["corr"])
    print(f"  -> {out / 'delta_corr.csv'} ({len(dc_rows)} rows over {len(buckets)} layers)")

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
    print(f"  -> {out / 'layer_sizes.csv'}")
    for r in ls_rows:
        if r["frag_ratio_mean"] != "":
            flag = ("  <-- exceeds MMPA convention"
                    if r["frag_ratio_mean"] > config.SIZE_CONVENTION["max_frag_ratio"]
                    else "")
            print(f"     {r['layer']:12s} mean frag_ratio={r['frag_ratio_mean']:.3f}{flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(config.DB))
    ap.add_argument("--properties", nargs="+", default=config.PROPERTIES)
    ap.add_argument("--output-dir", default=str(config.RESULTS))
    ap.add_argument("--radius", type=int, default=config.RADIUS_GENERAL,
                    help="Environment radius to keep. -1 = all.")
    ap.add_argument("--track-contexts", action="store_true", default=True)
    ap.add_argument("--no-track-contexts", action="store_true",
                    help="Skip context-set tracking (faster, less memory).")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() // 4),
                    help="Number of parallel workers.")
    ap.add_argument("--pair-id-mod", type=int, default=None,
                    help="Keep ~1/N of pairs (smoke / benchmark).")
    ap.add_argument("--progress-every", type=int, default=200000)
    args = ap.parse_args()

    track_contexts = args.track_contexts and not args.no_track_contexts
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    conn = ml.connect(args.db)
    radii = ml.environment_radii(conn)
    radius = None if args.radius == -1 else args.radius
    if radius is not None and radii and radius not in radii:
        print(f"WARNING: radius {radius} absent (DB radii={radii}); "
              f"using min radius {radii[0]} instead")
        radius = radii[0]
    print(f"DB environment radii present: {radii}; using radius={radius}")
    prop_ids = ml.property_name_ids(conn, args.properties)
    props = list(prop_ids.keys())
    missing = [p for p in args.properties if p not in props]
    if missing:
        print(f"WARNING: properties absent from DB and skipped: {missing}")
    # peek load size (workers reload their own copy)
    n_pv = conn.execute("SELECT COUNT(*) FROM compound_property").fetchone()[0]
    conn.close()
    print(f"compound_property rows≈{n_pv}; properties={props}")
    print(f"workers={args.workers}  pair_id_mod={args.pair_id_mod}  "
          f"track_contexts={track_contexts}")

    worker_args = [
        (args.db, props, prop_ids, radius, track_contexts, wid, args.workers,
         args.pair_id_mod, args.progress_every)
        for wid in range(args.workers)
    ]

    t0 = time.time()
    if args.workers == 1:
        parts = [_process_shard(worker_args[0])]
    else:
        # spawn avoids RDKit/fork issues; each worker opens its own DB handle
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.workers) as pool:
            parts = pool.map(_process_shard, worker_args)
    t_workers = time.time() - t0
    print(f"Workers finished in {t_workers:.1f}s")

    t1 = time.time()
    merged = _merge_results(parts, props)
    print(f"Merge finished in {time.time() - t1:.1f}s")

    _write_outputs(merged, props, out)
    print(f"TOTAL wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
