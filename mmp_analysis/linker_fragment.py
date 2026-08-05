#!/usr/bin/env python3
"""
linker_fragment.py -- fragment 108k+62 with acyclic C-cut + N-N linker cuts.

mmpdb accepts only one cut-smarts string; this module unions matches from
linker_cut_smarts.CUT_PATTERNS via MultiCutFragmentFilter.

Usage:
  <tartarus python> linker_fragment.py
  <tartarus python> linker_fragment.py --smi data/mols.smi -o data/mols_linker.fragments -j 8
"""
from __future__ import print_function

import argparse
import sys
from pathlib import Path

from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

import config
import linker_cut_smarts as lcs
from mmpdblib.do_fragment import (
    FragmentFilter,
    create_pool,
    get_fragment_filter,
    make_fragment_records,
)
from mmpdblib import fragment_io, fileio
from mmpdblib import config as mmpdb_config


class MultiCutFragmentFilter(FragmentFilter):
    """Union cut-bond matches from multiple SMARTS patterns."""

    def __init__(self, cut_patterns, max_heavies, max_rotatable_bonds,
                 rotatable_pattern, salt_remover, num_cuts, method, options):
        super(MultiCutFragmentFilter, self).__init__(
            max_heavies=max_heavies,
            max_rotatable_bonds=max_rotatable_bonds,
            rotatable_pattern=rotatable_pattern,
            salt_remover=salt_remover,
            cut_pattern=cut_patterns[0],
            num_cuts=num_cuts,
            method=method,
            options=options,
        )
        self.cut_patterns = cut_patterns

    def get_cut_atom_pairs(self, mol):
        seen = set()
        pairs = []
        for pat in self.cut_patterns:
            for match in mol.GetSubstructMatches(pat, uniquify=True):
                key = tuple(sorted(match))
                if key in seen:
                    continue
                seen.add(key)
                pairs.append(match)
        return pairs


def build_fragment_filter(num_cuts=2):
    opts = mmpdb_config.DEFAULT_FRAGMENT_OPTIONS
    fragment_options = mmpdb_config.FragmentOptions(
        max_heavies=35,
        max_rotatable_bonds=15,
        rotatable_smarts=opts.rotatable_smarts,
        cut_smarts=lcs.CUT_C_ACYCLIC,
        num_cuts=num_cuts,
        salt_remover="<none>",
        method="chiral",
    )
    base = get_fragment_filter(fragment_options)
    patterns = lcs.compile_cut_patterns()
    return MultiCutFragmentFilter(
        patterns,
        max_heavies=base.max_heavies,
        max_rotatable_bonds=base.max_rotatable_bonds,
        rotatable_pattern=base.rotatable_pattern,
        salt_remover=base.salt_remover,
        num_cuts=base.num_cuts,
        method=base.method,
        options=base.options,
    )


def run_fragmentation(smi_path, frag_path, num_cuts=2, num_jobs=4):
    smi_path = Path(smi_path)
    frag_path = Path(frag_path)
    frag_path.parent.mkdir(parents=True, exist_ok=True)

    filt = build_fragment_filter(num_cuts=num_cuts)
    pool = create_pool(num_jobs)

    print(f"Cut patterns: {lcs.CUT_SMARTS_LABEL}")
    print(f"Input:  {smi_path}")
    print(f"Output: {frag_path}")
    print(f"num_cuts={num_cuts}, num_jobs={num_jobs}")

    try:
        with fileio.read_smiles_file(str(smi_path), "smi", "whitespace", False) as reader:
            with fragment_io.open_fragment_writer(str(frag_path), filt.options) as writer:
                records = list(make_fragment_records(
                    reader, filt, cache=None, pool=pool))
                writer.write_records(records)
    finally:
        pool.terminate()
        pool.join()

    n_ok = sum(1 for r in records if r.fragments)
    n_zero = sum(1 for r in records if not r.fragments and not getattr(r, "errmsg", None))
    n_err = sum(1 for r in records if getattr(r, "errmsg", None))
    print(f"Fragmented {len(records)} records")
    print(f"  with cuts: {n_ok}; zero-cut: {n_zero}; errors: {n_err}")
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smi", default=str(config.SMI_FILE))
    ap.add_argument("-o", "--output", default=str(config.FRAG_FILE))
    ap.add_argument("--num-cuts", type=int, default=2)
    ap.add_argument("-j", "--num-jobs", type=int, default=4)
    args = ap.parse_args()

    if not Path(args.smi).exists():
        sys.exit(f"ERROR: SMILES file not found: {args.smi}")

    run_fragmentation(args.smi, args.output, num_cuts=args.num_cuts,
                      num_jobs=args.num_jobs)


if __name__ == "__main__":
    main()
