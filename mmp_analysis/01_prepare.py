#!/usr/bin/env python3
"""
01_prepare.py -- v3 stage 1: prepare combined input for mmpdb.

Reads a single master CSV containing BOTH the database molecules (108k)
and the candidate molecules (62), with a column flagging which is which.
Validates SMILES with RDKit, canonicalizes, deduplicates, and writes the
files mmpdb needs (mols.smi, properties.tsv) plus a manifest.json trace.

Candidates and database go into ONE fragmentation/index pass so candidate
molecules participate in pairing (anchored MMPA).

Usage:
  <tartarus python> 01_prepare.py --input master.csv --out-dir data
  # properties default to config.PROPERTIES (Hf Q OBgood Pe D P IS)

Input CSV columns expected (column names configurable via flags):
  smiles          SMILES string
  mol_id          unique identifier (used everywhere downstream)
  is_candidate    1/0 or True/False -- candidate flag
  Hf, Q, ...      property values (numeric or blank)

Output:
  data/mols.smi          -- "smiles<TAB>id" per line for `mmpdb fragment`
  data/properties.tsv    -- id-keyed TAB-separated property table for `mmpdb index`
  data/candidates.txt    -- list of candidate IDs, one per line
  data/database.txt      -- list of database IDs, one per line
  data/manifest.json     -- provenance: counts, dropped rows, candidate IDs
"""

import argparse
import csv
import json
import sys
from pathlib import Path

try:
    import config
except Exception:
    config = None

try:
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
except ImportError:
    sys.stderr.write("ERROR: RDKit not installed. `conda install -c conda-forge rdkit`\n")
    sys.exit(1)


def canonicalize(smi):
    """Return canonical SMILES, or None on failure."""
    if not smi or not smi.strip():
        return None
    mol = Chem.MolFromSmiles(smi.strip())
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def truthy(v):
    """Parse 1/0/True/False/yes/no flag values."""
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in ('1', 'true', 't', 'yes', 'y')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', required=True)
    parser.add_argument('--smiles-col', default='smiles')
    parser.add_argument('--id-col', default='mol_id')
    parser.add_argument('--is-candidate-col', default='is_candidate',
                        help='Column whose truthy value marks a candidate. '
                             'If absent, all rows are treated as database.')
    parser.add_argument('--properties', nargs='+',
                        default=(config.PROPERTIES if config else None),
                        required=(config is None))
    parser.add_argument('--out-dir', default='data')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.input, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    print(f"Read {len(rows)} rows from {args.input}")

    if args.smiles_col not in fieldnames:
        sys.exit(f"ERROR: SMILES column '{args.smiles_col}' not in input")
    if args.id_col not in fieldnames:
        sys.exit(f"ERROR: ID column '{args.id_col}' not in input")
    missing = [p for p in args.properties if p not in fieldnames]
    if missing:
        print(f"WARNING: property columns missing from input, skipping: {missing}")
        args.properties = [p for p in args.properties if p in fieldnames]
    if not args.properties:
        sys.exit("ERROR: no usable property columns remain")

    has_cand_col = args.is_candidate_col in fieldnames
    if not has_cand_col:
        print(f"WARNING: candidate-flag column '{args.is_candidate_col}' not in "
              "input; treating ALL molecules as database (no candidates).")

    seen = {}
    output_rows = []
    candidates = []
    database = []
    n_invalid = n_dup = 0

    for row in rows:
        canon = canonicalize(row[args.smiles_col])
        if canon is None:
            n_invalid += 1
            continue
        if canon in seen:
            n_dup += 1
            continue
        mid = row[args.id_col].strip()
        if not mid:
            n_invalid += 1
            continue
        seen[canon] = mid

        # Parse properties (allow blank values)
        out = {'id': mid, 'smiles': canon}
        valid = True
        for p in args.properties:
            v = row[p].strip()
            if v == '':
                out[p] = ''
            else:
                try:
                    out[p] = float(v)
                except ValueError:
                    valid = False
                    break
        if not valid:
            n_invalid += 1
            continue

        is_cand = truthy(row.get(args.is_candidate_col)) if has_cand_col else False
        out['_is_candidate'] = is_cand

        output_rows.append(out)
        (candidates if is_cand else database).append(mid)

    print(f"  valid + unique:     {len(output_rows)}")
    print(f"    of which candidates: {len(candidates)}")
    print(f"    of which database:   {len(database)}")
    print(f"  invalid:            {n_invalid}")
    print(f"  duplicates:         {n_dup}")

    # Write SMILES file
    smi_path = out_dir / 'mols.smi'
    with open(smi_path, 'w') as f:
        for r in output_rows:
            f.write(f"{r['smiles']}\t{r['id']}\n")
    print(f"Wrote {smi_path}")

    # Write properties TSV (mmpdb index consumes a TAB-separated table keyed by
    # ID). Blank cells are allowed. Note: TSV, NOT CSV -- this is the file the
    # whole v3 pipeline expects (config.PROP_FILE).
    prop_path = out_dir / 'properties.tsv'
    fieldnames_out = ['id'] + args.properties
    with open(prop_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_out, delimiter='\t')
        writer.writeheader()
        for r in output_rows:
            writer.writerow({k: r[k] for k in fieldnames_out})
    print(f"Wrote {prop_path}")

    # Write candidate / database ID lists
    with open(out_dir / 'candidates.txt', 'w') as f:
        f.write('\n'.join(candidates) + ('\n' if candidates else ''))
    with open(out_dir / 'database.txt', 'w') as f:
        f.write('\n'.join(database) + ('\n' if database else ''))
    print(f"Wrote candidate/database ID lists")

    # Write a manifest so downstream stages (and reviewers) can trace exactly
    # what entered the index and what was dropped.
    manifest = {
        'input': str(args.input),
        'properties': args.properties,
        'n_input_rows': len(rows),
        'n_valid_unique': len(output_rows),
        'n_candidates': len(candidates),
        'n_database': len(database),
        'n_invalid': n_invalid,
        'n_duplicates': n_dup,
        'candidate_ids': candidates,
    }
    with open(out_dir / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {out_dir / 'manifest.json'}")


if __name__ == '__main__':
    main()
