# mmp_analysis

Matched Molecular Pair Analysis (MMPA) for the linker-inclusive energetic
materials library. Mines transformations across substituent / ring / bridge
axes, stratifies by design layer, and produces publication figures.

## What a clean clone can reproduce

| Goal | Works from clone alone? | Notes |
|---|---|---|
| **A. Re-draw published figures** from shipped CSVs | **Yes** | Needs only the `mmp` conda env |
| **B. Re-run analysis stages 03–15** | Needs `pairs.mmpdb` | ~160 GB index, not in git — build with step C, or copy from the cluster |
| **C. Rebuild DB from `master.csv`** | Scripts are included | Needs large CPU/RAM/disk; fragments (~305 MB) and DB (~160 GB) are gitignored |

`data/mols_linker.fragments` and `results_linker/pairs.mmpdb` are **intentionally not committed**.

## Setup

```bash
git clone https://github.com/justinsfyeh/BADsVS.git
cd BADsVS/mmp_analysis
conda env create -f environment.yml
conda activate mmp
```

## A. Reproduce figures (recommended first check)

```bash
python 15_axis_figures.py \
  --results results_linker_s100/ \
  --output-dir results_linker_s100/figures/

python 14_layer_figures.py \
  --results results_linker_s100/ \
  --output-dir results_linker_s100/figures/
```

Inputs used: `results_linker_s100/transformations.csv`, `corr_compare.csv`, etc.
(already in the repo). Optional panels S3/S5 need stage-13/10 extras and may skip.

## B. Re-run the s100 analysis (min_support=100)

Requires an existing index at `results_linker/pairs.mmpdb` (or set `MMPA_DB`).

```bash
conda activate mmp
export MMPA_DB=$PWD/results_linker/pairs.mmpdb
export MMPA_RESULTS=$PWD/results_linker_s100
mkdir -p "$MMPA_RESULTS/aggregated" "$MMPA_RESULTS/figures"

python 03_fragment_goodness.py --min-support 100
python 04_mine_transformations.py --min-support 100
python 05_context_rules.py
python 06_extract_pairs.py
python 07_candidate_endpoints.py --min-support 100
python 08_aggregate.py
python 09_correlations.py
python 10_candidate_profiles.py
python 11_recommend.py --min-support 100
python 12_figures.py
python 13_layer_report.py
python 14_layer_figures.py --results results_linker_s100/ \
  --output-dir results_linker_s100/figures/
python 15_axis_figures.py --results results_linker_s100/ \
  --output-dir results_linker_s100/figures/
```

## C. Rebuild inputs + MMP index from scratch

```bash
conda activate mmp

# 1) SMILES + property table from the master table
python 01_prepare.py --input master.csv --out-dir data
# writes: data/mols.smi, data/properties.tsv, candidates.txt, database.txt, manifest.json

# 2+3) Linker-aware fragmentation (C-acyclic + N-N) then mmpdb index
#     Argument 5 = 0 → fragment + index; = 1 → index only (reuse fragments)
NUM_JOBS=8 ./02_build_db_linker.sh data results_linker 2 12 0
# writes: data/mols_linker.fragments
#         results_linker/pairs.mmpdb   (~160 GB; needs large RAM/disk)
```

Fragmentation uses `linker_fragment.py` + `linker_cut_smarts.py` (multi-cut SMARTS
because mmpdb accepts only one cut string). Indexing attaches `data/properties.tsv`.

Then continue with **B**.

## Layout

```
mmp_analysis/
  01_prepare.py             master.csv → data/mols.smi + properties.tsv
  linker_cut_smarts.py      C-acyclic + N-N cut SMARTS
  linker_fragment.py        multi-cut fragmentation
  02_build_db_linker.sh     fragment + mmpdb index
  03_…15_*.py               analysis + figures
  config.py  layers.py  mmpa_lib.py  figstyle.py
  environment.yml           conda env name: mmp
  master.csv  candidates.csv
  data/                     properties.tsv, candidates.txt (fragments gitignored)
  results_linker/           pairs.mmpdb placeholder (DB gitignored)
  results_linker_s100/      support-filtered CSVs + figures (shipped)
```

## Database vs s100 results

There is **one** MMP index: `results_linker/pairs.mmpdb`.

`results_linker_s100/` is only analysis output. It does not contain a DB; jobs set
`MMPA_DB` → the shared index and `MMPA_RESULTS` → the s100 tree, with
`min_support=100` on stages 03/04/07/11.
