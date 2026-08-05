# mmp_analysis

Matched Molecular Pair Analysis (MMPA) pipeline for the linker-inclusive
energetic-materials library: mine transformations across substituent / ring /
bridge axes, stratify by design layer, and produce publication figures.

## Layout

```
mmp_analysis/
  02_build_db_linker.sh     # fragment → mmpdb index
  03_…15_*.py               # analysis + figure stages
  config.py  layers.py  mmpa_lib.py  figstyle.py
  environment.yml           # conda env `mmp`
  master.csv  candidates.csv
  data/
    properties.tsv
    candidates.txt
    # mols_linker.fragments  (~305 MB) — not in git; needed to rebuild the DB
  results_linker/           # placeholder — pairs.mmpdb (~160 GB) lives on /work
  results_linker_s100/      # support-filtered analysis outputs + figures
```

## Database vs results

There is **one** MMP index:

```
results_linker/pairs.mmpdb   # built once by job_index_linker.pbs (~160 GB)
```

`results_linker_s100/` does **not** contain its own DB. Jobs set:

```bash
export MMPA_DB=.../results_linker/pairs.mmpdb
export MMPA_RESULTS=.../results_linker_s100
```

and re-run the pipeline with `min_support=100`, writing CSVs/figures only.

Neither `pairs.mmpdb` nor `data/mols_linker.fragments` are committed (see `.gitignore`).

## Reproduce figures (from shipped s100 tables)

```bash
conda env create -f environment.yml   # or: conda activate mmp
cd mmp_analysis
python 15_axis_figures.py \
  --results results_linker_s100/ \
  --output-dir results_linker_s100/figures/
```

Layer figures: `python 14_layer_figures.py --results results_linker_s100/ ...`

## Full re-analysis (needs the DB on disk)

```bash
conda activate mmp
export MMPA_DB=/path/to/pairs.mmpdb
export MMPA_RESULTS=$PWD/results_linker_s100

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
python 14_layer_figures.py --results results_linker_s100/
python 15_axis_figures.py --results results_linker_s100/ \
  --output-dir results_linker_s100/figures/
```
