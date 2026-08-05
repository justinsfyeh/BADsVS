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
  job_*.pbs                 # Slurm wrappers
  environment.yml
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
export MMPA_DB=/path/to/pairs.mmpdb
export MMPA_RESULTS=$PWD/results_linker_s100
sbatch job_analyze_s100.pbs
```

Stages 03–13 are defined in that job script; axis figures are stage 15.
