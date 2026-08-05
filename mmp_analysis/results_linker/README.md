# results_linker

The MMP index `pairs.mmpdb` (~160 GB) is **not** shipped in this repository.

Rebuild on a machine with large RAM/disk:

```bash
conda activate mmp
python 01_prepare.py --input master.csv --out-dir data
NUM_JOBS=8 ./02_build_db_linker.sh data results_linker 2 12 0
```

Or point analysis jobs at an existing index:

```bash
export MMPA_DB=/path/to/pairs.mmpdb
export MMPA_RESULTS=$PWD/results_linker_s100
./run_s100.sh
```
