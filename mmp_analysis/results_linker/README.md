# results_linker

The MMP index `pairs.mmpdb` (~160 GB) is **not** shipped in this repository.

It lives on the compute host at:

```
/work/r12524031/mmp_linker/results_linker/pairs.mmpdb
```

Rebuild by indexing `data/mols_linker.fragments` with mmpdb (see `02_build_db_linker.sh`).

Analysis jobs set:

```bash
export MMPA_DB=.../results_linker/pairs.mmpdb
export MMPA_RESULTS=.../results_linker_s100   # or results_linker
```
