#!/usr/bin/env bash
# run_s100.sh -- re-run support-filtered analysis (stages 03–15).
# Requires results_linker/pairs.mmpdb (or MMPA_DB) and conda env `mmp`.
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v python >/dev/null; then
  echo "Activate conda env mmp first." >&2
  exit 1
fi

export MMPA_DB="${MMPA_DB:-$PWD/results_linker/pairs.mmpdb}"
export MMPA_RESULTS="${MMPA_RESULTS:-$PWD/results_linker_s100}"
mkdir -p "$MMPA_RESULTS/aggregated" "$MMPA_RESULTS/figures"

if [[ ! -f "$MMPA_DB" ]]; then
  echo "ERROR: MMP DB not found: $MMPA_DB" >&2
  echo "Build it with ./02_build_db_linker.sh or set MMPA_DB." >&2
  exit 1
fi

echo "MMPA_DB=$MMPA_DB"
echo "MMPA_RESULTS=$MMPA_RESULTS"

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
python 14_layer_figures.py --results "$MMPA_RESULTS" --output-dir "$MMPA_RESULTS/figures"
python 15_axis_figures.py --results "$MMPA_RESULTS" --output-dir "$MMPA_RESULTS/figures"

echo "Done. Figures in $MMPA_RESULTS/figures/"
