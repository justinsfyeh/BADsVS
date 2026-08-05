#!/bin/bash
# 02_build_db_linker.sh -- linker-aware mmpdb build (108k + 62)
#
# Same index parameters as 02_build_db.sh, but fragmentation uses acyclic
# C-cut + N-N cut (see linker_cut_smarts.py / linker_fragment.py).
#
# Inputs (from 01_prepare.py):  data/mols.smi, data/properties.tsv
# Outputs (do NOT overwrite legacy results/):
#   data/mols_linker.fragments
#   results_linker/pairs.mmpdb
#
# Usage (after: conda activate mmp):
#   ./02_build_db_linker.sh data results_linker 2 12 0        # fragment + index
#   ./02_build_db_linker.sh data results_linker 2 12 1        # index only
#
# Requires: linker_fragment.py, linker_cut_smarts.py, data/mols.smi,
#           data/properties.tsv. Fragmentation needs substantial CPU;
#           indexing needs ~100+ GB RAM and writes ~160 GB pairs.mmpdb.

set -euo pipefail

DATA_DIR="${1:-data}"
RESULTS_DIR="${2:-results_linker}"
NUM_CUTS="${3:-2}"
MAX_VAR_HEAVIES="${4:-12}"
SKIP_FRAGMENT="${5:-0}"

PY="${PY:-$(command -v python3 || command -v python)}"
NUM_JOBS="${NUM_JOBS:-4}"

mkdir -p "$RESULTS_DIR" logs

SMI_FILE="${DATA_DIR}/mols.smi"
PROP_FILE="${DATA_DIR}/properties.tsv"
FRAG_FILE="${DATA_DIR}/mols_linker.fragments"
DB_FILE="${RESULTS_DIR}/pairs.mmpdb"

if [[ ! -f "$SMI_FILE" ]]; then
    echo "ERROR: $SMI_FILE not found. Run 01_prepare.py first." >&2
    exit 1
fi
if [[ ! -f "$PROP_FILE" ]]; then
    echo "ERROR: $PROP_FILE not found. Run 01_prepare.py first." >&2
    exit 1
fi

if ! command -v mmpdb >/dev/null 2>&1; then
    echo "ERROR: 'mmpdb' not in PATH. Activate the mmp conda env." >&2
    exit 1
fi

MAX_HEAVIES_TRANSF="${MAX_HEAVIES_TRANSF:-8}"
MAX_VAR_RATIO="${MAX_VAR_RATIO:-0.5}"

echo "Linker-aware MMP build"
echo "  cut patterns: C-acyclic + N-N acyclic (!@, single bonds only)"
echo "  num_cuts=$NUM_CUTS, max_variable_heavies=$MAX_VAR_HEAVIES"
echo "  max_heavies=35, max_rotatable_bonds=15, salt=<none>"
echo "  max_heavies_transf=$MAX_HEAVIES_TRANSF, max_variable_ratio=$MAX_VAR_RATIO"
echo "  skip_fragment=$SKIP_FRAGMENT, num_jobs=$NUM_JOBS"
echo "  frag_out=$FRAG_FILE"
echo "  db_out=$DB_FILE"
echo

if [[ "$SKIP_FRAGMENT" != "1" ]]; then
    echo "[$(date)] Fragmenting (linker cuts)..."
    "$PY" "$(dirname "$0")/linker_fragment.py" \
        --smi "$SMI_FILE" \
        --output "$FRAG_FILE" \
        --num-cuts "$NUM_CUTS" \
        -j "$NUM_JOBS"
elif [[ ! -f "$FRAG_FILE" ]]; then
    echo "ERROR: skip_fragment=1 but $FRAG_FILE missing." >&2
    exit 1
else
    echo "[$(date)] Reusing existing fragments: $FRAG_FILE"
fi

echo "[$(date)] Indexing pairs (NO --symmetric)..."
# Disable RDKit in the SAME process as mmpdb (a separate python -c does not work).
RATIO_ARGS=()
if [[ -n "$MAX_VAR_RATIO" && "$MAX_VAR_RATIO" != "none" ]]; then
    RATIO_ARGS=(--max-variable-ratio "$MAX_VAR_RATIO")
fi
"$PY" - "$FRAG_FILE" "$DB_FILE" "$MAX_VAR_HEAVIES" "$MAX_HEAVIES_TRANSF" "$PROP_FILE" "${RATIO_ARGS[@]}" <<'PY'
from rdkit import RDLogger, rdBase
RDLogger.DisableLog("rdApp.*")
rdBase.DisableLog("rdApp.*")

import sys
from mmpdblib import commandline

frag, db, max_var, max_transf, prop = sys.argv[1:6]
extra = sys.argv[6:]
sys.argv = [
    "mmpdb", "index", frag,
    "--output", db,
    "--max-variable-heavies", max_var,
    "--max-heavies-transf", max_transf,
    "--properties", prop,
] + extra
raise SystemExit(commandline.main())
PY

echo "[$(date)] Done."
echo "Pair database: $DB_FILE"
echo "Quick inspect: mmpdb list $DB_FILE"
