#!/usr/bin/env bash
# =============================================================================
# HeteroRAG POC — Data Loader Shell Wrapper
# RUNBOOK Step 2: ./scripts/load_stackoverflow_dump.sh
#
# For a single community:
#   ./scripts/load_stackoverflow_dump.sh
#
# With overrides:
#   DATA_DIR=./data/stats COMMUNITY=stats ./scripts/load_stackoverflow_dump.sh
#   ONLY=postgres ./scripts/load_stackoverflow_dump.sh
#   DRY_RUN=1 ./scripts/load_stackoverflow_dump.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data}"
BATCH_SIZE="${BATCH_SIZE:-1000}"
ONLY="${ONLY:-}"
DRY_RUN="${DRY_RUN:-}"
COMMUNITY="${COMMUNITY:-}"   # auto-derived from data dir name if not set

echo "============================================"
echo " HeteroRAG POC — Stack Overflow Data Loader"
echo "============================================"
echo " Data directory : ${DATA_DIR}"
echo " Community      : ${COMMUNITY:-auto}"
echo " Batch size     : ${BATCH_SIZE}"
echo " Only           : ${ONLY:-all services}"
echo " Dry run        : ${DRY_RUN:-false}"
echo "============================================"

if [ ! -d "${DATA_DIR}" ]; then
    echo "ERROR: Data directory not found: ${DATA_DIR}"
    echo "  Place your Stack Exchange XML files in ${DATA_DIR}"
    exit 1
fi

FLAGS="--data-dir ${DATA_DIR} --batch-size ${BATCH_SIZE}"
[ -n "${ONLY}" ]      && FLAGS="${FLAGS} --only ${ONLY}"
[ -n "${DRY_RUN}" ]   && FLAGS="${FLAGS} --dry-run"
[ -n "${COMMUNITY}" ] && FLAGS="${FLAGS} --community-name ${COMMUNITY}"

python3 -c "import psycopg2, elasticsearch, neo4j" 2>/dev/null || {
    echo "Installing Python dependencies..."
    pip install psycopg2-binary elasticsearch neo4j --quiet
}

echo "Starting load..."
python3 "${SCRIPT_DIR}/load_stackoverflow_dump.py" ${FLAGS}

echo ""
echo "Load complete."
