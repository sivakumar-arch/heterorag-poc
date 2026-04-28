#!/usr/bin/env bash
# =============================================================================
# HeteroRAG POC — Multi-Community Data Loader
# scripts/load_multi_community.sh
#
# Loads three Stack Exchange communities into the same PostgreSQL / Neo4j /
# Elasticsearch instance, namespacing each community's _load_progress entries
# so the resume logic works correctly across all three.
#
# Expected directory layout (place extracted XMLs here):
#   data/
#   ├── stats/          ← stats.stackexchange.com (Cross Validated)
#   │   ├── Users.xml
#   │   ├── Posts.xml
#   │   ├── Tags.xml
#   │   ├── Votes.xml   (optional)
#   │   ├── Badges.xml  (optional)
#   │   └── PostLinks.xml (optional)
#   ├── dba/            ← dba.stackexchange.com
#   │   └── ...
#   └── datascience/    ← datascience.stackexchange.com
#       └── ...
#
# Usage:
#   ./scripts/load_multi_community.sh               # load all three
#   ./scripts/load_multi_community.sh stats         # load only stats
#   ONLY=postgres ./scripts/load_multi_community.sh # load only PostgreSQL for all
#
# Environment overrides:
#   DATA_ROOT   — parent directory (default: ./data)
#   BATCH_SIZE  — write batch size (default: 1000)
#   ONLY        — restrict to postgres / neo4j / elasticsearch
#   DRY_RUN     — set to 1 to parse without writing
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/data}"
BATCH_SIZE="${BATCH_SIZE:-1000}"
ONLY="${ONLY:-}"
DRY_RUN="${DRY_RUN:-}"

# Communities to load — pass names as arguments, or load all three by default
if [ $# -gt 0 ]; then
    COMMUNITIES=("$@")
else
    COMMUNITIES=("stats" "dba" "datascience")
fi

echo "============================================================"
echo " HeteroRAG POC — Multi-Community Loader"
echo "============================================================"
echo " Data root  : ${DATA_ROOT}"
echo " Communities: ${COMMUNITIES[*]}"
echo " Batch size : ${BATCH_SIZE}"
echo " Only       : ${ONLY:-all services}"
echo " Dry run    : ${DRY_RUN:-false}"
echo "============================================================"
echo ""

FAILED=()

for COMMUNITY in "${COMMUNITIES[@]}"; do
    DATA_DIR="${DATA_ROOT}/${COMMUNITY}"

    echo "------------------------------------------------------------"
    echo " Loading community: ${COMMUNITY}"
    echo " Source directory : ${DATA_DIR}"
    echo "------------------------------------------------------------"

    if [ ! -d "${DATA_DIR}" ]; then
        echo "  ERROR: Directory not found: ${DATA_DIR}"
        echo "  Skipping ${COMMUNITY}."
        FAILED+=("${COMMUNITY} (directory missing)")
        continue
    fi

    # Check minimum required files
    MISSING_REQUIRED=()
    for f in Users.xml Posts.xml Tags.xml; do
        [ ! -f "${DATA_DIR}/${f}" ] && MISSING_REQUIRED+=("${f}")
    done

    if [ ${#MISSING_REQUIRED[@]} -gt 0 ]; then
        echo "  ERROR: Missing required files: ${MISSING_REQUIRED[*]}"
        echo "  Skipping ${COMMUNITY}."
        FAILED+=("${COMMUNITY} (missing: ${MISSING_REQUIRED[*]})")
        continue
    fi

    # Build flags
    FLAGS="--data-dir ${DATA_DIR} --batch-size ${BATCH_SIZE} --community-name ${COMMUNITY}"
    [ -n "${ONLY}" ]    && FLAGS="${FLAGS} --only ${ONLY}"
    [ -n "${DRY_RUN}" ] && FLAGS="${FLAGS} --dry-run"

    echo "  Running: python3 scripts/load_stackoverflow_dump.py ${FLAGS}"
    echo ""

    if python3 "${SCRIPT_DIR}/load_stackoverflow_dump.py" ${FLAGS}; then
        echo ""
        echo "  ✓ ${COMMUNITY} loaded successfully"
    else
        echo ""
        echo "  ✗ ${COMMUNITY} load FAILED (exit code $?)"
        FAILED+=("${COMMUNITY} (loader error)")
    fi
    echo ""
done

echo "============================================================"
echo " Multi-community load summary"
echo "============================================================"
echo " Communities attempted : ${#COMMUNITIES[@]}"
echo " Failed                : ${#FAILED[@]}"
if [ ${#FAILED[@]} -gt 0 ]; then
    for item in "${FAILED[@]}"; do
        echo "   - ${item}"
    done
fi

if [ ${#FAILED[@]} -eq 0 ]; then
    echo ""
    echo " All communities loaded successfully."
    echo ""
    echo " Next steps:"
    echo "   docker compose --profile migrate up flyway    # Step 3: SQL migrations"
    echo "   liquibase update                               # Step 4: Graph migrations"
    echo "   python ground-truth/document/setup_index.py   # Step 5: ES index"
    echo "   python ground-truth/document/run_gt_queries.py --generate-fixtures  # Step 7"
    echo "   python evaluation/run_benchmark.py --mock      # Step 8 smoke test"
else
    echo ""
    echo " Some communities failed — check logs above."
    exit 1
fi
