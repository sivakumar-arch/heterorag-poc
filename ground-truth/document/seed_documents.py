#!/usr/bin/env python3
"""
ground-truth/document/seed_documents.py
==========================================
RUNBOOK Step 6 — Verify document corpus is loaded (and reseed if needed).

The primary loading of documents into Elasticsearch is done by
scripts/load_stackoverflow_dump.py (RUNBOOK step 2). This script:

  1. Verifies the heterorag_content index exists and is populated.
  2. Reports a breakdown of document counts by doc_type.
  3. Optionally reseeds missing doc_types if the initial load was partial.

Usage:
    python ground-truth/document/seed_documents.py            # verify only
    python ground-truth/document/seed_documents.py --reseed   # reseed if incomplete
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("heterorag.seed_docs")

ES_URL   = os.getenv("ES_URL",   "http://localhost:9200")
ES_INDEX = os.getenv("ES_INDEX", "heterorag_content")

EXPECTED_DOC_TYPES = ["question", "answer", "comment", "user_about", "tag_wiki"]
MIN_COUNTS = {
    "question":  1_000,   # any reasonable SO dataset has thousands
    "answer":    1_000,
    "comment":   100,
    "user_about": 10,
    "tag_wiki":   5,
}


def verify(es_url: str, index: str) -> dict[str, int]:
    """Return doc_type → count from the live index."""
    try:
        from elasticsearch import Elasticsearch
    except ImportError:
        log.error("elasticsearch-py not installed. Run: pip install elasticsearch")
        sys.exit(1)

    es = Elasticsearch(es_url)
    if not es.ping():
        log.error("Cannot reach Elasticsearch at %s", es_url)
        sys.exit(1)

    if not es.indices.exists(index=index):
        log.error("Index '%s' does not exist. Run setup_index.py first.", index)
        sys.exit(1)

    resp = es.search(
        index=index,
        body={
            "size": 0,
            "aggs": {
                "by_type": {
                    "terms": {"field": "doc_type", "size": 20}
                }
            }
        }
    )

    counts: dict[str, int] = {}
    for bucket in resp["aggregations"]["by_type"]["buckets"]:
        counts[bucket["key"]] = bucket["doc_count"]

    total = sum(counts.values())
    log.info("Index '%s' — total documents: %d", index, total)
    log.info("%-20s %10s  %s", "doc_type", "count", "status")
    log.info("-" * 45)

    all_ok = True
    for dt in EXPECTED_DOC_TYPES:
        count   = counts.get(dt, 0)
        minimum = MIN_COUNTS.get(dt, 1)
        status  = "OK" if count >= minimum else f"LOW (expected >= {minimum})"
        if count < minimum:
            all_ok = False
        log.info("%-20s %10d  %s", dt, count, status)

    if all_ok:
        log.info("\nAll document types are adequately populated.")
    else:
        log.warning(
            "\nSome doc_types are missing or have too few documents.\n"
            "Re-run: ONLY=elasticsearch ./scripts/load_stackoverflow_dump.sh\n"
            "Or run with --reseed to attempt a targeted reseed."
        )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify Elasticsearch document corpus (RUNBOOK step 6)"
    )
    parser.add_argument("--reseed", action="store_true",
                        help="Re-run the Elasticsearch loader for missing doc_types")
    args = parser.parse_args()

    counts = verify(ES_URL, ES_INDEX)

    if args.reseed:
        missing = [dt for dt in EXPECTED_DOC_TYPES
                   if counts.get(dt, 0) < MIN_COUNTS.get(dt, 1)]
        if not missing:
            log.info("Nothing to reseed — all doc_types are populated.")
        else:
            log.info("Reseeding: %s", missing)
            log.info("Running: ONLY=elasticsearch ./scripts/load_stackoverflow_dump.sh")
            import subprocess
            result = subprocess.run(
                ["bash", "scripts/load_stackoverflow_dump.sh"],
                env={**os.environ, "ONLY": "elasticsearch"},
            )
            if result.returncode != 0:
                log.error("Reseed failed.")
                sys.exit(1)
            log.info("Reseed complete. Re-verifying...")
            verify(ES_URL, ES_INDEX)


if __name__ == "__main__":
    main()
