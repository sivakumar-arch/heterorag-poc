#!/usr/bin/env python3
"""
ground-truth/document/run_gt_queries.py
==========================================
RUNBOOK Step 7 — Generate document ground truth fixtures.

Runs BM25 retrieval for all 20 Class 3 (Document-only) benchmark questions
against the live Elasticsearch index and stores the top-10 hit IDs as
fixture files in ground-truth/document/fixtures/.

These fixtures are the ground truth for Recall@10 evaluation of document
retrieval. They are generated ONCE from the fixed dataset and committed to
the repository — they must not be regenerated between benchmark runs, as
that would make the ground truth a function of the retrieval system being
evaluated (circular).

Usage:
    python ground-truth/document/run_gt_queries.py --generate-fixtures
    python ground-truth/document/run_gt_queries.py --verify   # check fixtures exist
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("heterorag.gt_doc")

ES_URL      = os.getenv("ES_URL",   "http://localhost:9200")
ES_INDEX    = os.getenv("ES_INDEX", "heterorag_content")
FIXTURE_DIR = Path(__file__).parent / "fixtures"

# ---------------------------------------------------------------------------
# Class 3 — Document-only ground truth queries
# ---------------------------------------------------------------------------

CLASS3_QUERIES = [
    ("c3_q01", "Python GIL global interpreter lock thread concurrency"),
    ("c3_q02", "AttributeError Python object has no attribute"),
    ("c3_q03", "NullPointerException Java null reference fix"),
    ("c3_q04", "list comprehension Python syntax explanation"),
    ("c3_q05", "difference between == and is Python identity equality"),
    ("c3_q06", "javascript tag wiki explanation"),
    ("c3_q07", "memory leak C++ pointer delete free"),
    ("c3_q08", "async await JavaScript asynchronous promise explanation"),
    ("c3_q09", "Python decorator function wrapper explanation"),
    ("c3_q10", "SQL JOIN types INNER LEFT RIGHT OUTER explanation"),
    ("c3_q11", "segmentation fault C segfault pointer memory"),
    ("c3_q12", "REST vs GraphQL API comparison differences"),
    ("c3_q13", "python tag wiki explanation overview"),
    ("c3_q14", "IndentationError Python fix whitespace tab space"),
    ("c3_q15", "Docker networking bridge host container ports"),
    ("c3_q16", "process vs thread difference concurrency parallelism"),
    ("c3_q17", "pandas DataFrame indexing loc iloc KeyError"),
    ("c3_q18", "machine learning data scientist profile career"),
    ("c3_q19", "optimize SQL query index performance slow"),
    ("c3_q20", "JavaScript closure function scope variable explanation"),
]


def generate_fixtures(es_url: str, index: str, fixture_dir: Path, top_k: int = 10) -> None:
    """Execute BM25 queries and write fixture files."""
    try:
        from elasticsearch import Elasticsearch
    except ImportError:
        log.error("elasticsearch-py not installed. Run: pip install elasticsearch")
        sys.exit(1)

    es = Elasticsearch(es_url)
    if not es.ping():
        log.error("Cannot reach Elasticsearch at %s", es_url)
        sys.exit(1)

    # Check index exists and has documents
    if not es.indices.exists(index=index):
        log.error("Index '%s' does not exist. Run setup_index.py and load data first.", index)
        sys.exit(1)

    count = es.count(index=index)["count"]
    if count == 0:
        log.error("Index '%s' is empty. Load the Stack Overflow data first (RUNBOOK step 2).", index)
        sys.exit(1)

    log.info("Elasticsearch index '%s' has %d documents", index, count)

    fixture_dir.mkdir(parents=True, exist_ok=True)
    generated = 0

    for question_id, bm25_query in CLASS3_QUERIES:
        fixture_path = fixture_dir / f"{question_id}.json"

        if fixture_path.exists():
            log.info("  SKIP (exists): %s", question_id)
            continue

        try:
            resp = es.search(
                index=index,
                body={
                    "query": {
                        "multi_match": {
                            "query":  bm25_query,
                            "fields": ["title^2", "body", "tag_name"],
                            "type":   "best_fields",
                        }
                    },
                    "size": top_k,
                    "_source": ["doc_type", "post_id", "tag_name"],
                },
            )

            hits      = resp["hits"]["hits"]
            hit_ids   = [h["_id"] for h in hits]
            hit_scores = [h["_score"] for h in hits]

            fixture = {
                "question_id":   question_id,
                "bm25_query":    bm25_query,
                "expected_ids":  hit_ids,
                "scores":        hit_scores,
                "top_k":         top_k,
                "index":         index,
                "doc_count":     count,
                "generated_by":  "run_gt_queries.py",
            }

            with fixture_path.open("w") as f:
                json.dump(fixture, f, indent=2)

            log.info("  GENERATED: %s (%d hits, top score=%.2f)",
                     question_id, len(hit_ids),
                     hit_scores[0] if hit_scores else 0.0)
            generated += 1

        except Exception as exc:
            log.error("  FAILED: %s — %s", question_id, exc)

    log.info("Done. Generated %d new fixtures in %s", generated, fixture_dir)


def verify_fixtures(fixture_dir: Path) -> bool:
    """Check all 20 fixture files exist and are valid JSON."""
    all_ok = True
    for question_id, _ in CLASS3_QUERIES:
        path = fixture_dir / f"{question_id}.json"
        if not path.exists():
            log.warning("MISSING: %s.json", question_id)
            all_ok = False
            continue
        try:
            with path.open() as f:
                data = json.load(f)
            ids = data.get("expected_ids", [])
            log.info("OK: %s — %d expected IDs", question_id, len(ids))
        except Exception as exc:
            log.error("CORRUPT: %s — %s", question_id, exc)
            all_ok = False

    if all_ok:
        log.info("All 20 Class 3 fixtures are present and valid.")
    else:
        log.warning("Some fixtures are missing. Run with --generate-fixtures.")
    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate or verify document ground truth fixtures (RUNBOOK step 7)"
    )
    parser.add_argument("--generate-fixtures", action="store_true",
                        help="Run BM25 queries and write fixture files")
    parser.add_argument("--verify", action="store_true",
                        help="Check all fixture files exist")
    parser.add_argument("--fixture-dir", type=Path,
                        default=FIXTURE_DIR,
                        help=f"Fixture output directory (default: {FIXTURE_DIR})")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of hits to store per question (default: 10)")
    args = parser.parse_args()

    if not args.generate_fixtures and not args.verify:
        parser.print_help()
        sys.exit(0)

    if args.generate_fixtures:
        generate_fixtures(ES_URL, ES_INDEX, args.fixture_dir, args.top_k)

    if args.verify:
        ok = verify_fixtures(args.fixture_dir)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
