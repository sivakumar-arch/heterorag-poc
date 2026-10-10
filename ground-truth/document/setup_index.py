#!/usr/bin/env python3
"""
HeteroRAG POC — Elasticsearch Index Setup
RUNBOOK Step 5: python ground-truth/document/setup_index.py

Creates the heterorag_content index with the shared mapping (heterorag/es_index.py) if
it does not exist. Safe to re-run: idempotent. The mapping is defined in one place so
the loader and this script cannot disagree about analysis or fields.
"""

import logging
import os
import sys
from pathlib import Path

from elasticsearch import Elasticsearch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from heterorag.es_index import FORBIDDEN_FIELDS, INDEX_BODY, INDEX_NAME  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("heterorag.es_setup")

ES_URL = os.getenv("ES_URL", "http://localhost:9200")


def main():
    es = Elasticsearch(ES_URL)

    if not es.ping():
        log.error("Cannot reach Elasticsearch at %s", ES_URL)
        sys.exit(1)

    if es.indices.exists(index=INDEX_NAME):
        props = set(es.indices.get_mapping(index=INDEX_NAME)[INDEX_NAME]["mappings"]
                    .get("properties", {}))
        stale = sorted(props & FORBIDDEN_FIELDS)
        if stale:
            log.error("Index '%s' maps fields owned by other services: %s. "
                      "Delete it (or run `docker compose down -v`) and reload.",
                      INDEX_NAME, stale)
            sys.exit(1)
        log.info("Index '%s' already exists — no action needed", INDEX_NAME)
    else:
        es.indices.create(index=INDEX_NAME, body=INDEX_BODY)
        log.info("Index '%s' created successfully", INDEX_NAME)

    info = es.indices.get(index=INDEX_NAME)
    log.info("Index settings confirmed: shards=%s",
             info[INDEX_NAME]["settings"]["index"]["number_of_shards"])
    log.info("Setup complete")


if __name__ == "__main__":
    main()
