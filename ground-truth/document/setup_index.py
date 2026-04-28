#!/usr/bin/env python3
"""
HeteroRAG POC — Elasticsearch Index Setup
RUNBOOK Step 5: python ground-truth/document/setup_index.py

Creates the heterorag_content index with correct mappings if it does not exist.
Safe to re-run — idempotent.
"""

import os
import sys
import logging
from elasticsearch import Elasticsearch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("heterorag.es_setup")

ES_URL    = os.getenv("ES_URL", "http://localhost:9200")
INDEX     = "heterorag_content"

INDEX_BODY = {
    "settings": {
        "number_of_shards":   1,
        "number_of_replicas": 0,
        "analysis": {
            "filter": {
                "english_stop": {"type": "stop", "stopwords": "_english_"},
                "english_stemmer": {"type": "stemmer", "language": "english"},
            },
            "analyzer": {
                "english_custom": {
                    "tokenizer": "standard",
                    "filter": ["lowercase", "english_stop", "english_stemmer"],
                }
            }
        }
    },
    "mappings": {
        "properties": {
            # Routing / classification
            "doc_type":       {"type": "keyword"},        # question|answer|comment|user_about|tag_wiki
            "post_id":        {"type": "integer"},
            "parent_id":      {"type": "integer"},
            "user_id":        {"type": "integer"},
            "score":          {"type": "integer"},
            "creation_date":  {"type": "date", "format": "strict_date_optional_time||epoch_millis"},
            # Full-text fields — BM25 retrieval targets
            "title":          {"type": "text", "analyzer": "english_custom"},
            "body":           {"type": "text", "analyzer": "english_custom"},
            # Structured fields
            "tags":           {"type": "keyword"},        # per-question tag list
            "tag_name":       {"type": "keyword"},        # for tag_wiki docs
        }
    }
}


def main():
    es = Elasticsearch(ES_URL)

    if not es.ping():
        log.error("Cannot reach Elasticsearch at %s", ES_URL)
        sys.exit(1)

    if es.indices.exists(index=INDEX):
        log.info("Index '%s' already exists — no action needed", INDEX)
    else:
        es.indices.create(index=INDEX, body=INDEX_BODY)
        log.info("Index '%s' created successfully", INDEX)

    # Verify
    info = es.indices.get(index=INDEX)
    log.info("Index settings confirmed: shards=%s",
             info[INDEX]["settings"]["index"]["number_of_shards"])
    log.info("Setup complete")


if __name__ == "__main__":
    main()
