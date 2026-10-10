"""
heterorag/es_index.py
=====================
Single definition of the Elasticsearch index used by the Content Service.

Both ``scripts/load_stackoverflow_dump.py`` and ``ground-truth/document/setup_index.py``
import this module. Before, each carried its own mapping and analyzer, so the analysis
chain of the index (and therefore BM25 behaviour) depended on which script ran first.

Data ownership rule (D2): the index holds TEXT only. Question and answer documents carry
the post id, question documents also the title; comments carry post id and user id (event
identifiers, D10); user_about documents carry the user id; tag wikis carry the tag name.
Scores, dates, authorship, parents and tags are owned by the other services.
"""

from __future__ import annotations

INDEX_NAME = "heterorag_content"

# Fields that must NOT appear in the mapping (they would duplicate data owned elsewhere).
FORBIDDEN_FIELDS = frozenset({"parent_id", "score", "creation_date", "tags"})

INDEX_BODY = {
    "settings": {
        "number_of_shards":   1,
        "number_of_replicas": 0,
        "analysis": {
            "analyzer": {
                "english": {
                    "type":      "standard",
                    "stopwords": "_english_",
                }
            }
        },
    },
    "mappings": {
        "properties": {
            "doc_type": {"type": "keyword"},   # question|answer|comment|user_about|tag_wiki
            "post_id":  {"type": "integer"},   # questions, answers, comments
            "user_id":  {"type": "integer"},   # comments and user_about
            "tag_name": {"type": "keyword"},   # tag wikis
            "title":    {"type": "text", "analyzer": "english"},
            "body":     {"type": "text", "analyzer": "english"},
        }
    },
}
