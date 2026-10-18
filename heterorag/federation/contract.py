"""
heterorag/federation/contract.py
================================
What each store owns and how an entity is identified in it (Interface Specification v2.1,
sections 4 and 5; closes gap G11, the identifier contract).

The catalogue is built from the Layer 1 service descriptors, so ownership has one source.
The translator prompts and the step validator both read it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .plan import Store

# How an entity of each type is identified in each store (first returned column or field).
ID_CONTRACT: dict[Store, dict[str, str]] = {
    Store.SQL: {
        "user": "users.id",
        "question": "posts.id (restrict to post_type_id = 1)",
        "answer": "posts.id (restrict to post_type_id = 2)",
        "tag": "tags.tag_name",
    },
    Store.GRAPH: {
        "user": "User.id",
        "question": "Question.id",
        "answer": "Answer.id",
        "tag": "Tag.name",
    },
    Store.DOC: {
        "question": "documents with doc_type 'question'; identifier field post_id",
        "answer": "documents with doc_type 'answer'; identifier field post_id",
        "user": "documents with doc_type 'user_about'; identifier field user_id",
        "tag": "documents with doc_type 'tag_wiki'; identifier field tag_name",
    },
}

# Node label and SQL table that own each entity's identifier (used by the entity-type checks).
LABEL_OF_ENTITY = {"user": "User", "question": "Question", "answer": "Answer", "tag": "Tag"}
SQL_TABLE_OF_ENTITY = {"user": "users", "question": "posts", "answer": "posts", "tag": "tags"}

# Document type that carries each entity's identifier in the document store.
DOC_TYPE_OF_ENTITY = {"question": "question", "answer": "answer", "user": "user_about", "tag": "tag_wiki"}


_SQL_TEXT_TYPES = {"text", "varchar", "character varying", "char", "character", "citext"}


@dataclass(frozen=True)
class SqlCatalogue:
    tables: Mapping[str, tuple[str, ...]]            # table -> columns
    text_columns: frozenset = frozenset()            # columns of a text type (compared without regard to case)


@dataclass(frozen=True)
class GraphCatalogue:
    labels: frozenset
    relationships: frozenset
    properties: frozenset


@dataclass(frozen=True)
class DocCatalogue:
    doc_types: frozenset
    fields: frozenset
    keyword_fields: frozenset = frozenset()          # exact-value text fields (case-sensitive unless told otherwise)


@dataclass(frozen=True)
class Catalogue:
    sql: SqlCatalogue
    graph: GraphCatalogue
    doc: DocCatalogue


def catalogue_from_descriptors(sql, graph, doc) -> Catalogue:
    """Build the ownership catalogue from the three Layer 1 descriptors."""
    s, g, d = sql.schema_spec, graph.schema_spec, doc.schema_spec
    return Catalogue(
        sql=SqlCatalogue({t.table_name: tuple(c.name for c in t.columns) for t in s.tables},
                         frozenset(c.name for t in s.tables for c in t.columns
                                   if c.data_type.lower().split("(")[0].strip() in _SQL_TEXT_TYPES)),
        graph=GraphCatalogue(
            labels=frozenset(n.label for n in g.node_types),
            relationships=frozenset(e.edge_type for e in g.edge_types),
            properties=frozenset(p for n in g.node_types for p in n.properties)
            | frozenset(p for e in g.edge_types for p in e.properties),
        ),
        doc=DocCatalogue(
            doc_types=frozenset(t.doc_type for t in d.doc_types),
            fields=frozenset(f.field_name for t in d.doc_types for f in t.fields),
            keyword_fields=frozenset(f.field_name for t in d.doc_types for f in t.fields
                                     if f.field_type == "keyword"),
        ),
    )


def poc_catalogue() -> Catalogue:
    from ..layer1.poc_descriptors import (CONTENT_SERVICE_DESCRIPTOR, KNOWLEDGE_GRAPH_DESCRIPTOR,
                                          USER_ACTIVITY_DESCRIPTOR)
    return catalogue_from_descriptors(USER_ACTIVITY_DESCRIPTOR, KNOWLEDGE_GRAPH_DESCRIPTOR,
                                      CONTENT_SERVICE_DESCRIPTOR)
