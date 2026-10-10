"""
heterorag/layer1/poc_descriptors.py
=====================================
Concrete ServiceDescriptors for the three POC services.

These descriptors encode the exact SchemaSpec defined in Foundation Doc v1.6 §7.2.
They are the authoritative machine-readable representation of what each service owns.

Usage:
    from heterorag.layer1.poc_descriptors import build_poc_registry

    registry = build_poc_registry(embed_fn=my_embed_fn, llm_confirm_fn=my_confirm_fn)
    # registry is ready — all three services pre-registered

The three services:
    Service 1 — user-activity-service    (PostgreSQL / SQL)
    Service 2 — knowledge-graph-service  (Neo4j / Graph)
    Service 3 — content-service          (Elasticsearch / Document)
"""

from __future__ import annotations

import os
from typing import Callable

from .models import (
    ColumnSpec,
    ConnectionConfig,
    DocumentFieldSpec,
    DocumentSchemaSpec,
    DocumentTypeSpec,
    EdgeTypeSpec,
    GraphSchemaSpec,
    NodeTypeSpec,
    PersistenceType,
    ServiceDescriptor,
    SQLSchemaSpec,
    TableSpec,
)
from .registry import EmbeddingFn, ServiceRegistry


# =============================================================================
# Service 1 — User & Activity Service (PostgreSQL / SQL)
# Foundation Doc §7.2: Users, Posts, Votes, Badges
# Owns (data ownership rule D2): ATTRIBUTES only -- performance metrics, counts,
#       temporal data, badge history, tag usage counts.
# Does NOT own relationships (authorship, parent/accepted answer, tags of a post,
#       post links) or any text. Identifiers are the only shared keys.
# =============================================================================

_SQL_SCHEMA_SPEC = SQLSchemaSpec(
    tables=[
        TableSpec(
            table_name="users",
            description=(
                "Stack Overflow user accounts (attributes only). "
                "Use for reputation scores, vote counts, activity dates, and locations. "
                "The free-text 'about me' is NOT stored here; it lives in the Content Service."
            ),
            columns=[
                ColumnSpec(name="id",           data_type="integer",     nullable=False,
                           description="Unique user identifier"),
                ColumnSpec(name="reputation",   data_type="integer",     nullable=False,
                           description="Community reputation score — primary quality metric for users"),
                ColumnSpec(name="creation_date",data_type="timestamptz", nullable=True,
                           description="Account creation timestamp"),
                ColumnSpec(name="display_name", data_type="text",        nullable=True),
                ColumnSpec(name="location",     data_type="text",        nullable=True,
                           description="Free-text user-supplied location (not normalised)"),
                ColumnSpec(name="up_votes",     data_type="integer",     nullable=False,
                           description="Total up-votes cast by this user"),
                ColumnSpec(name="down_votes",   data_type="integer",     nullable=False,
                           description="Total down-votes cast by this user"),
                ColumnSpec(name="views",        data_type="integer",     nullable=False,
                           description="Profile page view count"),
                ColumnSpec(name="account_id",   data_type="integer",     nullable=True),
            ],
        ),
        TableSpec(
            table_name="posts",
            description=(
                "Questions (post_type_id=1) and answers (post_type_id=2). "
                "Contains numeric metrics and dates only: score, view count, answer count, "
                "comment count, creation, activity and close dates. "
                "NOT stored here: who wrote a post, which question an answer belongs to, "
                "which answer was accepted, the tags of a post, and links between questions "
                "(all of these are relationships and live in the Knowledge Graph Service); "
                "titles and bodies live in the Content Service. "
                "To combine these metrics with authorship, tags or links, obtain the post "
                "ids from the Knowledge Graph Service and filter on posts.id."
            ),
            columns=[
                ColumnSpec(name="id",                  data_type="integer",     nullable=False,
                           description="Post identifier, shared key with the other services"),
                ColumnSpec(name="post_type_id",        data_type="smallint",    nullable=False,
                           description="1=Question, 2=Answer"),
                ColumnSpec(name="score",               data_type="integer",     nullable=False,
                           description="Net community vote score"),
                ColumnSpec(name="view_count",          data_type="integer",     nullable=False),
                ColumnSpec(name="answer_count",        data_type="integer",     nullable=False,
                           description="Answer count as recorded in the dump (a number, not the answers)"),
                ColumnSpec(name="comment_count",       data_type="integer",     nullable=False),
                ColumnSpec(name="creation_date",       data_type="timestamptz", nullable=True),
                ColumnSpec(name="last_activity_date",  data_type="timestamptz", nullable=True),
                ColumnSpec(name="closed_date",         data_type="timestamptz", nullable=True),
            ],
        ),
        TableSpec(
            table_name="votes",
            description=(
                "Individual votes cast on posts (event records: each row names the post "
                "and, where recorded, the voting user). "
                "vote_type_id: 1=AcceptedByOriginator, 2=UpMod, 3=DownMod, "
                "4=Offensive, 5=Favourite, 8=Bounty, 9=Spam, 11=ApproveEditSuggestion, "
                "12=RejectEditSuggestion, 15=HighRepCloseVote, 16=HighRepReopenVote."
            ),
            columns=[
                ColumnSpec(name="id",            data_type="integer",     nullable=False),
                ColumnSpec(name="post_id",        data_type="integer",     nullable=False),
                ColumnSpec(name="vote_type_id",   data_type="smallint",    nullable=False),
                ColumnSpec(name="creation_date",  data_type="timestamptz", nullable=True),
                ColumnSpec(name="user_id",        data_type="integer",     nullable=True),
                ColumnSpec(name="bounty_amount",  data_type="integer",     nullable=True),
            ],
        ),
        TableSpec(
            table_name="badges",
            description=(
                "Community badges awarded to users. "
                "class: 1=Gold, 2=Silver, 3=Bronze. "
                "tag_based=TRUE means the badge was earned through activity on a specific tag."
            ),
            columns=[
                ColumnSpec(name="id",       data_type="integer",     nullable=False),
                ColumnSpec(name="user_id",  data_type="integer",     nullable=False),
                ColumnSpec(name="name",     data_type="text",        nullable=False,
                           description="Badge name e.g. 'python', 'Enlightened', 'Revival'"),
                ColumnSpec(name="class",    data_type="smallint",    nullable=False,
                           description="1=Gold, 2=Silver, 3=Bronze"),
                ColumnSpec(name="tag_based",data_type="boolean",     nullable=False),
                ColumnSpec(name="date",     data_type="timestamptz", nullable=True),
            ],
        ),
        TableSpec(
            table_name="tags",
            description=(
                "Tag usage counts. Owns the count of questions per tag. "
                "Which questions carry a tag, and tag co-occurrence, live in the "
                "Knowledge Graph Service; the tag wiki text lives in the Content Service."
            ),
            columns=[
                ColumnSpec(name="id",              data_type="integer", nullable=False),
                ColumnSpec(name="tag_name",        data_type="text",    nullable=False,
                           description="Tag name, shared key with the other services"),
                ColumnSpec(name="count",           data_type="integer", nullable=False,
                           description="Number of questions tagged with this tag"),
            ],
        ),
        TableSpec(
            table_name="comments",
            description=(
                "Comment metadata (score, user, post linkage). "
                "Comment TEXT is NOT stored here; query the Content Service for comment text."
            ),
            columns=[
                ColumnSpec(name="id",            data_type="integer",     nullable=False),
                ColumnSpec(name="post_id",       data_type="integer",     nullable=False),
                ColumnSpec(name="score",         data_type="integer",     nullable=False),
                ColumnSpec(name="user_id",       data_type="integer",     nullable=True),
                ColumnSpec(name="creation_date", data_type="timestamptz", nullable=True),
            ],
        ),
    ]
)

USER_ACTIVITY_DESCRIPTOR = ServiceDescriptor(
    service_id       = "user-activity-service",
    display_name     = "User & Activity Service",
    persistence_type = PersistenceType.SQL,
    schema_spec      = _SQL_SCHEMA_SPEC,
    schema_version   = "2.0.0",
    capability_summary = (
        "Answers questions about numeric attributes of Stack Overflow users and posts. "
        "Use for: reputation scores, vote counts, badge counts and names, question and answer "
        "scores, view counts, answer and comment counts, tag usage counts, and "
        "temporal filters (e.g. posts created in 2022). "
        "Does NOT contain relationships (who wrote a post, which question an answer belongs "
        "to, accepted answers, the tags of a post, links or duplicates between questions), "
        "and does NOT contain any text (titles, bodies, comments, about-me)."
    ),
    connection = ConnectionConfig(
        host     = os.getenv("PG_HOST", "localhost"),
        port     = int(os.getenv("PG_PORT", "5432")),
        database = os.getenv("PG_DBNAME", "heterorag"),
        extra_params = {
            "user":     os.getenv("PG_USER",     "heterorag"),
            "password": os.getenv("PG_PASSWORD", "heterorag_secret"),
        },
    ),
)


# =============================================================================
# Service 2 — Knowledge Graph Service (Neo4j / Graph)
# Foundation Doc §7.2: User, Question, Answer, Tag nodes;
#   ASKED, ANSWERED, ANSWERS, ACCEPTED, TAGGED_WITH, CO_OCCURS_WITH,
#   LINKED_TO, DUPLICATE_OF edges
# Owns: connectivity, topology, traversal paths, tag relationships, duplicate chains.
# =============================================================================

_GRAPH_SCHEMA_SPEC = GraphSchemaSpec(
    node_types=[
        NodeTypeSpec(
            label       = "User",
            properties  = ["id"],
            description = "A Stack Overflow user, identified by id only. Use for traversal: "
                          "who asked or answered what. Reputation, names and other attributes "
                          "live in the User & Activity Service.",
        ),
        NodeTypeSpec(
            label       = "Question",
            properties  = ["id"],
            description = "A question post, identified by id only. Linked to its author, answers, "
                          "tags, and other questions. Score, counts, dates live in the User & "
                          "Activity Service; the title and body live in the Content Service.",
        ),
        NodeTypeSpec(
            label       = "Answer",
            properties  = ["id"],
            description = "An answer post, identified by id only. Connected to its question and "
                          "its author. Score and dates live in the User & Activity Service; the "
                          "body lives in the Content Service.",
        ),
        NodeTypeSpec(
            label       = "Tag",
            properties  = ["name"],
            description = "A tag, identified by name. Connected to questions via TAGGED_WITH and to "
                          "other tags via CO_OCCURS_WITH (weighted by question co-occurrence "
                          "frequency). The usage count lives in the User & Activity Service.",
        ),
    ],
    edge_types=[
        EdgeTypeSpec(
            edge_type   = "ASKED",
            from_label  = "User",
            to_label    = "Question",
            properties  = [],
            description = "User authored a Question.",
        ),
        EdgeTypeSpec(
            edge_type   = "ANSWERED",
            from_label  = "User",
            to_label    = "Answer",
            properties  = [],
            description = "User authored an Answer.",
        ),
        EdgeTypeSpec(
            edge_type   = "ANSWERS",
            from_label  = "Answer",
            to_label    = "Question",
            properties  = [],
            description = "This Answer addresses this Question.",
        ),
        EdgeTypeSpec(
            edge_type   = "ACCEPTED",
            from_label  = "Question",
            to_label    = "Answer",
            properties  = [],
            description = "The question owner accepted this Answer as correct.",
        ),
        EdgeTypeSpec(
            edge_type   = "TAGGED_WITH",
            from_label  = "Question",
            to_label    = "Tag",
            properties  = [],
            description = "Question is tagged with this Tag.",
        ),
        EdgeTypeSpec(
            edge_type   = "CO_OCCURS_WITH",
            from_label  = "Tag",
            to_label    = "Tag",
            properties  = ["weight"],
            description = "Two tags appear together on the same question. "
                          "weight = number of questions that carry both tags. "
                          "Undirected — use MATCH (t1)-[r:CO_OCCURS_WITH]-(t2).",
        ),
        EdgeTypeSpec(
            edge_type   = "LINKED_TO",
            from_label  = "Question",
            to_label    = "Question",
            properties  = [],
            description = "Editorial cross-reference between questions (PostLinks LinkTypeId=1).",
        ),
        EdgeTypeSpec(
            edge_type   = "DUPLICATE_OF",
            from_label  = "Question",
            to_label    = "Question",
            properties  = [],
            description = "This question was closed as a duplicate of the target (PostLinks LinkTypeId=3).",
        ),
    ],
)

KNOWLEDGE_GRAPH_DESCRIPTOR = ServiceDescriptor(
    service_id       = "knowledge-graph-service",
    display_name     = "Knowledge Graph Service",
    persistence_type = PersistenceType.GRAPH,
    schema_spec      = _GRAPH_SCHEMA_SPEC,
    schema_version   = "2.0.0",
    capability_summary = (
        "Answers questions about relationships and topology in Stack Overflow. "
        "Use for: which tags co-occur and form communities, shortest path between users, "
        "questions linked or marked as duplicates, tag neighbourhood traversal, "
        "network of who answered whose questions, graph-based ranking (PageRank), "
        "community detection on the tag graph. "
        "Owns every relationship: authorship (ASKED, ANSWERED), which question an answer belongs "
        "to (ANSWERS), accepted answers (ACCEPTED), the tags of a question (TAGGED_WITH), and links "
        "and duplicates between questions. "
        "Does NOT contain any attribute (scores, counts, dates, reputation, names) or any text."
    ),
    connection = ConnectionConfig(
        host     = os.getenv("NEO4J_HOST", "localhost"),
        port     = int(os.getenv("NEO4J_BOLT_PORT", "7687")),
        extra_params = {
            "user":     os.getenv("NEO4J_USER",     "neo4j"),
            "password": os.getenv("NEO4J_PASSWORD", "heterorag_secret"),
            "scheme":   "bolt",
        },
    ),
)


# =============================================================================
# Service 3 — Content Service (Elasticsearch / Document)
# Foundation Doc §7.2: Question bodies, Answer bodies, Comments,
#   User AboutMe, Tag wikis
# Owns: explanatory content, code, error messages, conceptual discussions,
#       user self-descriptions.
# =============================================================================

_DOCUMENT_SCHEMA_SPEC = DocumentSchemaSpec(
    index_name = "heterorag_content",
    doc_types  = [
        DocumentTypeSpec(
            doc_type    = "question",
            description = "Title and full text of Stack Overflow question posts including code blocks. "
                          "Use for: finding questions that discuss a topic, contain specific "
                          "error messages, or explain a concept. Authors, tags, scores and dates "
                          "are NOT stored here.",
            fields=[
                DocumentFieldSpec(name="doc_type",      field_type="keyword",  searchable=False),
                DocumentFieldSpec(name="post_id",       field_type="integer",  searchable=False,
                                  description="Post identifier, shared key with the other services"),
                DocumentFieldSpec(name="title",         field_type="text",     searchable=True,
                                  description="Question title — high-weight BM25 field"),
                DocumentFieldSpec(name="body",          field_type="text",     searchable=True,
                                  description="Full question body including code"),
            ],
        ),
        DocumentTypeSpec(
            doc_type    = "answer",
            description = "Full text of Stack Overflow answer posts including code. "
                          "Use for: finding explanations, code samples, solutions to errors. "
                          "The question an answer belongs to and its author are NOT stored here.",
            fields=[
                DocumentFieldSpec(name="doc_type",      field_type="keyword",  searchable=False),
                DocumentFieldSpec(name="post_id",       field_type="integer",  searchable=False),
                DocumentFieldSpec(name="body",          field_type="text",     searchable=True),
            ],
        ),
        DocumentTypeSpec(
            doc_type    = "comment",
            description = "Full text of comments on questions or answers. "
                          "Shorter than posts; use when the query specifically seeks clarifications. "
                          "Each comment names the post it is on and its author (event record).",
            fields=[
                DocumentFieldSpec(name="doc_type",      field_type="keyword",  searchable=False),
                DocumentFieldSpec(name="post_id",       field_type="integer",  searchable=False,
                                  description="The question or answer this comment is on"),
                DocumentFieldSpec(name="user_id",       field_type="integer",  searchable=False,
                                  description="The user who wrote the comment"),
                DocumentFieldSpec(name="body",          field_type="text",     searchable=True),
            ],
        ),
        DocumentTypeSpec(
            doc_type    = "user_about",
            description = "User AboutMe free-text profile descriptions. "
                          "Use for: finding users who describe themselves in a certain way.",
            fields=[
                DocumentFieldSpec(name="doc_type", field_type="keyword", searchable=False),
                DocumentFieldSpec(name="user_id",  field_type="integer", searchable=False),
                DocumentFieldSpec(name="body",     field_type="text",    searchable=True),
            ],
        ),
        DocumentTypeSpec(
            doc_type    = "tag_wiki",
            description = "Tag wiki excerpt pages explaining what a tag means and how to use it. "
                          "Use for: understanding what a technology tag represents.",
            fields=[
                DocumentFieldSpec(name="doc_type", field_type="keyword", searchable=False),
                DocumentFieldSpec(name="tag_name", field_type="keyword", searchable=False,
                                  description="Tag name, shared key with the other services"),
                DocumentFieldSpec(name="body",     field_type="text",    searchable=True),
            ],
        ),
    ],
)

CONTENT_SERVICE_DESCRIPTOR = ServiceDescriptor(
    service_id       = "content-service",
    display_name     = "Content Service",
    persistence_type = PersistenceType.DOCUMENT,
    schema_spec      = _DOCUMENT_SCHEMA_SPEC,
    schema_version   = "2.0.0",
    capability_summary = (
        "Answers questions requiring full text search over Stack Overflow content. "
        "Use for: finding posts that explain a concept, contain specific code or error messages, "
        "discuss a technology, describe how to solve a problem, "
        "or where a user describes themselves in a certain way. "
        "Contains: question titles and bodies, answer bodies, comments, user profiles, tag wiki pages. "
        "Does NOT contain attributes (scores, counts, dates) or relationships (authorship, "
        "tags of a post, which question an answer belongs to, links between questions)."
    ),
    connection = ConnectionConfig(
        host     = os.getenv("ES_HOST", "localhost"),
        port     = int(os.getenv("ES_PORT", "9200")),
        database = "heterorag_content",
        extra_params = {
            "scheme": "http",
        },
    ),
)


# =============================================================================
# Factory function — builds and pre-populates the registry for the POC
# =============================================================================

def build_poc_registry(
    embed_fn:       EmbeddingFn | None = None,
    llm_confirm_fn: Callable[[str, list[ServiceDescriptor]], list[str]] | None = None,
    heartbeat_timeout_seconds: int = 60,
) -> ServiceRegistry:
    """
    Construct a ServiceRegistry pre-populated with all three POC services.

    Args:
        embed_fn:
            Optional embedding function (str → unit-norm list[float]).
            Plugged into Stage 1 of the RelevanceFilter.
            If None, Stage 1 is skipped and all services pass to Stage 2.

        llm_confirm_fn:
            Optional LLM confirmation function (query, descriptors → list[service_id]).
            Plugged into Stage 2 of the RelevanceFilter.
            If None, Stage 2 is skipped and Stage 1 output is returned as-is.

        heartbeat_timeout_seconds:
            How long (seconds) a service can go without a heartbeat before it is
            marked inactive. Default 60s is appropriate for the POC; production
            deployments should use a longer window.

    Returns:
        A fully initialised ServiceRegistry with all three POC services registered.
    """
    registry = ServiceRegistry(
        embed_fn                  = embed_fn,
        llm_confirm_fn            = llm_confirm_fn,
        heartbeat_timeout_seconds = heartbeat_timeout_seconds,
    )

    for descriptor in [
        USER_ACTIVITY_DESCRIPTOR,
        KNOWLEDGE_GRAPH_DESCRIPTOR,
        CONTENT_SERVICE_DESCRIPTOR,
    ]:
        registry.register(descriptor)

    return registry
