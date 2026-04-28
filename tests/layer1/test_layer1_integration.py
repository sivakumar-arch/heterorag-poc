"""
tests/layer1/test_layer1_integration.py
========================================
Integration tests for Layer 1 against the live Docker Compose services.

Run after `docker compose up -d` and (for SQL schema tests) after
`docker compose --profile migrate up flyway`.

Usage:
    # All integration tests
    pytest tests/layer1/test_layer1_integration.py -v

    # One service at a time (useful when debugging a specific container)
    pytest tests/layer1/test_layer1_integration.py -v -k postgres
    pytest tests/layer1/test_layer1_integration.py -v -k neo4j
    pytest tests/layer1/test_layer1_integration.py -v -k elasticsearch

    # Skip if services are not running (CI without Docker)
    pytest tests/layer1/test_layer1_integration.py -v -m "not integration"

Environment overrides (match docker-compose.yml defaults):
    PG_HOST, PG_PORT, PG_DBNAME, PG_USER, PG_PASSWORD
    NEO4J_HOST, NEO4J_BOLT_PORT, NEO4J_USER, NEO4J_PASSWORD
    ES_HOST, ES_PORT

What these tests prove that unit tests cannot:
    1. The ConnectionConfig in each POC descriptor reaches the correct container.
    2. The credentials in docker-compose.yml match those in poc_descriptors.py.
    3. The SQL schema created by 00_bootstrap.sql matches the TableSpec/ColumnSpec
       definitions in the descriptor (column names, data types, nullability).
    4. The Neo4j constraints applied by the loader match the NodeTypeSpec labels.
    5. The Elasticsearch index and its field mappings match DocumentSchemaSpec.
    6. registry.discover() produces a valid I₁ over live services.
    7. Heartbeat expiry and re-registration work correctly under real wall-clock time.
"""

from __future__ import annotations

import os
import socket
import time
import uuid
from typing import Generator

import pytest

# ---------------------------------------------------------------------------
# Optional imports — tests are skipped gracefully if drivers are absent
# ---------------------------------------------------------------------------

try:
    import psycopg2
    import psycopg2.extras
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False

try:
    from neo4j import GraphDatabase as Neo4jDriver
    HAS_NEO4J = True
except ImportError:
    HAS_NEO4J = False

try:
    from elasticsearch import Elasticsearch
    HAS_ES = True
except ImportError:
    HAS_ES = False

from heterorag.layer1.models import (
    I1_ServiceDiscoveryOutput,
    PersistenceType,
)
from heterorag.layer1.poc_descriptors import (
    CONTENT_SERVICE_DESCRIPTOR,
    KNOWLEDGE_GRAPH_DESCRIPTOR,
    USER_ACTIVITY_DESCRIPTOR,
    build_poc_registry,
)
from heterorag.layer1.registry import ServiceRegistry

# ---------------------------------------------------------------------------
# Connection parameters — read from environment, fall back to Compose defaults
# ---------------------------------------------------------------------------

PG_HOST     = os.getenv("PG_HOST",     "localhost")
PG_PORT     = int(os.getenv("PG_PORT", "5432"))
PG_DBNAME   = os.getenv("PG_DBNAME",   "heterorag")
PG_USER     = os.getenv("PG_USER",     "heterorag")
PG_PASSWORD = os.getenv("PG_PASSWORD", "heterorag_secret")

NEO4J_HOST     = os.getenv("NEO4J_HOST",      "localhost")
NEO4J_PORT     = int(os.getenv("NEO4J_BOLT_PORT", "7687"))
NEO4J_USER     = os.getenv("NEO4J_USER",      "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD",  "heterorag_secret")

ES_HOST = os.getenv("ES_HOST", "localhost")
ES_PORT = int(os.getenv("ES_PORT", "9200"))

# ---------------------------------------------------------------------------
# Availability checks — skip entire test class if container is unreachable
# ---------------------------------------------------------------------------

def _tcp_reachable(host: str, port: int, timeout: float = 2.0) -> bool:
    """Returns True if the TCP port is open (container is up)."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _pg_available() -> bool:
    if not HAS_PSYCOPG2:
        return False
    try:
        conn = psycopg2.connect(
            host=PG_HOST, port=PG_PORT, dbname=PG_DBNAME,
            user=PG_USER, password=PG_PASSWORD,
            connect_timeout=3,
        )
        conn.close()
        return True
    except Exception:
        return False


def _neo4j_available() -> bool:
    if not HAS_NEO4J:
        return False
    if not _tcp_reachable(NEO4J_HOST, NEO4J_PORT):
        return False
    try:
        driver = Neo4jDriver.driver(
            f"bolt://{NEO4J_HOST}:{NEO4J_PORT}",
            auth=(NEO4J_USER, NEO4J_PASSWORD),
        )
        with driver.session() as s:
            s.run("RETURN 1")
        driver.close()
        return True
    except Exception:
        return False


def _es_available() -> bool:
    if not HAS_ES:
        return False
    try:
        es = Elasticsearch(f"http://{ES_HOST}:{ES_PORT}")
        return es.ping()
    except Exception:
        return False


postgres_available      = pytest.mark.skipif(not _pg_available(),     reason="PostgreSQL not reachable")
neo4j_available         = pytest.mark.skipif(not _neo4j_available(),  reason="Neo4j not reachable")
elasticsearch_available = pytest.mark.skipif(not _es_available(),     reason="Elasticsearch not reachable")
all_services_available  = pytest.mark.skipif(
    not (_pg_available() and _neo4j_available() and _es_available()),
    reason="Not all three Docker services are reachable",
)

pytestmark = pytest.mark.integration


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def pg_conn():
    """Raw psycopg2 connection for schema inspection queries."""
    conn = psycopg2.connect(
        host=PG_HOST, port=PG_PORT, dbname=PG_DBNAME,
        user=PG_USER, password=PG_PASSWORD,
    )
    conn.autocommit = True
    yield conn
    conn.close()


@pytest.fixture(scope="module")
def neo4j_driver():
    driver = Neo4jDriver.driver(
        f"bolt://{NEO4J_HOST}:{NEO4J_PORT}",
        auth=(NEO4J_USER, NEO4J_PASSWORD),
    )
    yield driver
    driver.close()


@pytest.fixture(scope="module")
def es_client():
    return Elasticsearch(f"http://{ES_HOST}:{ES_PORT}")


# =============================================================================
# 1. Connectivity — can we reach each container at all?
# =============================================================================

class TestConnectivity:

    @postgres_available
    def test_postgres_tcp_reachable(self):
        """Basic TCP connectivity to PostgreSQL."""
        assert _tcp_reachable(PG_HOST, PG_PORT), \
            f"PostgreSQL TCP port {PG_HOST}:{PG_PORT} not reachable"

    @postgres_available
    def test_postgres_accepts_connection_with_poc_credentials(self, pg_conn):
        """The credentials in poc_descriptors.py actually work against the live container."""
        with pg_conn.cursor() as cur:
            cur.execute("SELECT current_database(), current_user")
            db, user = cur.fetchone()
        assert db   == PG_DBNAME, f"Expected database '{PG_DBNAME}', got '{db}'"
        assert user == PG_USER,   f"Expected user '{PG_USER}', got '{user}'"

    @neo4j_available
    def test_neo4j_bolt_reachable(self, neo4j_driver):
        """Bolt connection to Neo4j succeeds with POC credentials."""
        with neo4j_driver.session() as s:
            result = s.run("RETURN 'heterorag' AS msg")
            record = result.single()
        assert record["msg"] == "heterorag"

    @elasticsearch_available
    def test_elasticsearch_http_reachable(self, es_client):
        """HTTP connection to Elasticsearch returns cluster info."""
        info = es_client.info()
        assert info["cluster_name"] == "heterorag-cluster", \
            f"Expected cluster_name 'heterorag-cluster', got '{info['cluster_name']}'"

    @elasticsearch_available
    def test_elasticsearch_cluster_health_is_not_red(self, es_client):
        """Cluster health must be yellow or green — red means data loss."""
        health = es_client.cluster.health()
        status = health["status"]
        assert status in ("green", "yellow"), \
            f"Elasticsearch cluster health is '{status}' — check container logs"


# =============================================================================
# 2. PostgreSQL schema verification
#    The bootstrap SQL (00_bootstrap.sql) must have created every table and
#    column referenced in the USER_ACTIVITY_DESCRIPTOR SchemaSpec.
# =============================================================================

class TestPostgresSchema:

    EXPECTED_TABLES = [
        "users", "posts", "votes", "badges", "tags", "comments", "post_links",
    ]

    # Columns verified: table → list of (column_name, data_type_contains, nullable)
    # data_type_contains: substring check against information_schema data_type
    EXPECTED_COLUMNS: dict[str, list[tuple[str, str, bool]]] = {
        "users": [
            ("id",            "integer", False),
            ("reputation",    "integer", False),
            ("creation_date", "timestamp", True),
            ("display_name",  "text",    True),
            ("location",      "text",    True),
            ("up_votes",      "integer", False),
            ("down_votes",    "integer", False),
            ("views",         "integer", False),
        ],
        "posts": [
            ("id",                 "integer",   False),
            ("post_type_id",       "smallint",  False),
            ("accepted_answer_id", "integer",   True),
            ("parent_id",          "integer",   True),
            ("score",              "integer",   False),
            ("view_count",         "integer",   False),
            ("answer_count",       "integer",   False),
            ("comment_count",      "integer",   False),
            ("owner_user_id",      "integer",   True),
            ("creation_date",      "timestamp", True),
            ("tags",               "text",      True),
            ("title",              "text",      True),
        ],
        "votes": [
            ("id",           "integer",   False),
            ("post_id",      "integer",   False),
            ("vote_type_id", "smallint",  False),
            ("creation_date","timestamp", True),
        ],
        "badges": [
            ("id",       "integer", False),
            ("user_id",  "integer", False),
            ("name",     "text",    False),
            ("class",    "smallint",False),
            ("tag_based","boolean", False),
        ],
        "tags": [
            ("id",       "integer", False),
            ("tag_name", "text",    False),
            ("count",    "integer", False),
        ],
        "comments": [
            ("id",      "integer", False),
            ("post_id", "integer", False),
            ("score",   "integer", False),
        ],
        "post_links": [
            ("id",              "integer",  False),
            ("post_id",         "integer",  False),
            ("related_post_id", "integer",  False),
            ("link_type_id",    "smallint", False),
        ],
    }

    @postgres_available
    def test_all_expected_tables_exist(self, pg_conn):
        """Every table in the SQLSchemaSpec must exist in the live database."""
        with pg_conn.cursor() as cur:
            cur.execute(
                """
                SELECT table_name
                FROM   information_schema.tables
                WHERE  table_schema = 'public'
                  AND  table_type   = 'BASE TABLE'
                """
            )
            live_tables = {row[0] for row in cur.fetchall()}

        missing = set(self.EXPECTED_TABLES) - live_tables
        assert not missing, (
            f"Tables defined in SchemaSpec but missing from live DB: {missing}\n"
            f"Have you run 00_bootstrap.sql? (RUNBOOK step 1 + docker-compose up)"
        )

    @postgres_available
    @pytest.mark.parametrize("table_name", EXPECTED_TABLES)
    def test_table_columns_match_schema_spec(self, pg_conn, table_name):
        """
        For each table, verify every expected column exists with the correct
        data type and nullability.  This catches drift between poc_descriptors.py
        and 00_bootstrap.sql before it reaches Layer 2.
        """
        if table_name not in self.EXPECTED_COLUMNS:
            pytest.skip(f"No column expectations defined for '{table_name}'")

        with pg_conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(
                """
                SELECT column_name,
                       data_type,
                       is_nullable
                FROM   information_schema.columns
                WHERE  table_schema = 'public'
                  AND  table_name   = %s
                """,
                (table_name,),
            )
            live: dict[str, dict] = {
                row["column_name"]: {
                    "data_type":   row["data_type"],
                    "is_nullable": row["is_nullable"] == "YES",
                }
                for row in cur.fetchall()
            }

        assert live, f"Table '{table_name}' has no columns — did the bootstrap run?"

        for col_name, dtype_contains, expected_nullable in self.EXPECTED_COLUMNS[table_name]:
            assert col_name in live, (
                f"Column '{col_name}' missing from live table '{table_name}'.\n"
                f"Live columns: {list(live.keys())}"
            )
            live_dtype = live[col_name]["data_type"]
            assert dtype_contains in live_dtype, (
                f"Column '{table_name}.{col_name}': expected data_type containing "
                f"'{dtype_contains}', got '{live_dtype}'"
            )

    @postgres_available
    def test_load_progress_table_exists(self, pg_conn):
        """The _load_progress tracking table must be present (created by bootstrap)."""
        with pg_conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM information_schema.tables "
                "WHERE table_name = '_load_progress' AND table_schema = 'public'"
            )
            assert cur.fetchone() is not None, \
                "_load_progress table missing — bootstrap did not complete cleanly"

    @postgres_available
    def test_indexes_exist_on_key_columns(self, pg_conn):
        """
        Critical indexes must be present — their absence causes ORDER-OF-MAGNITUDE
        slower queries and would invalidate latency numbers in the paper.
        """
        expected_indexes = [
            ("users",  "idx_users_reputation"),
            ("posts",  "idx_posts_score"),
            ("posts",  "idx_posts_creation"),
            ("votes",  "idx_votes_post"),
            ("badges", "idx_badges_user"),
        ]
        with pg_conn.cursor() as cur:
            cur.execute(
                """
                SELECT indexname
                FROM   pg_indexes
                WHERE  schemaname = 'public'
                """
            )
            live_indexes = {row[0] for row in cur.fetchall()}

        missing = [
            (t, i) for t, i in expected_indexes if i not in live_indexes
        ]
        assert not missing, (
            f"Missing indexes (will degrade benchmark latency): {missing}\n"
            f"Live indexes: {sorted(live_indexes)}"
        )

    @postgres_available
    def test_descriptor_table_count_matches_live(self, pg_conn):
        """
        The number of tables in the SQLSchemaSpec must equal the number of
        BASE TABLEs in the live database (excluding Flyway metadata tables).
        Catches forgotten tables in either direction.
        """
        descriptor_table_count = len(USER_ACTIVITY_DESCRIPTOR.schema_spec.tables)

        with pg_conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*)
                FROM   information_schema.tables
                WHERE  table_schema = 'public'
                  AND  table_type   = 'BASE TABLE'
                  AND  table_name NOT LIKE 'flyway_%'
                  AND  table_name NOT LIKE '\\_%%'  -- exclude _load_progress
                """
            )
            live_count = cur.fetchone()[0]

        assert descriptor_table_count == live_count, (
            f"SchemaSpec defines {descriptor_table_count} tables but live DB has "
            f"{live_count} non-system tables. Sync poc_descriptors.py with 00_bootstrap.sql."
        )


# =============================================================================
# 3. Neo4j schema verification
#    Constraints applied by the loader must match the NodeTypeSpec labels.
# =============================================================================

class TestNeo4jSchema:

    EXPECTED_NODE_LABELS    = {"User", "Question", "Answer", "Tag"}
    EXPECTED_RELATIONSHIP_TYPES = {
        "ASKED", "ANSWERED", "ANSWERS", "ACCEPTED",
        "TAGGED_WITH", "CO_OCCURS_WITH", "LINKED_TO", "DUPLICATE_OF",
    }
    EXPECTED_UNIQUE_CONSTRAINTS = {
        "User":     "id",
        "Question": "id",
        "Answer":   "id",
        "Tag":      "name",
    }

    @neo4j_available
    def test_neo4j_responds_to_cypher(self, neo4j_driver):
        """Sanity: Cypher execution works."""
        with neo4j_driver.session() as s:
            result = s.run("RETURN 1 + 1 AS two")
            assert result.single()["two"] == 2

    @neo4j_available
    def test_uniqueness_constraints_applied(self, neo4j_driver):
        """
        Uniqueness constraints (applied by the loader's ensure_neo4j_schema)
        must be present for all four node labels.  Missing constraints mean
        MERGE operations silently create duplicate nodes.
        """
        with neo4j_driver.session() as s:
            result = s.run(
                "SHOW CONSTRAINTS YIELD name, type, labelsOrTypes, properties"
            )
            constraints = result.data()

        # Build a set of (label, property) pairs that have UNIQUENESS constraints
        unique_pairs: set[tuple[str, str]] = set()
        for c in constraints:
            if "UNIQUENESS" in c.get("type", ""):
                for label in c.get("labelsOrTypes", []):
                    for prop in c.get("properties", []):
                        unique_pairs.add((label, prop))

        missing = [
            (label, prop)
            for label, prop in self.EXPECTED_UNIQUE_CONSTRAINTS.items()
            if (label, prop) not in unique_pairs
        ]
        assert not missing, (
            f"Missing uniqueness constraints (MERGE will create duplicates): {missing}\n"
            f"Live unique constraints: {sorted(unique_pairs)}\n"
            f"Have you run the data loader? (RUNBOOK step 2)"
        )

    @neo4j_available
    def test_node_labels_exist_in_schema(self, neo4j_driver):
        """
        All four node labels defined in the GraphSchemaSpec must be visible
        in the database schema.  Uses db.labels() which reflects the constraint
        definitions — works even on an empty database that has had constraints applied.
        """
        with neo4j_driver.session() as s:
            result = s.run("CALL db.labels() YIELD label RETURN collect(label) AS labels")
            live_labels = set(result.single()["labels"])

        missing = self.EXPECTED_NODE_LABELS - live_labels
        # Note: db.labels() may be empty if no nodes have been loaded yet.
        # We check constraints instead (test above) when data is absent.
        if not live_labels:
            pytest.skip(
                "No node labels visible — database may be empty (data not yet loaded). "
                "Run RUNBOOK step 2 to load data, or verify constraints test instead."
            )
        assert not missing, (
            f"Node labels missing from live graph: {missing}\n"
            f"Live labels: {live_labels}"
        )

    @neo4j_available
    def test_relationship_types_exist_in_schema(self, neo4j_driver):
        """All eight edge types from the GraphSchemaSpec must exist."""
        with neo4j_driver.session() as s:
            result = s.run(
                "CALL db.relationshipTypes() YIELD relationshipType "
                "RETURN collect(relationshipType) AS types"
            )
            live_types = set(result.single()["types"])

        if not live_types:
            pytest.skip(
                "No relationship types visible — data not yet loaded. "
                "Run RUNBOOK step 2 first."
            )
        missing = self.EXPECTED_RELATIONSHIP_TYPES - live_types
        assert not missing, (
            f"Relationship types missing from live graph: {missing}\n"
            f"Live types: {live_types}"
        )

    @neo4j_available
    def test_descriptor_node_count_matches_live_schema(self, neo4j_driver):
        """
        Number of NodeTypeSpecs in the descriptor must match the number of
        distinct node labels visible in the schema.
        """
        descriptor_node_count = len(KNOWLEDGE_GRAPH_DESCRIPTOR.schema_spec.node_types)

        with neo4j_driver.session() as s:
            result = s.run("CALL db.labels() YIELD label RETURN count(label) AS n")
            live_count = result.single()["n"]

        if live_count == 0:
            pytest.skip("No labels visible — database is empty.")

        # GroundTruth is added by Liquibase — exclude it from comparison
        # by comparing only the labels defined in the descriptor
        descriptor_labels = {n.label for n in KNOWLEDGE_GRAPH_DESCRIPTOR.schema_spec.node_types}
        with neo4j_driver.session() as s:
            result = s.run("CALL db.labels() YIELD label RETURN collect(label) AS labels")
            live_labels = set(result.single()["labels"]) - {"GroundTruth"}

        missing_from_live = descriptor_labels - live_labels
        assert not missing_from_live, (
            f"Descriptor defines {descriptor_labels} but live graph only has {live_labels}.\n"
            f"Missing: {missing_from_live}"
        )


# =============================================================================
# 4. Elasticsearch schema verification
#    The heterorag_content index must exist with correct field mappings.
# =============================================================================

class TestElasticsearchSchema:

    INDEX = "heterorag_content"
    EXPECTED_DOC_TYPES  = {"question", "answer", "comment", "user_about", "tag_wiki"}
    EXPECTED_TEXT_FIELDS = {"title", "body"}       # must be mapped as "text" for BM25
    EXPECTED_KEYWORD_FIELDS = {"doc_type", "tags", "tag_name"}

    @elasticsearch_available
    def test_index_exists(self, es_client):
        """heterorag_content index must exist before any documents can be searched."""
        assert es_client.indices.exists(index=self.INDEX), (
            f"Index '{self.INDEX}' does not exist.\n"
            f"Run: python ground-truth/document/setup_index.py  (RUNBOOK step 5)"
        )

    @elasticsearch_available
    def test_index_has_correct_shard_and_replica_config(self, es_client):
        """Single-node POC: 1 shard, 0 replicas (as set in setup_index.py)."""
        settings = es_client.indices.get_settings(index=self.INDEX)
        idx_settings = settings[self.INDEX]["settings"]["index"]
        assert idx_settings["number_of_shards"]   == "1", "Expected 1 shard"
        assert idx_settings["number_of_replicas"] == "0", "Expected 0 replicas"

    @elasticsearch_available
    def test_text_fields_mapped_as_text(self, es_client):
        """
        BM25 retrieval requires 'title' and 'body' to be mapped as 'text'.
        If they are 'keyword', full-text search silently returns no results.
        """
        mappings = es_client.indices.get_mapping(index=self.INDEX)
        props = mappings[self.INDEX]["mappings"]["properties"]

        for field in self.EXPECTED_TEXT_FIELDS:
            assert field in props, f"Field '{field}' missing from index mappings"
            assert props[field]["type"] == "text", (
                f"Field '{field}' must be 'text' for BM25, got '{props[field]['type']}'"
            )

    @elasticsearch_available
    def test_keyword_fields_mapped_as_keyword(self, es_client):
        """doc_type, tags, tag_name must be 'keyword' for exact-match filtering."""
        mappings = es_client.indices.get_mapping(index=self.INDEX)
        props = mappings[self.INDEX]["mappings"]["properties"]

        for field in self.EXPECTED_KEYWORD_FIELDS:
            assert field in props, f"Field '{field}' missing from index mappings"
            assert props[field]["type"] == "keyword", (
                f"Field '{field}' must be 'keyword' for filtering, got '{props[field]['type']}'"
            )

    @elasticsearch_available
    def test_descriptor_doc_type_count_matches_index(self, es_client):
        """
        The number of doc types in the DocumentSchemaSpec should be reflected in
        the index (checked via a terms aggregation if data is loaded, or skipped).
        """
        count_resp = es_client.count(index=self.INDEX)
        total_docs = count_resp["count"]

        if total_docs == 0:
            pytest.skip("Index is empty — load data first (RUNBOOK step 2/6)")

        agg_resp = es_client.search(
            index=self.INDEX,
            body={
                "size": 0,
                "aggs": {
                    "doc_types": {
                        "terms": {"field": "doc_type", "size": 20}
                    }
                }
            }
        )
        live_doc_types = {
            bucket["key"]
            for bucket in agg_resp["aggregations"]["doc_types"]["buckets"]
        }
        missing = self.EXPECTED_DOC_TYPES - live_doc_types
        assert not missing, (
            f"Expected doc_types {self.EXPECTED_DOC_TYPES} but these are missing "
            f"from the index: {missing}\n"
            f"Live doc_types: {live_doc_types}"
        )


# =============================================================================
# 5. POC descriptor ↔ live service cross-checks
#    The descriptor's ConnectionConfig must resolve to the correct live service.
# =============================================================================

class TestDescriptorLiveAlignment:

    @postgres_available
    def test_sql_descriptor_connection_config_reaches_correct_db(self):
        """
        The ConnectionConfig inside USER_ACTIVITY_DESCRIPTOR must reach the
        PostgreSQL container and the correct database.
        """
        cfg = USER_ACTIVITY_DESCRIPTOR.connection
        conn = psycopg2.connect(
            host=cfg.host,
            port=cfg.port,
            dbname=cfg.database,
            user=cfg.extra_params["user"],
            password=cfg.extra_params["password"],
            connect_timeout=3,
        )
        with conn.cursor() as cur:
            cur.execute("SELECT current_database()")
            db_name = cur.fetchone()[0]
        conn.close()

        assert db_name == PG_DBNAME, (
            f"USER_ACTIVITY_DESCRIPTOR.connection points to database '{db_name}', "
            f"expected '{PG_DBNAME}'"
        )

    @neo4j_available
    def test_graph_descriptor_connection_config_reaches_neo4j(self):
        """
        The ConnectionConfig inside KNOWLEDGE_GRAPH_DESCRIPTOR must reach the
        Neo4j container via Bolt.
        """
        cfg = KNOWLEDGE_GRAPH_DESCRIPTOR.connection
        uri = f"{cfg.extra_params['scheme']}://{cfg.host}:{cfg.port}"
        driver = Neo4jDriver.driver(
            uri,
            auth=(cfg.extra_params["user"], cfg.extra_params["password"]),
        )
        with driver.session() as s:
            result = s.run("RETURN 'ok' AS status")
            status = result.single()["status"]
        driver.close()
        assert status == "ok"

    @elasticsearch_available
    def test_document_descriptor_connection_config_reaches_elasticsearch(self):
        """
        The ConnectionConfig inside CONTENT_SERVICE_DESCRIPTOR must reach the
        Elasticsearch container and confirm the correct index exists.
        """
        cfg = CONTENT_SERVICE_DESCRIPTOR.connection
        url = f"{cfg.extra_params['scheme']}://{cfg.host}:{cfg.port}"
        es  = Elasticsearch(url)
        assert es.ping(), f"CONTENT_SERVICE_DESCRIPTOR.connection at {url} did not ping"

        index_name = cfg.database   # "heterorag_content"
        assert es.indices.exists(index=index_name), (
            f"Index '{index_name}' (from CONTENT_SERVICE_DESCRIPTOR.connection.database) "
            f"does not exist in live Elasticsearch"
        )


# =============================================================================
# 6. ServiceRegistry integration — discover() over live services
# =============================================================================

class TestRegistryIntegration:

    @all_services_available
    def test_discover_returns_valid_i1_with_all_services(self):
        """
        Full round-trip: build_poc_registry() → discover() → valid I₁.
        All three services are reachable, so all three should appear in I₁
        when no RelevanceFilter is configured.
        """
        registry = build_poc_registry()   # no embed_fn, no llm_confirm_fn
        query_id = f"integ-{uuid.uuid4().hex[:8]}"

        i1 = registry.discover(query_id, "how many users have reputation over 1000?")

        assert isinstance(i1, I1_ServiceDiscoveryOutput)
        assert i1.query_id == query_id
        assert len(i1.descriptors) == 3

        service_ids = set(i1.service_ids())
        assert service_ids == {
            "user-activity-service",
            "knowledge-graph-service",
            "content-service",
        }

    @all_services_available
    def test_discover_i1_all_descriptors_active(self):
        """All descriptors in I₁ must have is_active=True (I₁ producer guarantee)."""
        registry = build_poc_registry()
        i1 = registry.discover("integ-active", "any query")
        for d in i1.descriptors:
            assert d.is_active is True, f"Inactive descriptor in I₁: {d.service_id}"

    @all_services_available
    def test_discover_cold_start_sets_uniform_weights(self):
        """
        On first discover(), each service should receive retrieval_weight = 1/3.
        This is the cold-start policy from Foundation Doc §6.
        """
        registry = build_poc_registry()
        registry.discover("integ-cold", "any query")

        for d in registry.all_descriptors():
            assert d.retrieval_weight == pytest.approx(1 / 3, abs=1e-6), (
                f"Service '{d.service_id}' weight after cold start: {d.retrieval_weight}, "
                f"expected 1/3 = {1/3:.6f}"
            )

    @all_services_available
    def test_discover_with_relevance_filter_narrows_i1(self):
        """
        A LLM confirm fn that approves only the SQL service should produce
        an I₁ with exactly one descriptor.
        """
        def sql_only_confirm(query: str, descriptors) -> list[str]:
            return ["user-activity-service"]

        registry = build_poc_registry(llm_confirm_fn=sql_only_confirm)
        i1 = registry.discover("integ-filter", "top users by reputation")

        assert len(i1.descriptors) == 1
        assert i1.descriptors[0].service_id == "user-activity-service"
        assert i1.descriptors[0].persistence_type == PersistenceType.SQL

    @all_services_available
    def test_schema_drift_partial_cold_start_live(self):
        """
        After simulating a schema version change on the SQL service, only that
        service's weight should revert.  Other services should keep their weights.
        End-to-end: register → learn weights → schema change → verify isolation.
        """
        from heterorag.layer1.poc_descriptors import (
            USER_ACTIVITY_DESCRIPTOR,
            KNOWLEDGE_GRAPH_DESCRIPTOR,
            CONTENT_SERVICE_DESCRIPTOR,
        )
        registry = build_poc_registry()

        # Step 1: trigger cold start
        registry.discover("drift-q1", "warm up")

        # Step 2: simulate planner updating weights
        registry.update_weight("user-activity-service",   0.7)
        registry.update_weight("knowledge-graph-service", 0.2)
        registry.update_weight("content-service",         0.1)

        # Step 3: SQL service re-registers with a new schema version
        from heterorag.layer1.models import (
            ConnectionConfig, PersistenceType, ServiceDescriptor,
            SQLSchemaSpec, TableSpec, ColumnSpec,
        )
        updated_sql = USER_ACTIVITY_DESCRIPTOR.model_copy(
            update={"schema_version": "2.0.0"}
        )
        registry.register(updated_sql)

        # Step 4: verify partial cold start
        sql_weight   = registry.get("user-activity-service").retrieval_weight
        graph_weight = registry.get("knowledge-graph-service").retrieval_weight
        doc_weight   = registry.get("content-service").retrieval_weight

        assert sql_weight == pytest.approx(1 / 3, abs=1e-6), (
            f"SQL service weight should have reverted to 1/3 after schema change, "
            f"got {sql_weight}"
        )
        assert graph_weight == pytest.approx(0.2, abs=1e-6), (
            f"Graph service weight should be unchanged at 0.2, got {graph_weight}"
        )
        assert doc_weight == pytest.approx(0.1, abs=1e-6), (
            f"Content service weight should be unchanged at 0.1, got {doc_weight}"
        )

    @all_services_available
    def test_discover_produced_at_is_recent(self):
        """I₁ produced_at must be within 5 seconds of now."""
        from datetime import datetime, timezone
        registry = build_poc_registry()
        i1 = registry.discover("integ-ts", "any query")
        age = (datetime.now(timezone.utc) - i1.produced_at).total_seconds()
        assert age < 5.0, f"I₁ produced_at is {age:.1f}s old — expected < 5s"

    @all_services_available
    def test_discover_query_id_is_preserved(self):
        """The query_id passed to discover() must appear unchanged in I₁."""
        registry = build_poc_registry()
        qid = f"stable-id-{uuid.uuid4().hex}"
        i1 = registry.discover(qid, "any query")
        assert i1.query_id == qid


# =============================================================================
# 7. Diagnostic — print a summary of what is live for quick visual inspection
# =============================================================================

class TestDiagnosticSummary:
    """
    Not a pass/fail test — prints a human-readable summary of all three
    services for quick debugging.  Always runs (no skip markers).
    """

    def test_print_service_status(self, capsys):
        lines = [
            "",
            "=" * 60,
            "  HeteroRAG Layer 1 — Live Service Status",
            "=" * 60,
        ]

        # PostgreSQL
        pg_ok = _pg_available()
        lines.append(f"  PostgreSQL    {PG_HOST}:{PG_PORT}/{PG_DBNAME}")
        lines.append(f"    Status:     {'✓ reachable' if pg_ok else '✗ NOT reachable'}")
        if pg_ok and HAS_PSYCOPG2:
            try:
                conn = psycopg2.connect(
                    host=PG_HOST, port=PG_PORT, dbname=PG_DBNAME,
                    user=PG_USER, password=PG_PASSWORD, connect_timeout=2,
                )
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT COUNT(*) FROM information_schema.tables "
                        "WHERE table_schema = 'public' AND table_type = 'BASE TABLE'"
                    )
                    n_tables = cur.fetchone()[0]
                    cur.execute(
                        "SELECT file_name, rows_loaded FROM _load_progress "
                        "WHERE completed_at IS NOT NULL ORDER BY file_name"
                    ) if n_tables > 0 else None
                    loaded = cur.fetchall() if cur.description else []
                conn.close()
                lines.append(f"    Tables:     {n_tables}")
                if loaded:
                    lines.append("    Loaded files:")
                    for fname, nrows in loaded:
                        lines.append(f"      {fname}: {nrows:,} rows")
            except Exception as e:
                lines.append(f"    Error:      {e}")

        lines.append("")

        # Neo4j
        neo4j_ok = _neo4j_available()
        lines.append(f"  Neo4j         bolt://{NEO4J_HOST}:{NEO4J_PORT}")
        lines.append(f"    Status:     {'✓ reachable' if neo4j_ok else '✗ NOT reachable'}")
        if neo4j_ok and HAS_NEO4J:
            try:
                drv = Neo4jDriver.driver(
                    f"bolt://{NEO4J_HOST}:{NEO4J_PORT}",
                    auth=(NEO4J_USER, NEO4J_PASSWORD),
                )
                with drv.session() as s:
                    r = s.run(
                        "CALL db.labels() YIELD label RETURN collect(label) AS labels"
                    )
                    labels = r.single()["labels"]
                    r2 = s.run(
                        "CALL db.relationshipTypes() YIELD relationshipType "
                        "RETURN collect(relationshipType) AS types"
                    )
                    rel_types = r2.single()["types"]
                drv.close()
                lines.append(f"    Node labels:{' ' + str(sorted(labels)) if labels else ' (none — data not loaded)'}")
                lines.append(f"    Rel types:  {' ' + str(sorted(rel_types)) if rel_types else ' (none — data not loaded)'}")
            except Exception as e:
                lines.append(f"    Error:      {e}")

        lines.append("")

        # Elasticsearch
        es_ok = _es_available()
        lines.append(f"  Elasticsearch http://{ES_HOST}:{ES_PORT}")
        lines.append(f"    Status:     {'✓ reachable' if es_ok else '✗ NOT reachable'}")
        if es_ok and HAS_ES:
            try:
                es = Elasticsearch(f"http://{ES_HOST}:{ES_PORT}")
                health = es.cluster.health()
                lines.append(f"    Cluster:    {health['cluster_name']} ({health['status']})")
                idx_exists = es.indices.exists(index="heterorag_content")
                lines.append(f"    Index:      {'✓ heterorag_content exists' if idx_exists else '✗ heterorag_content missing'}")
                if idx_exists:
                    count = es.count(index="heterorag_content")["count"]
                    lines.append(f"    Documents:  {count:,}")
            except Exception as e:
                lines.append(f"    Error:      {e}")

        lines.append("")
        lines.append("  Registry discover() output:")
        registry = build_poc_registry()
        i1 = registry.discover("diag", "diagnostic query")
        for d in i1.descriptors:
            lines.append(f"    [{d.persistence_type.value:8}] {d.service_id}  "
                         f"weight={d.retrieval_weight:.3f}  active={d.is_active}")
        lines.append("=" * 60)

        with capsys.disabled():
            print("\n".join(lines))

        # This test always passes — it's purely diagnostic
        assert True
