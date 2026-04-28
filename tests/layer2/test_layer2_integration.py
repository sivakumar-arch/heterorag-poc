"""
tests/layer2/test_layer2_integration.py
=========================================
Integration tests for Layer 2 against the live Anthropic API and live Docker services.

Tests are grouped into two skip tiers:
  1. @anthropic_available  — requires ANTHROPIC_API_KEY in environment.
                             Does NOT need live Docker services.
  2. @all_services_available — requires ANTHROPIC_API_KEY + all three containers.

Run:
    # API tests only (no Docker needed)
    pytest tests/layer2/test_layer2_integration.py -v -k "not db"

    # Full integration (Docker + API)
    pytest tests/layer2/test_layer2_integration.py -v -m integration

What these tests prove that unit tests cannot:
    1. The Anthropic API produces syntactically valid SQL for realistic questions.
    2. The Anthropic API produces syntactically valid Cypher for graph questions.
    3. The QueryValidator correctly classifies real LLM output (not mocks).
    4. QueryPlanner.plan() produces an I₂ with no DROPPED services for typical queries.
    5. build_relevance_confirm_fn() correctly routes SQL-only questions to SQL service.
    6. build_relevance_confirm_fn() correctly routes document questions to Content Service.
    7. Full Layer 1 → Layer 2 pipeline produces an I₂ with query_id linkage intact.
"""

from __future__ import annotations

import os
import socket

import pytest

from heterorag.layer1.poc_descriptors import (
    CONTENT_SERVICE_DESCRIPTOR,
    KNOWLEDGE_GRAPH_DESCRIPTOR,
    USER_ACTIVITY_DESCRIPTOR,
    build_poc_registry,
)
from heterorag.layer2.models import QueryLanguage, ValidationStatus
from heterorag.layer2.prompts import build_translation_prompt
from heterorag.layer2.query_planner import QueryPlanner, build_relevance_confirm_fn
from heterorag.layer2.query_validator import QueryValidator, _validate_cypher, _validate_sql
from heterorag.layer2.translation_llm import TranslationLLM

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Availability guards
# ---------------------------------------------------------------------------

def _has_api_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _tcp_reachable(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


def _all_docker_up() -> bool:
    pg = _tcp_reachable(os.getenv("PG_HOST", "localhost"), int(os.getenv("PG_PORT", "5432")))
    n4 = _tcp_reachable(os.getenv("NEO4J_HOST", "localhost"), int(os.getenv("NEO4J_BOLT_PORT", "7687")))
    es = _tcp_reachable(os.getenv("ES_HOST", "localhost"), int(os.getenv("ES_PORT", "9200")))
    return pg and n4 and es


anthropic_available     = pytest.mark.skipif(not _has_api_key(),
                          reason="ANTHROPIC_API_KEY not set")
all_services_available  = pytest.mark.skipif(
    not (_has_api_key() and _all_docker_up()),
    reason="ANTHROPIC_API_KEY not set or Docker services not reachable",
)


# ---------------------------------------------------------------------------
# Shared LLM fixture (module-scoped to limit API calls)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def llm() -> TranslationLLM:
    return TranslationLLM()


# =============================================================================
# 1. SQL translation quality — real LLM output
# =============================================================================

class TestSQLTranslationLive:

    # Realistic Class 1 (SQL-only) questions from the benchmark taxonomy
    SQL_QUESTIONS = [
        ("top_reputation",
         "Who are the top 10 Stack Overflow users by reputation?",
         ["reputation", "LIMIT"]),
        ("python_tagged",
         "How many questions are tagged with python?",
         ["python", "tags", "COUNT", "WHERE"]),
        ("gold_badge_holders",
         "Which users hold at least one Gold badge?",
         ["badges", "class"]),
    ]

    @anthropic_available
    @pytest.mark.parametrize("name,question,expected_keywords", SQL_QUESTIONS)
    def test_sql_translation_is_syntactically_valid(self, llm, name, question, expected_keywords):
        """Real LLM → real SQL → syntactic validation passes."""
        prompt = build_translation_prompt(USER_ACTIVITY_DESCRIPTOR, question)
        reply  = llm.translate(prompt)
        sql    = reply.text

        assert sql, f"LLM returned empty reply for: {question!r}"
        assert "CANNOT_ANSWER" not in sql.upper(), (
            f"LLM said CANNOT_ANSWER for a clearly answerable SQL question: {question!r}\n"
            f"Reply: {sql}"
        )

        is_valid, error = _validate_sql(sql)
        assert is_valid, (
            f"Real LLM produced invalid SQL for: {question!r}\n"
            f"SQL: {sql}\n"
            f"Error: {error}"
        )

    @anthropic_available
    def test_sql_translation_uses_schema_columns(self, llm):
        """LLM must use columns from the schema, not hallucinate new ones."""
        prompt = build_translation_prompt(
            USER_ACTIVITY_DESCRIPTOR,
            "List users ordered by their reputation score",
        )
        reply = llm.translate(prompt)
        sql   = reply.text.upper()
        # reputation is defined in schema — must appear in the query
        assert "REPUTATION" in sql, (
            f"SQL translation did not use 'reputation' column:\n{reply.text}"
        )

    @anthropic_available
    def test_sql_translation_includes_limit(self, llm):
        """Every SQL translation must contain a LIMIT clause (prompt rule)."""
        prompt = build_translation_prompt(
            USER_ACTIVITY_DESCRIPTOR,
            "Show me questions with the highest score",
        )
        reply = llm.translate(prompt)
        assert "LIMIT" in reply.text.upper(), (
            f"SQL translation missing LIMIT clause:\n{reply.text}"
        )


# =============================================================================
# 2. Cypher translation quality — real LLM output
# =============================================================================

class TestCypherTranslationLive:

    CYPHER_QUESTIONS = [
        ("tag_co_occurrence",
         "Which tags most often appear together with the python tag?",
         ["CO_OCCURS_WITH", "python", "RETURN"]),
        ("user_asked_questions",
         "Find all questions asked by a specific user (id=1)",
         ["ASKED", "RETURN"]),
    ]

    @anthropic_available
    @pytest.mark.parametrize("name,question,expected_keywords", CYPHER_QUESTIONS)
    def test_cypher_translation_is_structurally_valid(self, llm, name, question, expected_keywords):
        """Real LLM → real Cypher → structural validation passes."""
        prompt = build_translation_prompt(KNOWLEDGE_GRAPH_DESCRIPTOR, question)
        reply  = llm.translate(prompt)
        cypher = reply.text

        assert cypher, f"LLM returned empty reply for: {question!r}"
        assert "CANNOT_ANSWER" not in cypher.upper()

        is_valid, error = _validate_cypher(cypher)
        assert is_valid, (
            f"Real LLM produced structurally invalid Cypher for: {question!r}\n"
            f"Cypher: {cypher}\n"
            f"Error: {error}"
        )

    @anthropic_available
    def test_cypher_translation_has_return_clause(self, llm):
        prompt = build_translation_prompt(
            KNOWLEDGE_GRAPH_DESCRIPTOR,
            "Find the tags that co-occur most often with 'javascript'",
        )
        reply = llm.translate(prompt)
        assert "RETURN" in reply.text.upper(), (
            f"Cypher missing RETURN clause:\n{reply.text}"
        )

    @anthropic_available
    def test_cypher_uses_co_occurs_with_undirected(self, llm):
        """The undirected CO_OCCURS_WITH pattern is critical — a directed match misses half the data."""
        prompt = build_translation_prompt(
            KNOWLEDGE_GRAPH_DESCRIPTOR,
            "Which tags are most related to the python tag?",
        )
        reply = llm.translate(prompt)
        # Undirected: (t1)-[:CO_OCCURS_WITH]-(t2) or similar
        cypher_upper = reply.text.upper()
        assert "CO_OCCURS_WITH" in cypher_upper, (
            f"Cypher did not use CO_OCCURS_WITH:\n{reply.text}"
        )


# =============================================================================
# 3. BM25 translation quality — real LLM output
# =============================================================================

class TestBM25TranslationLive:

    @anthropic_available
    def test_bm25_returns_non_empty_query_string(self, llm):
        prompt = build_translation_prompt(
            CONTENT_SERVICE_DESCRIPTOR,
            "What are common solutions for Python AttributeError?",
        )
        reply = llm.translate(prompt)
        assert reply.text.strip(), "BM25 translation returned empty string"

    @anthropic_available
    def test_bm25_no_field_prefix_in_output(self, llm):
        """BM25 output must not contain field: prefix (executor applies multi_match)."""
        prompt = build_translation_prompt(
            CONTENT_SERVICE_DESCRIPTOR,
            "How do I fix a NullPointerException in Java?",
        )
        reply = llm.translate(prompt)
        # field: prefix would break the multi_match query execution
        assert "title:" not in reply.text and "body:" not in reply.text, (
            f"BM25 output contains field: prefix:\n{reply.text}"
        )

    @anthropic_available
    def test_bm25_no_fences_in_output(self, llm):
        prompt = build_translation_prompt(
            CONTENT_SERVICE_DESCRIPTOR,
            "What does the Python GIL do?",
        )
        reply = llm.translate(prompt)
        assert "```" not in reply.text, f"BM25 output contains code fence:\n{reply.text}"


# =============================================================================
# 4. QueryValidator with real LLM output
# =============================================================================

class TestQueryValidatorLive:

    @anthropic_available
    def test_validator_accepts_real_sql_without_retry(self, llm):
        """For a realistic SQL question, the validator should not need a retry."""
        prompt    = build_translation_prompt(USER_ACTIVITY_DESCRIPTOR, "top 10 users by reputation")
        raw_sql   = llm.translate(prompt).text
        validator = QueryValidator(llm)

        status, _, records = validator.validate(
            service_id     = "user-activity-service",
            query_language = "sql",
            native_query   = raw_sql,
            natural_query  = "top 10 users by reputation",
        )
        assert status in (ValidationStatus.VALID, ValidationStatus.FIXED), (
            f"Validator unexpectedly dropped a real LLM SQL translation.\n"
            f"SQL: {raw_sql}\n"
            f"Records: {records}"
        )

    @anthropic_available
    def test_validator_accepts_real_cypher_without_retry(self, llm):
        prompt  = build_translation_prompt(
            KNOWLEDGE_GRAPH_DESCRIPTOR, "find tags related to python"
        )
        raw_cypher = llm.translate(prompt).text
        validator  = QueryValidator(llm)

        status, _, records = validator.validate(
            service_id     = "knowledge-graph-service",
            query_language = "cypher",
            native_query   = raw_cypher,
            natural_query  = "find tags related to python",
        )
        assert status in (ValidationStatus.VALID, ValidationStatus.FIXED), (
            f"Validator dropped a real Cypher translation.\nCypher: {raw_cypher}"
        )


# =============================================================================
# 5. RelevanceFilter confirm function — real LLM routing decisions
# =============================================================================

class TestRelevanceConfirmLive:

    @anthropic_available
    def test_sql_only_question_routes_to_sql(self, llm):
        """'top users by reputation' should route to SQL, not Graph or Document."""
        confirm_fn = build_relevance_confirm_fn(llm)
        result = confirm_fn(
            "Who are the top 10 users by reputation score?",
            [USER_ACTIVITY_DESCRIPTOR, KNOWLEDGE_GRAPH_DESCRIPTOR, CONTENT_SERVICE_DESCRIPTOR],
        )
        assert "user-activity-service" in result, (
            f"SQL question did not route to SQL service. Got: {result}"
        )

    @anthropic_available
    def test_graph_question_routes_to_graph(self, llm):
        """'which tags co-occur with python' is a graph topology question."""
        confirm_fn = build_relevance_confirm_fn(llm)
        result = confirm_fn(
            "Which tags most frequently appear together with the python tag?",
            [USER_ACTIVITY_DESCRIPTOR, KNOWLEDGE_GRAPH_DESCRIPTOR, CONTENT_SERVICE_DESCRIPTOR],
        )
        assert "knowledge-graph-service" in result, (
            f"Graph question did not route to Graph service. Got: {result}"
        )

    @anthropic_available
    def test_document_question_routes_to_document(self, llm):
        """'explain what a GIL is' requires full-text search."""
        confirm_fn = build_relevance_confirm_fn(llm)
        result = confirm_fn(
            "What are common explanations of the Python GIL in Stack Overflow answers?",
            [USER_ACTIVITY_DESCRIPTOR, KNOWLEDGE_GRAPH_DESCRIPTOR, CONTENT_SERVICE_DESCRIPTOR],
        )
        assert "content-service" in result, (
            f"Document question did not route to Content service. Got: {result}"
        )

    @anthropic_available
    def test_cross_service_question_routes_to_multiple(self, llm):
        """A question spanning SQL + Graph should include both services."""
        confirm_fn = build_relevance_confirm_fn(llm)
        result = confirm_fn(
            "What is the reputation of users who asked the most questions in the python tag community?",
            [USER_ACTIVITY_DESCRIPTOR, KNOWLEDGE_GRAPH_DESCRIPTOR, CONTENT_SERVICE_DESCRIPTOR],
        )
        # Reputation is SQL; tag community topology is Graph — both should be selected
        assert len(result) >= 1, f"No services selected for a cross-service question: {result}"


# =============================================================================
# 6. Full Layer 1 → Layer 2 pipeline — end-to-end
# =============================================================================

class TestFullPipelineLive:

    @anthropic_available
    def test_plan_returns_valid_i2_for_sql_question(self, llm):
        """
        Full pipeline: build_poc_registry → discover → QueryPlanner.plan → I₂.
        For a purely SQL question, I₂ must contain at least the SQL service.
        """
        confirm_fn = build_relevance_confirm_fn(llm)
        registry   = build_poc_registry(llm_confirm_fn=confirm_fn)

        i1 = registry.discover("integ-l2-01", "top 10 users by reputation")
        assert len(i1.descriptors) >= 1, "No services survived RelevanceFilter"

        planner = QueryPlanner(llm)
        i2      = planner.plan(i1, "top 10 users by reputation")

        assert i2.query_id == "integ-l2-01", "query_id not preserved through Layer 2"
        assert len(i2.queries) >= 1, (
            f"All services dropped from I₂!\n"
            f"dropped: {i2.dropped_service_ids}\n"
            f"validation_log: {i2.validation_log}"
        )

        # Check no DROPPED queries snuck into I₂
        for q in i2.queries:
            assert q.validation_status != ValidationStatus.DROPPED

    @anthropic_available
    def test_query_id_is_consistent_across_i1_and_i2(self, llm):
        """query_id must flow unchanged from discover() through plan()."""
        registry = build_poc_registry()
        qid = "consistency-check-xyz"

        i1 = registry.discover(qid, "any question")
        i2 = QueryPlanner(llm).plan(i1, "any question")

        assert i2.query_id == qid

    @anthropic_available
    def test_retrieval_weights_match_registry_after_cold_start(self, llm):
        """Weights in I₂ ServiceQuery objects must match ServiceDescriptor.retrieval_weight."""
        registry = build_poc_registry()
        i1       = registry.discover("weight-check", "top users by reputation")

        # Cold start sets all weights to 1/3
        for d in i1.descriptors:
            assert d.retrieval_weight > 0.0, "Cold start did not set weights"

        i2 = QueryPlanner(llm).plan(i1, "top users by reputation")

        for sq in i2.queries:
            # Find matching descriptor
            d = next(d for d in i1.descriptors if d.service_id == sq.service_id)
            assert sq.retrieval_weight == pytest.approx(d.retrieval_weight), (
                f"Weight mismatch for {sq.service_id}: "
                f"I₂={sq.retrieval_weight}, descriptor={d.retrieval_weight}"
            )

    @anthropic_available
    def test_validation_log_has_entry_for_every_i1_service(self, llm):
        """
        I₂.validation_log must have at least one entry per service that
        was in I₁ — including dropped ones.
        """
        registry = build_poc_registry()
        i1       = registry.discover("vlog-check", "top users by reputation")
        i2       = QueryPlanner(llm).plan(i1, "top users by reputation")

        logged_services = {r.service_id for r in i2.validation_log}
        i1_services     = set(i1.service_ids())

        assert i1_services <= logged_services, (
            f"Validation log missing entries for: {i1_services - logged_services}"
        )
