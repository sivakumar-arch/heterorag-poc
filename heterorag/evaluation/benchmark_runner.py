"""
heterorag/evaluation/benchmark_runner.py
==========================================
Step 12 — Benchmark Runner.

Runs all 120 benchmark questions against HeteroRAG Full and all four baselines,
collecting GenerationResults for each (query, system) pair.

Architecture:
  - Questions are defined inline as the 120-question taxonomy from Foundation Doc §7.3.
  - Ground truth is loaded from the live PostgreSQL database (SQL views created by
    Flyway V1 migration) and from Neo4j fixture nodes (created by Liquibase).
  - Questions are run question-by-question (outer loop), system-by-system (inner).
    This gives each system equal database conditions per question.
  - Results are written incrementally to results/raw_results.jsonl so a crash
    mid-run can be resumed by skipping already-completed (query_id, system) pairs.
  - The runner is intentionally synchronous to avoid confounding latency measurements
    with Python concurrency overhead.

Ground truth format per question:
  {
    "question_id":      "c1_q01",
    "query_class":      1,
    "required_services": ["sql"],
    "natural_query":    "Who are the top 10 Stack Overflow users by reputation?",
    "gt_sql_view":      "gt_c1_q01",       # Flyway view name (SQL questions)
    "gt_cypher":        null,              # Cypher template (Graph questions)
    "gt_doc_fixture":   null,             # document fixture file (Doc questions)
    "af_metric":        "f1",             # f1 | ndcg@10 | recall@10
  }
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from heterorag.evaluation.baselines import (
    B1_SQLOnlyRouter,
    B2_DocumentOnlyRAG,
    B3_LLMFunctionCallingRouter,
    B4_FixedPlanAblation,
    BaselineSystem,
)
from heterorag.layer1.poc_descriptors import build_poc_registry
from heterorag.layer2.translation_llm import TranslationLLM
from heterorag.layer3 import ConnectionRegistry
from heterorag.layer1.poc_descriptors import (
    USER_ACTIVITY_DESCRIPTOR,
    KNOWLEDGE_GRAPH_DESCRIPTOR,
    CONTENT_SERVICE_DESCRIPTOR,
)
from heterorag.layer4.generation import GenerationLLM, GenerationResult
from heterorag.layer4.pipeline import HeteroRAGPipeline

log = logging.getLogger(__name__)


# =============================================================================
# Benchmark question taxonomy — 120 questions
# =============================================================================

@dataclass
class BenchmarkQuestion:
    question_id:      str
    query_class:      int         # 1–5 (4a→41, 4b→42, 4c→43 for integer storage)
    query_class_label: str        # "1","2","3","4a","4b","4c","5"
    required_services: list[str]  # subset of ["sql","graph","document"]
    natural_query:    str
    gt_sql_view:      str | None  # name of Flyway view for SQL component
    gt_cypher:        str | None  # Cypher string for Graph component
    gt_doc_fixture:   str | None  # fixture filename for Document component
    af_metric:        str         # "f1" | "ndcg10" | "recall10"
    difficulty:       str = "medium"  # simple|medium|medium-hard|hard


def load_benchmark_questions() -> list[BenchmarkQuestion]:
    """
    Returns the full 120-question benchmark taxonomy.
    Questions span 7 classes × 4 difficulty levels as defined in Foundation Doc §7.3.
    """
    questions: list[BenchmarkQuestion] = []

    # ------------------------------------------------------------------
    # Class 1 — SQL-only (20 questions)
    # ------------------------------------------------------------------
    class1_specs = [
        ("c1_q01", "Who are the top 10 Stack Overflow users by reputation?",
         "gt_c1_q01", "f1", "simple"),
        ("c1_q02", "Which users have more than 1000 reputation and joined before 2015?",
         "gt_c1_q02", "f1", "simple"),
        ("c1_q03", "What are the 20 highest-scored questions on Stack Overflow?",
         "gt_c1_q03", "f1", "simple"),
        ("c1_q04", "What are the top 20 tags by question count?",
         "gt_c1_q04", "f1", "simple"),
        ("c1_q05", "Which questions tagged with python have the highest scores?",
         "gt_c1_q05", "f1", "medium"),
        ("c1_q06", "Which users have cast more than 500 up-votes?",
         "gt_c1_q06", "f1", "simple"),
        ("c1_q07", "Which questions have an accepted answer and a score greater than 50?",
         "gt_c1_q07", "f1", "medium"),
        ("c1_q08", "Which 20 users have earned the most badges overall?",
         "gt_c1_q08", "f1", "medium"),
        ("c1_q09", "Who are the gold badge holders on Stack Overflow?",
         "gt_c1_q09", "f1", "medium"),
        ("c1_q10", "Which questions have more than 10 answers?",
         "gt_c1_q10", "f1", "simple"),
        ("c1_q11", "How many questions were posted each month in 2022?",
         "gt_c1_q11", "f1", "medium"),
        ("c1_q12", "Which users have both more than 200 up-votes and more than 50 down-votes?",
         "gt_c1_q12", "f1", "medium-hard"),
        ("c1_q13", "Which 20 posts have the most comments?",
         "gt_c1_q13", "f1", "simple"),
        ("c1_q14", "Which questions have more than 10,000 views?",
         "gt_c1_q14", "f1", "simple"),
        ("c1_q15", "Which users are located in London?",
         "gt_c1_q15", "f1", "simple"),
        ("c1_q16", "Which are the top 20 tag-based badges by number of holders?",
         "gt_c1_q16", "f1", "medium"),
        ("c1_q17", "What are the scores of all answers to question 11227809?",
         "gt_c1_q17", "f1", "medium"),
        ("c1_q18", "Which users have zero reputation?",
         "gt_c1_q18", "f1", "simple"),
        ("c1_q19", "Which questions were closed as duplicates?",
         "gt_c1_q19", "f1", "medium"),
        ("c1_q20", "What is the average score per tag for the 15 highest-quality tags?",
         "gt_c1_q20", "f1", "hard"),
    ]
    for qid, query, gt_view, metric, diff in class1_specs:
        questions.append(BenchmarkQuestion(
            question_id=qid, query_class=1, query_class_label="1",
            required_services=["sql"], natural_query=query,
            gt_sql_view=gt_view, gt_cypher=None, gt_doc_fixture=None,
            af_metric=metric, difficulty=diff,
        ))

    # ------------------------------------------------------------------
    # Class 2 — Graph-only (20 questions)
    # ------------------------------------------------------------------
    class2_specs = [
        ("c2_q01", "Which questions are linked to question 11227809 within 2 hops?",
         "MATCH (q:Question {id:11227809})-[:LINKED_TO*1..2]->(l:Question) RETURN DISTINCT l.id AS id ORDER BY id",
         "f1", "medium"),
        ("c2_q02", "Which tags most often appear together with the python tag?",
         "MATCH (t:Tag {name:'python'})-[r:CO_OCCURS_WITH]-(o:Tag) RETURN o.name AS tag, r.weight AS weight ORDER BY weight DESC LIMIT 20",
         "f1", "simple"),
        ("c2_q03", "Which users have answered questions tagged with javascript?",
         "MATCH (u:User)-[:ANSWERED]->(:Answer)-[:ANSWERS]->(:Question)-[:TAGGED_WITH]->(t:Tag {name:'javascript'}) RETURN DISTINCT u.id AS user_id ORDER BY user_id LIMIT 50",
         "f1", "medium"),
        ("c2_q04", "Which questions are marked as duplicates?",
         "MATCH (q:Question)-[:DUPLICATE_OF]->(o:Question) RETURN q.id AS duplicate_id, o.id AS original_id ORDER BY duplicate_id LIMIT 20",
         "f1", "simple"),
        ("c2_q05", "Which tags co-occur with both python and pandas?",
         "MATCH (t:Tag)-[:CO_OCCURS_WITH]-(py:Tag {name:'python'}), (t)-[:CO_OCCURS_WITH]-(pd:Tag {name:'pandas'}) RETURN t.name AS tag ORDER BY tag LIMIT 20",
         "f1", "medium-hard"),
        ("c2_q06", "Which users both asked and answered questions in the python tag?",
         "MATCH (u:User)-[:ASKED]->(q:Question)-[:TAGGED_WITH]->(t:Tag {name:'python'}), (u)-[:ANSWERED]->(a:Answer)-[:ANSWERS]->(q2:Question)-[:TAGGED_WITH]->(t) RETURN DISTINCT u.id AS user_id ORDER BY user_id LIMIT 30",
         "f1", "hard"),
        ("c2_q07", "Which questions have accepted answers where the answerer is not the asker?",
         "MATCH (asker:User)-[:ASKED]->(q:Question)-[:ACCEPTED]->(a:Answer)<-[:ANSWERED]-(answerer:User) WHERE asker.id <> answerer.id RETURN q.id AS question_id LIMIT 50",
         "f1", "medium"),
        ("c2_q08", "What are the top 15 tags that co-occur with the sql tag?",
         "MATCH (sql:Tag {name:'sql'})-[r:CO_OCCURS_WITH]-(n:Tag) RETURN n.name AS tag, r.weight AS weight ORDER BY weight DESC LIMIT 15",
         "f1", "simple"),
        ("c2_q09", "Which questions belong to the longest duplicate chains?",
         "MATCH (q:Question)-[:DUPLICATE_OF*1..3]->(root:Question) WHERE NOT EXISTS {MATCH (root)-[:DUPLICATE_OF]->()} RETURN root.id AS root_id, COUNT(q) AS dup_count ORDER BY dup_count DESC LIMIT 20",
         "f1", "hard"),
        ("c2_q10", "What are the top 20 questions by number of linked questions?",
         "MATCH (q:Question)<-[:LINKED_TO]-(linked:Question) RETURN q.id AS question_id, COUNT(linked) AS link_count ORDER BY link_count DESC LIMIT 20",
         "f1", "medium"),
        ("c2_q11", "Which users are most central in the question-answering network? (top 10 by degree)",
         "MATCH (u:User)-[:ASKED|ANSWERED]->(n) RETURN u.id AS user_id, COUNT(n) AS degree ORDER BY degree DESC LIMIT 10",
         "ndcg10", "hard"),
        ("c2_q12", "Which tag communities form around the most connected tags? (top 10 tags by co-occurrence degree)",
         "MATCH (t:Tag)-[:CO_OCCURS_WITH]-(other:Tag) RETURN t.name AS tag, COUNT(other) AS co_count ORDER BY co_count DESC LIMIT 10",
         "ndcg10", "hard"),
        ("c2_q13", "What is the 2-hop neighbourhood of the python tag in the co-occurrence graph?",
         "MATCH (t:Tag {name:'python'})-[:CO_OCCURS_WITH*1..2]-(n:Tag) RETURN DISTINCT n.name AS tag ORDER BY tag LIMIT 30",
         "f1", "medium-hard"),
        ("c2_q14", "Which questions are linked to the highest number of other questions?",
         "MATCH (q:Question)-[:LINKED_TO]->(other:Question) RETURN q.id AS question_id, COUNT(other) AS links ORDER BY links DESC LIMIT 10",
         "f1", "medium"),
        ("c2_q15", "Which users have answered questions in the most distinct tags?",
         "MATCH (u:User)-[:ANSWERED]->(:Answer)-[:ANSWERS]->(:Question)-[:TAGGED_WITH]->(t:Tag) RETURN u.id AS user_id, COUNT(DISTINCT t) AS tag_count ORDER BY tag_count DESC LIMIT 10",
         "f1", "hard"),
        ("c2_q16", "What are the tags in the neighbourhood of javascript within 1 hop?",
         "MATCH (js:Tag {name:'javascript'})-[r:CO_OCCURS_WITH]-(n:Tag) RETURN n.name AS tag, r.weight AS weight ORDER BY weight DESC LIMIT 20",
         "f1", "simple"),
        ("c2_q17", "How many questions have been asked by users who have also given accepted answers?",
         "MATCH (u:User)-[:ASKED]->(q:Question)-[:ACCEPTED]->(a:Answer)<-[:ANSWERED]-(u) RETURN COUNT(DISTINCT q) AS count",
         "f1", "hard"),
        ("c2_q18", "Which users have asked more than 5 questions?",
         "MATCH (u:User)-[:ASKED]->(q:Question) WITH u, COUNT(q) AS qcount WHERE qcount > 5 RETURN u.id AS user_id, qcount ORDER BY qcount DESC LIMIT 20",
         "f1", "medium"),
        ("c2_q19", "Which questions are tagged with both python and machine-learning?",
         "MATCH (q:Question)-[:TAGGED_WITH]->(t1:Tag {name:'python'}), (q)-[:TAGGED_WITH]->(t2:Tag {name:'machine-learning'}) RETURN q.id AS question_id ORDER BY question_id LIMIT 20",
         "f1", "medium"),
        ("c2_q20", "Which tags co-occur with sql and also with database?",
         "MATCH (t:Tag)-[:CO_OCCURS_WITH]-(sql:Tag {name:'sql'}), (t)-[:CO_OCCURS_WITH]-(db:Tag {name:'database'}) RETURN t.name AS tag ORDER BY tag LIMIT 15",
         "f1", "medium-hard"),
    ]
    for qid, query, cypher, metric, diff in class2_specs:
        questions.append(BenchmarkQuestion(
            question_id=qid, query_class=2, query_class_label="2",
            required_services=["graph"], natural_query=query,
            gt_sql_view=None, gt_cypher=cypher, gt_doc_fixture=None,
            af_metric=metric, difficulty=diff,
        ))

    # ------------------------------------------------------------------
    # Class 3 — Document-only (20 questions)
    # ------------------------------------------------------------------
    class3_specs = [
        ("c3_q01", "What are common explanations of the Python GIL on Stack Overflow?", "c3_q01.json", "recall10", "simple"),
        ("c3_q02", "Which posts discuss AttributeError in Python?", "c3_q02.json", "recall10", "simple"),
        ("c3_q03", "What are the most common solutions for NullPointerException in Java?", "c3_q03.json", "recall10", "simple"),
        ("c3_q04", "How do Stack Overflow answers explain list comprehension in Python?", "c3_q04.json", "recall10", "medium"),
        ("c3_q05", "Which posts explain the difference between == and is in Python?", "c3_q05.json", "recall10", "medium"),
        ("c3_q06", "What does the tag wiki for javascript say?", "c3_q06.json", "recall10", "simple"),
        ("c3_q07", "Which posts discuss memory leaks in C++?", "c3_q07.json", "recall10", "medium"),
        ("c3_q08", "How do answers on Stack Overflow explain async/await in JavaScript?", "c3_q08.json", "recall10", "medium"),
        ("c3_q09", "Which posts explain what a decorator does in Python?", "c3_q09.json", "recall10", "medium"),
        ("c3_q10", "What are the most helpful explanations of SQL JOIN types?", "c3_q10.json", "recall10", "medium"),
        ("c3_q11", "Which posts discuss segmentation faults in C?", "c3_q11.json", "recall10", "medium"),
        ("c3_q12", "How do Stack Overflow posts explain REST vs GraphQL?", "c3_q12.json", "recall10", "medium-hard"),
        ("c3_q13", "What is the tag wiki for python?", "c3_q13.json", "recall10", "simple"),
        ("c3_q14", "Which posts explain how to fix IndentationError in Python?", "c3_q14.json", "recall10", "simple"),
        ("c3_q15", "What are common explanations of Docker networking?", "c3_q15.json", "recall10", "medium"),
        ("c3_q16", "How do answers explain the difference between process and thread?", "c3_q16.json", "recall10", "medium-hard"),
        ("c3_q17", "Which posts discuss pandas DataFrame indexing errors?", "c3_q17.json", "recall10", "medium"),
        ("c3_q18", "How do users describe themselves in their profiles if they work in machine learning?", "c3_q18.json", "recall10", "hard"),
        ("c3_q19", "What do Stack Overflow answers say about optimizing database queries?", "c3_q19.json", "recall10", "medium-hard"),
        ("c3_q20", "Which posts explain what a closure is in JavaScript?", "c3_q20.json", "recall10", "medium"),
    ]
    for qid, query, fixture, metric, diff in class3_specs:
        questions.append(BenchmarkQuestion(
            question_id=qid, query_class=3, query_class_label="3",
            required_services=["document"], natural_query=query,
            gt_sql_view=None, gt_cypher=None, gt_doc_fixture=fixture,
            af_metric=metric, difficulty=diff,
        ))

    # ------------------------------------------------------------------
    # Class 4a — SQL + Graph (15 questions)
    # ------------------------------------------------------------------
    class4a_specs = [
        ("c4a_q01", "What are the reputations of users who asked the most Python questions, and which tags do they use together?", "medium-hard"),
        ("c4a_q02", "Which high-reputation users (reputation > 5000) are also central in the answering network?", "hard"),
        ("c4a_q03", "What is the score distribution of questions tagged with javascript, and which tags co-occur with javascript most often?", "hard"),
        ("c4a_q04", "Which users have earned gold badges and have also asked questions linked to duplicate chains?", "hard"),
        ("c4a_q05", "How does the number of questions per tag compare to the co-occurrence weight of that tag with python?", "hard"),
        ("c4a_q06", "Which questions with score > 100 are also connected to other questions via LINKED_TO?", "medium-hard"),
        ("c4a_q07", "What are the top tags by question count that also form strong co-occurrence communities?", "hard"),
        ("c4a_q08", "Which users with more than 1000 reputation have answered questions in the python tag graph?", "medium-hard"),
        ("c4a_q09", "How many down-votes have users received who are connected in the answering network?", "hard"),
        ("c4a_q10", "What are the acceptance rates of questions grouped by their most common co-occurring tags?", "hard"),
        ("c4a_q11", "Which users who asked questions in 2022 are also part of the python tag neighbourhood?", "medium-hard"),
        ("c4a_q12", "What is the view count of questions that are marked as duplicates in the graph?", "medium"),
        ("c4a_q13", "Which tag communities (from graph) correspond to the highest-scoring questions (from SQL)?", "hard"),
        ("c4a_q14", "Who are the top 10 answerers of javascript questions by badge count?", "medium-hard"),
        ("c4a_q15", "Which users have the highest ratio of accepted answers to total answers, and which tags do they specialise in?", "hard"),
    ]
    for qid, query, diff in class4a_specs:
        questions.append(BenchmarkQuestion(
            question_id=qid, query_class=41, query_class_label="4a",
            required_services=["sql", "graph"], natural_query=query,
            gt_sql_view=f"gt_{qid}_sql", gt_cypher=None, gt_doc_fixture=None,
            af_metric="f1", difficulty=diff,
        ))

    # ------------------------------------------------------------------
    # Class 4b — SQL + Document (15 questions)
    # ------------------------------------------------------------------
    class4b_specs = [
        ("c4b_q01", "What are the scores of the top Python questions and what do their bodies explain?", "medium"),
        ("c4b_q02", "Which high-reputation users have also written Stack Overflow answers explaining decorators?", "hard"),
        ("c4b_q03", "What is the view count of questions about SQL JOINs and what do their accepted answers say?", "medium-hard"),
        ("c4b_q04", "How many questions tagged java exist and what are the most common Java error explanations?", "medium"),
        ("c4b_q05", "What are the scores of questions about memory management and what do the top answers explain?", "medium-hard"),
        ("c4b_q06", "Which users with more than 5000 reputation have described themselves as working in data science?", "hard"),
        ("c4b_q07", "What are the most viewed questions about Python exceptions and what solutions do answers propose?", "medium-hard"),
        ("c4b_q08", "How does question score correlate with the depth of explanation in the accepted answer?", "hard"),
        ("c4b_q09", "Which closed questions about JavaScript have the most helpful answer bodies?", "hard"),
        ("c4b_q10", "What are the comment counts on questions that have detailed explanations about async programming?", "medium-hard"),
        ("c4b_q11", "Which tag has the highest average question score, and what does its tag wiki say?", "hard"),
        ("c4b_q12", "What are the reputations of users who wrote the most-voted answers about recursion?", "hard"),
        ("c4b_q13", "Which questions posted in 2023 about machine learning have comprehensive answer bodies?", "hard"),
        ("c4b_q14", "What is the acceptance rate of Python questions and what do accepted answers about list comprehension say?", "hard"),
        ("c4b_q15", "Which questions with more than 50 answers have the most informative bodies about algorithm complexity?", "hard"),
    ]
    for qid, query, diff in class4b_specs:
        questions.append(BenchmarkQuestion(
            question_id=qid, query_class=42, query_class_label="4b",
            required_services=["sql", "document"], natural_query=query,
            gt_sql_view=f"gt_{qid}_sql", gt_cypher=None, gt_doc_fixture=f"{qid}.json",
            af_metric="f1", difficulty=diff,
        ))

    # ------------------------------------------------------------------
    # Class 4c — Graph + Document (15 questions)
    # ------------------------------------------------------------------
    class4c_specs = [
        ("c4c_q01", "Which tags co-occur with python most often and what do the tag wikis for those tags say?", "medium"),
        ("c4c_q02", "Which users are central in the javascript answering network and how do they describe themselves?", "hard"),
        ("c4c_q03", "Which questions are linked to the most other questions and what are their body topics?", "medium-hard"),
        ("c4c_q04", "What tags form a community around machine-learning and what do their wikis explain?", "hard"),
        ("c4c_q05", "Which questions are duplicates of each other and what common problem do their bodies describe?", "hard"),
        ("c4c_q06", "Which users answered questions in the python tag graph and how do they describe their expertise?", "hard"),
        ("c4c_q07", "What is the 2-hop co-occurrence neighbourhood of the sql tag and what do those tag wikis say?", "hard"),
        ("c4c_q08", "Which questions are tagged with both python and data-science and what topics do their bodies cover?", "medium-hard"),
        ("c4c_q09", "Which tags have the highest co-occurrence weight with javascript and what are their wikis about?", "medium"),
        ("c4c_q10", "Which users both asked questions about concurrency in the graph and wrote detailed answers about threads?", "hard"),
        ("c4c_q11", "What are the most common tag co-occurrence pairs and what do the question bodies in those pairs discuss?", "hard"),
        ("c4c_q12", "Which duplicate question chains relate to Python AttributeError based on body content?", "hard"),
        ("c4c_q13", "Which users are connected in the java tag answering network and what Java topics do they discuss in answers?", "hard"),
        ("c4c_q14", "What tags form communities around web development and what do their top answer bodies explain?", "hard"),
        ("c4c_q15", "Which questions linked to question 11227809 also have bodies discussing similar Python concepts?", "hard"),
    ]
    for qid, query, diff in class4c_specs:
        questions.append(BenchmarkQuestion(
            question_id=qid, query_class=43, query_class_label="4c",
            required_services=["graph", "document"], natural_query=query,
            gt_sql_view=None, gt_cypher=None, gt_doc_fixture=f"{qid}.json",
            af_metric="f1", difficulty=diff,
        ))

    # ------------------------------------------------------------------
    # Class 5 — All three services (15 questions)
    # ------------------------------------------------------------------
    class5_specs = [
        ("c5_q01", "Who are the most active Python users by reputation, which tags do they cluster with in the graph, and what do their top answers explain?", "hard"),
        ("c5_q02", "What are the highest-scored questions about JavaScript, which tags co-occur with javascript in the graph, and what do the top answer bodies say?", "hard"),
        ("c5_q03", "Which gold badge holders are central in the answering network and how do they describe their expertise in their profiles?", "hard"),
        ("c5_q04", "What is the reputation distribution of users who asked questions in 2022, how are they connected in the graph, and what topics do their questions cover?", "hard"),
        ("c5_q05", "Which questions tagged with machine-learning have the most views, how are they linked in the graph, and what approaches do their accepted answers recommend?", "hard"),
        ("c5_q06", "Who are the top answerers by reputation in the python tag community, how central are they in the network, and what Python problems do they most often address in answers?", "hard"),
        ("c5_q07", "Which questions about SQL have the highest scores, how are they connected via PostLinks, and what SQL concepts do the accepted answers explain?", "hard"),
        ("c5_q08", "What are the most common tags used together with java, how do those tags score on average in SQL, and what Java topics do their wikis cover?", "hard"),
        ("c5_q09", "Which users have both high reputation and answered questions in multiple tag communities, and how do they describe their expertise?", "hard"),
        ("c5_q10", "What are the top-viewed questions about concurrency, how are they linked in the graph, and what solutions do accepted answers provide?", "hard"),
        ("c5_q11", "Which duplicate question clusters relate to Python errors, what are the score statistics for those questions, and what errors do their bodies describe?", "hard"),
        ("c5_q12", "Who are the highest-reputation users who asked questions about databases, how central are they in the database tag graph, and what database problems do they discuss?", "hard"),
        ("c5_q13", "What are the most comment-heavy questions about web frameworks, how are they tagged in the graph, and what framework comparisons do the top answers make?", "hard"),
        ("c5_q14", "Which users earned silver badges in the python tag, how are they connected in the python answering network, and what Python topics do they explain in their best answers?", "hard"),
        ("c5_q15", "What is the full picture of activity around the javascript tag: top scorers from SQL, tag neighbours from the graph, and explanatory content from answers?", "hard"),
    ]
    for qid, query, diff in class5_specs:
        questions.append(BenchmarkQuestion(
            question_id=qid, query_class=5, query_class_label="5",
            required_services=["sql", "graph", "document"], natural_query=query,
            gt_sql_view=f"gt_{qid}_sql", gt_cypher=None, gt_doc_fixture=f"{qid}.json",
            af_metric="f1", difficulty=diff,
        ))

    assert len(questions) == 120, f"Expected 120 questions, got {len(questions)}"
    return questions


# =============================================================================
# Ground truth loaders
# =============================================================================

def load_sql_ground_truth(view_name: str, pg_conn) -> list[dict]:
    """Execute a named Flyway GT view and return its rows as dicts."""
    try:
        import psycopg2.extras
        with pg_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(f"SELECT * FROM {view_name}")
            return [dict(r) for r in cur.fetchall()]
    except Exception as exc:
        log.error("GT load failed for view '%s': %s", view_name, exc)
        return []


def load_graph_ground_truth(cypher: str, neo4j_driver) -> list[dict]:
    """Execute a Cypher query against the live GT fixtures and return records."""
    try:
        with neo4j_driver.session() as session:
            result = session.run(cypher)
            return [dict(r) for r in result.fetch(100)]
    except Exception as exc:
        log.error("GT Cypher failed: %s", exc)
        return []


def load_doc_ground_truth(fixture_file: str, fixture_dir: Path) -> list[str]:
    """Load expected document IDs from a pre-generated fixture file."""
    path = fixture_dir / fixture_file
    if not path.exists():
        log.warning("Doc GT fixture not found: %s", path)
        return []
    with path.open() as f:
        data = json.load(f)
    return data.get("expected_ids", [])


# =============================================================================
# Raw run record
# =============================================================================

@dataclass
class RawRunRecord:
    """One (question, system) execution result — written to JSONL."""
    question_id:       str
    query_class_label: str
    required_services: list[str]
    system_name:       str
    queried_service_ids: list[str]
    answer:            str
    retrieval_ms:      float
    generation_ms:     float
    conflict_count:    int
    is_cannot_answer:  bool
    prompt_tokens:     int
    reply_tokens:      int
    model:             str
    timestamp:         str = field(default_factory=lambda: datetime.now_utc())

    @staticmethod
    def datetime_now_utc() -> str:
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()


# =============================================================================
# BenchmarkRunner
# =============================================================================

class BenchmarkRunner:
    """
    Step 12: executes all 120 questions × 5 systems and writes raw results.

    Usage:
        runner = BenchmarkRunner.from_env(output_dir=Path("results"))
        runner.run()
    """

    SYSTEMS_ORDER = [
        "HeteroRAG_Full",
        "B1_SQL_Only",
        "B2_Document_Only",
        "B3_LLM_FunctionCalling",
        "B4_Fixed_Plan",
    ]

    def __init__(
        self,
        systems:     dict[str, BaselineSystem],
        questions:   list[BenchmarkQuestion],
        output_dir:  Path,
        pg_conn=None,
        neo4j_driver=None,
        fixture_dir: Path | None = None,
        resume:      bool = True,
    ):
        self._systems      = systems
        self._questions    = questions
        self._output_dir   = output_dir
        self._pg_conn      = pg_conn
        self._neo4j_driver = neo4j_driver
        self._fixture_dir  = fixture_dir or output_dir / "fixtures"
        self._resume       = resume
        self._raw_path     = output_dir / "raw_results.jsonl"
        output_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(
        cls,
        output_dir:  Path,
        api_key:     str | None = None,
        resume:      bool = True,
        mock_llm:    bool = False,
    ) -> "BenchmarkRunner":
        """
        Factory: build all systems from environment variables.
        Set mock_llm=True for testing without real API calls.
        """
        trans_llm = TranslationLLM(
            api_key=api_key,
            inject_reply="SELECT id FROM users LIMIT 10" if mock_llm else None,
        )
        gen_llm = GenerationLLM(
            api_key=api_key,
            inject_answer="[MOCK ANSWER]" if mock_llm else None,
        )

        # Connection registry
        conn_reg = ConnectionRegistry()
        conn_reg.register("user-activity-service",  USER_ACTIVITY_DESCRIPTOR.connection)
        conn_reg.register("knowledge-graph-service", KNOWLEDGE_GRAPH_DESCRIPTOR.connection)
        conn_reg.register("content-service",        CONTENT_SERVICE_DESCRIPTOR.connection)

        # HeteroRAG Full
        registry = build_poc_registry(heartbeat_timeout_seconds=86400)  # 24h — never expires during benchmark
        registry.discover("warmup", "warmup")   # cold start
        heterorag_pipeline = HeteroRAGPipeline(trans_llm, gen_llm, conn_reg)

        class HeteroRAGSystem(BaselineSystem):
            system_name = "HeteroRAG_Full"
            def __init__(self, pipeline, reg):
                self._pipeline = pipeline
                self._registry = reg
            def run(self, query_id, natural_query):
                i1 = self._registry.discover(query_id, natural_query)
                return self._pipeline.run(i1, natural_query)

        systems = {
            "HeteroRAG_Full":          HeteroRAGSystem(heterorag_pipeline, registry),
            "B1_SQL_Only":             B1_SQLOnlyRouter(trans_llm, gen_llm, conn_reg),
            "B2_Document_Only":        B2_DocumentOnlyRAG(trans_llm, gen_llm, conn_reg),
            "B3_LLM_FunctionCalling":  B3_LLMFunctionCallingRouter(trans_llm, gen_llm, conn_reg),
            "B4_Fixed_Plan":           B4_FixedPlanAblation(trans_llm, gen_llm, conn_reg),
        }

        # Optional ground truth DB connections
        pg_conn = neo4j_driver = None
        try:
            import psycopg2
            pg_conn = psycopg2.connect(
                host=os.getenv("PG_HOST", "localhost"),
                port=int(os.getenv("PG_PORT", "5432")),
                dbname=os.getenv("PG_DBNAME", "heterorag"),
                user=os.getenv("PG_USER", "heterorag"),
                password=os.getenv("PG_PASSWORD", "heterorag_secret"),
                connect_timeout=5,
            )
        except Exception as e:
            log.warning("Cannot connect to PostgreSQL for GT loading: %s", e)
        try:
            from neo4j import GraphDatabase
            neo4j_driver = GraphDatabase.driver(
                f"bolt://{os.getenv('NEO4J_HOST','localhost')}:{os.getenv('NEO4J_BOLT_PORT','7687')}",
                auth=(os.getenv("NEO4J_USER","neo4j"), os.getenv("NEO4J_PASSWORD","heterorag_secret")),
            )
        except Exception as e:
            log.warning("Cannot connect to Neo4j for GT loading: %s", e)

        return cls(
            systems=systems,
            questions=load_benchmark_questions(),
            output_dir=output_dir,
            pg_conn=pg_conn,
            neo4j_driver=neo4j_driver,
            resume=resume,
        )

    def run(self) -> Path:
        """Execute the full benchmark. Returns path to raw_results.jsonl."""
        completed = self._load_completed()
        log.info("BenchmarkRunner: %d questions × %d systems  (completed=%d)",
                 len(self._questions), len(self._systems), len(completed))

        with self._raw_path.open("a") as fh:
            for q in self._questions:
                for sys_name, system in self._systems.items():
                    key = (q.question_id, sys_name)
                    if self._resume and key in completed:
                        log.debug("Skipping completed: %s × %s", q.question_id, sys_name)
                        continue

                    log.info("Running: %s × %s", q.question_id, sys_name)
                    try:
                        result = system.run(q.question_id, q.natural_query)
                        record = self._to_record(q, sys_name, result)
                    except Exception as exc:
                        log.error("FAILED %s × %s: %s", q.question_id, sys_name, exc)
                        record = self._error_record(q, sys_name, str(exc))

                    fh.write(json.dumps(record) + "\n")
                    fh.flush()

        log.info("BenchmarkRunner: complete. Results at %s", self._raw_path)
        return self._raw_path

    def _load_completed(self) -> set[tuple[str, str]]:
        completed: set[tuple[str, str]] = set()
        if self._raw_path.exists():
            with self._raw_path.open() as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        completed.add((obj["question_id"], obj["system_name"]))
                    except (json.JSONDecodeError, KeyError):
                        pass
        return completed

    def _to_record(self, q: BenchmarkQuestion, sys_name: str, r: GenerationResult) -> dict:
        return {
            "question_id":       q.question_id,
            "query_class_label": q.query_class_label,
            "required_services": q.required_services,
            "natural_query":     q.natural_query,
            "gt_sql_view":       q.gt_sql_view,
            "gt_cypher":         q.gt_cypher,
            "gt_doc_fixture":    q.gt_doc_fixture,
            "af_metric":         q.af_metric,
            "difficulty":        q.difficulty,
            "system_name":       sys_name,
            "queried_service_ids": r.queried_service_ids,
            "answer":            r.answer,
            "retrieval_ms":      r.retrieval_ms,
            "generation_ms":     r.generation_ms,
            "conflict_count":    r.conflict_count,
            "is_cannot_answer":  r.is_cannot_answer,
            "prompt_tokens":     r.prompt_tokens,
            "reply_tokens":      r.reply_tokens,
            "model":             r.model,
        }

    def _error_record(self, q: BenchmarkQuestion, sys_name: str, error: str) -> dict:
        return {
            "question_id": q.question_id, "query_class_label": q.query_class_label,
            "required_services": q.required_services, "natural_query": q.natural_query,
            "gt_sql_view": q.gt_sql_view, "gt_cypher": q.gt_cypher,
            "gt_doc_fixture": q.gt_doc_fixture, "af_metric": q.af_metric,
            "difficulty": q.difficulty, "system_name": sys_name,
            "queried_service_ids": [], "answer": f"[ERROR: {error}]",
            "retrieval_ms": 0.0, "generation_ms": 0.0, "conflict_count": 0,
            "is_cannot_answer": True, "prompt_tokens": 0, "reply_tokens": 0,
            "model": "error",
        }
