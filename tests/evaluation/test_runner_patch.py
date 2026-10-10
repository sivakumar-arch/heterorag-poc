"""
Tests for the rerun-hardening patch: retry/backoff, drop reasons, resume
semantics, sequential schedule, trace, and metrics exclusion.

No live services or API keys are needed: LLMs are scripted providers, and the
retrieval layer is replaced by an in-memory executor.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from heterorag.evaluation.benchmark_runner import (
    BenchmarkRunner, record_key, record_status,
)
from heterorag.evaluation.baselines import BaselineSystem
from heterorag.evaluation.metrics import (
    MetricsComputer, cluster_bootstrap_ci, compute_sc, compute_service_prf,
)
from heterorag.layer1.poc_descriptors import (
    CONTENT_SERVICE_DESCRIPTOR, KNOWLEDGE_GRAPH_DESCRIPTOR,
    USER_ACTIVITY_DESCRIPTOR, build_poc_registry,
)
from heterorag.layer2.translation_llm import TranslationLLM
from heterorag.layer3 import ConnectionRegistry
from heterorag.layer3.models import RawServiceResult, SourceType
from heterorag.layer3.retrieval_executor import ParallelRetrievalExecutor
from heterorag.layer4.generation import GenerationLLM, GenerationResult
from heterorag.layer4.pipeline import HeteroRAGPipeline
from heterorag.llm_provider import (
    LLM_USAGE, LLMProvider, LLMResponse, RetryingProvider, is_retryable_error,
)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

class RateLimitError(Exception):
    status_code = 429


class ScriptedProvider(LLMProvider):
    """Replies by service keyword in the prompt; optional failure schedule."""

    model = "scripted"

    def __init__(self, replies=None, fail_first: int = 0, exc=RateLimitError):
        self.replies = replies or {}
        self.fail_first = fail_first
        self.exc = exc
        self.calls = 0

    def complete(self, prompt, *, max_tokens=512, temperature=0.0):
        self.calls += 1
        if self.calls <= self.fail_first:
            raise self.exc("slow down")
        for key, text in self.replies.items():
            if key in prompt:
                return LLMResponse(text=text, prompt_tokens=10, reply_tokens=5, model="scripted")
        return LLMResponse(text="ok", prompt_tokens=10, reply_tokens=5, model="scripted")


SQL_PROMPT_KEY = "precise SQL query translator"
CYPHER_PROMPT_KEY = "precise Cypher query translator"
BM25_PROMPT_KEY = "BM25 search query builder"


def make_registry_with_rows(rows_by_service, delay_s=0.0):
    reg = ConnectionRegistry()
    for d in (USER_ACTIVITY_DESCRIPTOR, KNOWLEDGE_GRAPH_DESCRIPTOR, CONTENT_SERVICE_DESCRIPTOR):
        reg.register(d.service_id, d.connection)
    return reg


class FakeExecutor(ParallelRetrievalExecutor):
    """Replaces only the per-service dispatch; scheduling code is the real one."""

    def __init__(self, rows_by_service, delay_s=0.0, errors=None, **kw):
        super().__init__(ConnectionRegistry(), **kw)
        self._registry.register  # noqa
        for d in (USER_ACTIVITY_DESCRIPTOR, KNOWLEDGE_GRAPH_DESCRIPTOR, CONTENT_SERVICE_DESCRIPTOR):
            self._registry.register(d.service_id, d.connection)
        self.rows_by_service = rows_by_service
        self.delay_s = delay_s
        self.errors = errors or {}

    def _dispatch(self, sq, cfg, source_type):
        time.sleep(self.delay_s)
        if sq.service_id in self.errors:
            raise self.errors[sq.service_id]
        return list(self.rows_by_service.get(sq.service_id, []))


def build_pipeline(replies, rows, *, mode="parallel", delay_s=0.0, errors=None,
                   fail_first=0, abstain_short_circuit=False):
    prov = ScriptedProvider(replies, fail_first=fail_first)
    t = TranslationLLM(provider=RetryingProvider(prov, max_retries=0))
    g = GenerationLLM(inject_answer="ANSWER")
    p = HeteroRAGPipeline(t, g, ConnectionRegistry(), execution_mode=mode,
                          abstain_short_circuit=abstain_short_circuit)
    p._executor = FakeExecutor(rows, delay_s=delay_s, errors=errors, execution_mode=mode)
    return p, prov


GOOD = {
    SQL_PROMPT_KEY: "SELECT id FROM users LIMIT 5",
    CYPHER_PROMPT_KEY: "MATCH (t:Tag) RETURN t.name LIMIT 5",
    BM25_PROMPT_KEY: "pca eigenvalues",
}
ROWS = {
    "user-activity-service": [{"id": 1, "display_name": "a"}],
    "knowledge-graph-service": [{"name": "pca"}],
    "content-service": [{"_id": "7", "title": "t", "body": "b", "_score": 1.0}],
}


def i1_all():
    reg = build_poc_registry(heartbeat_timeout_seconds=86400)
    return reg.discover("q", "question")


# ----------------------------------------------------------------------------
# retry / backoff
# ----------------------------------------------------------------------------

def test_retryable_classification():
    assert is_retryable_error(RateLimitError("x"))
    assert is_retryable_error(Exception("Overloaded"))
    assert not is_retryable_error(ValueError("bad prompt"))


def test_retry_recovers_from_429_and_is_metered():
    waits = []
    prov = ScriptedProvider(fail_first=2)
    rp = RetryingProvider(prov, max_retries=5, base_delay=1.0, sleep=waits.append)
    before = LLM_USAGE.snapshot()
    resp = rp.complete("hello")
    d = LLM_USAGE.delta(LLM_USAGE.snapshot(), before)
    assert resp.text == "ok" and prov.calls == 3
    assert d["retries"] == 2 and d["calls"] == 1 and d["failures"] == 0
    assert len(waits) == 2 and waits[1] >= 0.75 * 2.0     # exponential: ~1s then ~2s (with jitter)


def test_retry_gives_up_and_raises():
    prov = ScriptedProvider(fail_first=99)
    rp = RetryingProvider(prov, max_retries=2, sleep=lambda s: None)
    try:
        rp.complete("x")
    except RateLimitError:
        assert prov.calls == 3
    else:
        raise AssertionError("expected RateLimitError")


def test_non_retryable_not_retried():
    prov = ScriptedProvider(fail_first=99, exc=ValueError)
    rp = RetryingProvider(prov, max_retries=5, sleep=lambda s: None)
    try:
        rp.complete("x")
    except ValueError:
        assert prov.calls == 1


# ----------------------------------------------------------------------------
# planner: drop reasons
# ----------------------------------------------------------------------------

def test_drop_reasons_abstain_invalid_and_error():
    replies = {
        SQL_PROMPT_KEY: "-- CANNOT_ANSWER",
        CYPHER_PROMPT_KEY: "DROP TABLE x",          # SQL masquerading as Cypher -> invalid
        BM25_PROMPT_KEY: "pca",
    }
    p, _ = build_pipeline(replies, ROWS)
    res = p.run(i1_all(), "question")
    dropped = res.trace["dropped"]
    assert dropped["user-activity-service"] == "abstain"
    assert dropped["knowledge-graph-service"] == "invalid_query"
    # the raw LLM text of dropped services is kept for diagnosis
    dt = res.trace["dropped_translations"]
    assert "CANNOT_ANSWER" in dt["user-activity-service"]
    assert dt["knowledge-graph-service"] == "DROP TABLE x"
    assert res.trace["translated_service_ids"] == ["content-service"]
    assert res.trace["infra_error"] is False


def test_translation_error_is_flagged_infra_not_a_normal_drop():
    # provider fails every call (retries disabled in build_pipeline) -> all translations error
    p, prov = build_pipeline(GOOD, ROWS, fail_first=999)
    res = p.run(i1_all(), "question")
    assert set(res.trace["dropped"].values()) == {"translation_error"}
    assert res.trace["infra_error"] is True
    # validator must NOT have burned extra retry calls on infrastructure failures
    assert prov.calls == 3


def test_abstain_short_circuit_saves_the_validator_retry():
    replies = {**GOOD, SQL_PROMPT_KEY: "-- CANNOT_ANSWER"}
    p0, prov0 = build_pipeline(replies, ROWS, abstain_short_circuit=False)
    p1, prov1 = build_pipeline(replies, ROWS, abstain_short_circuit=True)
    r0 = p0.run(i1_all(), "q"); r1 = p1.run(i1_all(), "q")
    assert prov0.calls == prov1.calls + 1          # the retry prompt
    assert r0.trace["dropped"] == r1.trace["dropped"] == {"user-activity-service": "abstain"}
    assert "CANNOT_ANSWER" in r1.trace["dropped_translations"]["user-activity-service"]


# ----------------------------------------------------------------------------
# executor schedule + trace
# ----------------------------------------------------------------------------

def test_sequential_costs_sum_parallel_costs_max():
    par, _ = build_pipeline(GOOD, ROWS, mode="parallel", delay_s=0.15)
    seq, _ = build_pipeline(GOOD, ROWS, mode="sequential", delay_s=0.15)
    rp = par.run(i1_all(), "q"); rs = seq.run(i1_all(), "q")
    assert rp.retrieval_ms < 300
    assert rs.retrieval_ms >= 3 * 150 * 0.95
    assert sorted(rp.queried_service_ids) == sorted(rs.queried_service_ids)
    assert rp.answer == rs.answer == "ANSWER"


def test_retrieval_error_is_query_error_not_infra_but_timeout_is():
    class Boom(Exception): pass
    p, _ = build_pipeline(GOOD, ROWS, errors={"user-activity-service": Boom("column x does not exist")})
    res = p.run(i1_all(), "q")
    r = res.trace["retrieval"]["user-activity-service"]
    assert r["succeeded"] is False and r["error_type"] == "Boom"
    assert res.trace["infra_error"] is False            # a wrong query is a system outcome
    assert "user-activity-service" not in res.queried_service_ids

    class OperationalError(Exception): pass
    p, _ = build_pipeline(GOOD, ROWS, errors={"content-service": OperationalError("conn refused")})
    assert p.run(i1_all(), "q").trace["infra_error"] is True


def test_trace_has_stage_timings_and_e2e():
    p, _ = build_pipeline(GOOD, ROWS)
    res = p.run(i1_all(), "q")
    ms = res.trace["ms"]
    assert set(ms) >= {"plan", "retrieve", "integrate", "generate", "e2e"}
    assert ms["e2e"] >= ms["plan"]


# ----------------------------------------------------------------------------
# runner: resume + retry semantics
# ----------------------------------------------------------------------------

class Flaky(BaselineSystem):
    system_name = "Flaky"

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def run(self, query_id, natural_query):
        self.calls += 1
        o = self.outcomes.pop(0) if self.outcomes else "ok"
        if o == "raise":
            raise RuntimeError("boom")
        r = GenerationResult(query_id=query_id, natural_query=natural_query, answer="fine",
                             queried_service_ids=["user-activity-service"], retrieval_ms=5.0)
        if o == "infra":
            r.trace = {"infra_error": True, "infra_reasons": ["translation:x:429"]}
        return r


def make_runner(tmp_path, system, n_q=2, **kw):
    from heterorag.evaluation.benchmark_runner import load_benchmark_questions
    qs = load_benchmark_questions()[:n_q]
    return BenchmarkRunner({"Flaky": system}, qs, tmp_path, sleep=lambda s: None, **kw), qs


def read(tmp_path):
    return [json.loads(l) for l in (tmp_path / "raw_results.jsonl").read_text().splitlines()]


def test_runner_retries_errors_within_run(tmp_path):
    runner, qs = make_runner(tmp_path, Flaky(["raise", "infra", "ok"]), n_q=1, max_attempts=3)
    runner.run()
    recs = read(tmp_path)
    assert len(recs) == 1 and recs[0]["status"] == "ok" and recs[0]["attempt"] == 3


def test_resume_reruns_failures_but_keeps_successes(tmp_path):
    sysm = Flaky(["raise", "ok"])
    runner, qs = make_runner(tmp_path, sysm, n_q=2, max_attempts=1)
    runner.run()                                   # q1 errors (1 attempt), q2 ok
    first = read(tmp_path)
    assert [r["status"] for r in first] == ["error", "ok"]
    assert first[0]["answer"].startswith("[ERROR")

    runner2, _ = make_runner(tmp_path, sysm, n_q=2, max_attempts=1)
    runner2.run()                                  # resume: only q1 is retried
    after = read(tmp_path)
    assert len(after) == 3 and after[-1]["question_id"] == qs[0].question_id
    assert after[-1]["status"] == "ok"
    assert sysm.calls == 3


def test_legacy_error_records_are_not_completed(tmp_path):
    (tmp_path / "raw_results.jsonl").write_text(json.dumps({
        "question_id": "c1_q01", "system_name": "Flaky", "answer": "[ERROR: 429]", "model": "error",
    }) + "\n")
    runner, _ = make_runner(tmp_path, Flaky([]), n_q=1)
    assert runner._load_completed() == set()
    assert record_status({"answer": "x", "model": "m"}) == "ok"
    assert record_key({"question_id": "a", "system_name": "b"}) == ("a", "b", 0)


def test_repeats_get_distinct_keys(tmp_path):
    runner, _ = make_runner(tmp_path, Flaky([]), n_q=1, repeats=3)
    runner.run()
    recs = read(tmp_path)
    assert sorted(r["repeat"] for r in recs) == [0, 1, 2]
    # re-running skips all three
    runner2, _ = make_runner(tmp_path, Flaky([]), n_q=1, repeats=3)
    runner2.run()
    assert len(read(tmp_path)) == 3


def test_record_contains_llm_usage_and_e2e(tmp_path):
    runner, _ = make_runner(tmp_path, Flaky([]), n_q=1)
    runner.run()
    r = read(tmp_path)[0]
    assert r["schema_version"] == 2 and "e2e_ms" in r and "llm" in r and "trace" in r
    assert (tmp_path / "run_meta.jsonl").exists()


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------

def _rec(qid, system, status="ok", queried=("user-activity-service",), cls="1", req=("sql",),
         repeat=0, trace=None, rl=10.0, **extra):
    r = {"question_id": qid, "system_name": system, "status": status, "repeat": repeat,
         "query_class_label": cls, "required_services": list(req), "natural_query": "q",
         "queried_service_ids": list(queried), "answer": "a", "retrieval_ms": rl,
         "generation_ms": 1.0, "conflict_count": 0, "is_cannot_answer": False,
         "prompt_tokens": 1, "reply_tokens": 1, "model": "m", "af_metric": "f1"}
    if trace is not None:
        r["trace"] = trace
    r.update(extra)
    return r


def write(tmp_path, recs):
    (tmp_path / "raw_results.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    return tmp_path / "raw_results.jsonl"


def test_metrics_exclude_failed_runs_and_use_latest_ok(tmp_path):
    recs = [
        _rec("c1_q01", "S", status="infra_error", queried=()),          # excluded
        _rec("c1_q01", "S", queried=("user-activity-service",)),         # retried -> ok
        _rec("c1_q02", "S", status="error", queried=()),                 # never recovered
        _rec("c1_q03", "S", queried=()),
        _rec("c1_q03", "S", queried=("user-activity-service",)),         # superseded by later ok
    ]
    mc = MetricsComputer(write(tmp_path, recs), tmp_path)
    out = mc.compute()
    assert out["source_coverage"]["S"]["1"] == 1.0
    dq = out["data_quality"]["S"]
    assert dq["ok"] == 2 and dq["infra_error"] == 1 and dq["error"] == 1
    assert dq["superseded"] == 1 and dq["n_missing"] == 1


def test_precision_aware_selection_penalises_query_everything():
    p, r, f1 = compute_service_prf(["sql"], ["user-activity-service", "content-service", "knowledge-graph-service"])
    assert (round(p, 3), r) == (0.333, 1.0) and compute_sc(["sql"], ["user-activity-service", "content-service"]) == 1.0
    assert compute_service_prf(["sql"], []) == (0.0, 0.0, 0.0)


def test_drop_breakdown_and_cost_outputs(tmp_path):
    tr = {"shortlisted_service_ids": ["user-activity-service", "knowledge-graph-service"],
          "translated_service_ids": ["knowledge-graph-service"],
          "dropped": {"user-activity-service": "abstain"},
          "retrieval": {"knowledge-graph-service": {"succeeded": True, "rows": 0}}}
    recs = [_rec("c4a_q01", "S", cls="4a", req=("sql", "graph", "document"),
                 queried=("knowledge-graph-service",), trace=tr, e2e_ms=100.0,
                 llm={"calls": 4, "prompt_tokens": 10, "reply_tokens": 5, "retries": 1})]
    out = MetricsComputer(write(tmp_path, recs), tmp_path).compute()
    d = out["drop_breakdown"]["S"]
    assert d["sql"]["abstain"] == 1 and d["graph"]["empty"] == 1 and d["document"]["not_selected"] == 1
    assert out["cost"]["S"]["llm_calls_mean"] == 4 and out["cost"]["S"]["llm_retries_total"] == 1
    prf = out["service_selection_prf"]["S"]
    assert prf["selected_recall"] == round(2 / 3, 4)
    for f in ("drop_breakdown.csv", "cost_per_query.csv", "service_selection_prf.csv"):
        assert (tmp_path / f).exists()


def test_cluster_bootstrap_resamples_questions_not_runs():
    vals = {"a": [1.0, 1.0, 1.0], "b": [0.0]}
    m, lo, hi = cluster_bootstrap_ci(vals, n_boot=500)
    assert m == 0.5 and lo <= m <= hi
    assert cluster_bootstrap_ci({"a": [1.0]}) == (1.0, 1.0, 1.0)


def test_new_system_names_appear_in_tables(tmp_path):
    recs = [_rec("c1_q01", "Select_Sequential"), _rec("c1_q01", "My_New_System")]
    out = MetricsComputer(write(tmp_path, recs), tmp_path).compute()
    assert "Select_Sequential" in out["source_coverage"] and "My_New_System" in out["source_coverage"]


def test_circuit_breaker_stops_a_dead_stack(tmp_path):
    sysm = Flaky(["raise"] * 50)
    runner, _ = make_runner(tmp_path, sysm, n_q=20, max_attempts=1, max_consecutive_failures=3)
    try:
        runner.run()
    except RuntimeError as e:
        assert "3 consecutive runs failed" in str(e)
        assert len(read(tmp_path)) == 3 and sysm.calls == 3
    else:
        raise AssertionError("expected the circuit breaker to trip")


def test_missing_driver_counts_as_infra_error():
    from heterorag.layer4.trace import INFRA_RETRIEVAL_ERRORS
    assert "ModuleNotFoundError" in INFRA_RETRIEVAL_ERRORS
