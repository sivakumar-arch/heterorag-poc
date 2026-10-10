"""
Regression tests for the loader and tag-format fixes.

The loader imports psycopg2, neo4j and elasticsearch at module level. Those drivers are
not needed to test the logic here, so lightweight stand-ins are installed first when the
real packages are absent.
"""

import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _install_driver_stubs():
    for name in ("psycopg2", "psycopg2.extras", "neo4j", "elasticsearch"):
        try:
            __import__(name)
        except ImportError:
            m = types.ModuleType(name)
            if name == "psycopg2.extras":
                m.execute_values = lambda *a, **k: None
            if name == "neo4j":
                m.GraphDatabase = object
            if name == "elasticsearch":
                m.Elasticsearch = object
                m.helpers = types.ModuleType("elasticsearch.helpers")
            sys.modules[name] = m
            if name == "psycopg2.extras":
                sys.modules["psycopg2"].extras = m


def _load_loader():
    _install_driver_stubs()
    spec = importlib.util.spec_from_file_location(
        "load_so_dump_under_test", ROOT / "scripts" / "load_stackoverflow_dump.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- parse_tags

def test_parse_tags_pipe_format():
    from heterorag.tags import parse_tags
    assert parse_tags("|bayesian|prior|elicitation|") == ["bayesian", "prior", "elicitation"]


def test_parse_tags_legacy_angle_format():
    from heterorag.tags import parse_tags
    assert parse_tags("<python><pandas>") == ["python", "pandas"]


def test_parse_tags_empty_and_none():
    from heterorag.tags import parse_tags
    assert parse_tags(None) == []
    assert parse_tags("") == []
    assert parse_tags("||") == []


def test_parse_tags_keeps_hyphenated_and_dotted_names():
    from heterorag.tags import parse_tags
    assert parse_tags("|python-3.x|c++|") == ["python-3.x", "c++"]


def test_normaliser_extracts_tags_from_pipe_string():
    from heterorag.layer3.result_normaliser import parse_tags as imported
    assert imported("|a|b|") == ["a", "b"]


# ------------------------------------------------------------ dry-run guard

class _FakeCursor:
    def __init__(self, log):
        self.log = log

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        self.log.append((sql, params))


class _FakeConn:
    def __init__(self):
        self.log = []
        self.commits = 0

    def cursor(self):
        return _FakeCursor(self.log)

    def commit(self):
        self.commits += 1


def test_mark_complete_is_skipped_on_dry_run():
    loader = _load_loader()
    conn = _FakeConn()
    loader.mark_complete(conn, "Users.xml", 10, "stats", dry_run=True)
    assert conn.log == []
    assert conn.commits == 0


def test_mark_complete_records_on_real_run():
    loader = _load_loader()
    conn = _FakeConn()
    loader.mark_complete(conn, "Users.xml", 10, "stats", dry_run=False)
    assert len(conn.log) == 1
    assert conn.log[0][1] == ("stats:Users.xml", 10)
    assert conn.commits == 1


# --------------------------------------------------------- ES post filtering

_POSTS_XML = """<?xml version="1.0" encoding="utf-8"?>
<posts>
  <row Id="1" PostTypeId="1" Title="Q" Body="question body" Tags="|python|pandas|" Score="3" />
  <row Id="2" PostTypeId="2" ParentId="1" Body="answer body" Score="1" />
  <row Id="3" PostTypeId="4" Body="tag wiki excerpt" />
  <row Id="4" PostTypeId="5" Body="tag wiki body" />
  <row Id="5" PostTypeId="2" ParentId="1" Body="" Score="0" />
</posts>
"""


def test_es_posts_index_only_questions_and_answers_with_parsed_tags(tmp_path):
    loader = _load_loader()
    (tmp_path / "Posts.xml").write_text(_POSTS_XML, encoding="utf-8")
    captured = []
    loader._es_bulk_flush = lambda es, batch, dry_run: captured.extend(batch)
    loader.load_posts_es(None, tmp_path, 1000, dry_run=False)
    kinds = {(d["post_id"], d["doc_type"]) for d in captured}
    assert kinds == {(1, "question"), (2, "answer")}
    q = next(d for d in captured if d["post_id"] == 1)
    assert q["tags"] == ["python", "pandas"]


# ------------------------------------------------- TAGGED_WITH edge creation

class _Summary:
    def __init__(self, n):
        self.counters = types.SimpleNamespace(relationships_created=n)


class _Result:
    def __init__(self, n=0, existing=None):
        self.n, self.existing = n, existing

    def consume(self):
        return _Summary(self.n)

    def single(self):
        return {"n": self.existing}


class _Session:
    def __init__(self, created_per_batch, existing):
        self.created_per_batch, self.existing = created_per_batch, existing

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute_write(self, fn):
        class _Tx:
            def __init__(s, outer):
                s.outer = outer

            def run(s, query, **kw):
                return _Result(n=s.outer.created_per_batch)
        return fn(_Tx(self))

    def run(self, query, **kw):
        return _Result(existing=self.existing)


class _Driver:
    def __init__(self, created_per_batch, existing=0):
        self.created_per_batch, self.existing = created_per_batch, existing

    def session(self):
        return _Session(self.created_per_batch, self.existing)


def test_tagged_edges_reports_created_count(tmp_path):
    loader = _load_loader()
    (tmp_path / "Posts.xml").write_text(_POSTS_XML, encoding="utf-8")
    loader.load_tagged_edges_neo4j(_Driver(created_per_batch=2), tmp_path, 1000, False)


def test_tagged_edges_raises_when_nothing_matches(tmp_path):
    loader = _load_loader()
    (tmp_path / "Posts.xml").write_text(_POSTS_XML, encoding="utf-8")
    raised = False
    try:
        loader.load_tagged_edges_neo4j(_Driver(created_per_batch=0, existing=0),
                                       tmp_path, 1000, False)
    except RuntimeError:
        raised = True
    assert raised


def test_tagged_edges_zero_created_is_fine_when_edges_already_exist(tmp_path):
    loader = _load_loader()
    (tmp_path / "Posts.xml").write_text(_POSTS_XML, encoding="utf-8")
    loader.load_tagged_edges_neo4j(_Driver(created_per_batch=0, existing=2), tmp_path, 1000, False)


def test_tagged_edges_run_after_post_nodes_in_main():
    src = (ROOT / "scripts" / "load_stackoverflow_dump.py").read_text(encoding="utf-8")
    body = src[src.index("def main"):]
    assert body.index("load_posts_neo4j(") < body.index("load_tagged_edges_neo4j(")


# ------------------------------------------------ AF ground-truth handling

class _PgCursor:
    def __init__(self, conn):
        self.conn = conn
        self.description = [("id",)]
        self._rows = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql):
        if self.conn.aborted:
            raise RuntimeError("current transaction is aborted")
        view = sql.split("FROM")[1].split()[0]
        if view not in self.conn.views:
            self.conn.aborted = True
            raise RuntimeError('relation "%s" does not exist' % view)
        self._rows = [(v,) for v in self.conn.views[view]]

    def fetchall(self):
        return self._rows


class _PgConn:
    def __init__(self, views):
        self.views, self.aborted, self.rollbacks = views, False, 0

    def cursor(self):
        return _PgCursor(self)

    def rollback(self):
        self.aborted = False
        self.rollbacks += 1


def _computer(tmp_path, pg):
    from heterorag.evaluation.metrics import MetricsComputer
    raw = tmp_path / "raw.jsonl"
    raw.write_text("")
    return MetricsComputer(raw, tmp_path / "out", pg_conn=pg)


def test_failed_gt_view_does_not_poison_later_lookups(tmp_path):
    from heterorag.evaluation.metrics import _gt_sql_rows
    pg = _PgConn({"gt_ok": [1234, 5678]})
    assert _gt_sql_rows("gt_missing", pg) == []
    assert pg.rollbacks == 1
    assert len(_gt_sql_rows("gt_ok", pg)) == 2


def test_af_is_none_when_declared_view_is_missing(tmp_path):
    mc = _computer(tmp_path, _PgConn({}))
    rec = {"af_metric": "f1", "answer": "ids 12345", "gt_sql_view": "gt_nope", "gt_cypher": None}
    assert mc._af_for_record(rec) is None


def test_af_is_none_when_declared_view_is_empty(tmp_path):
    mc = _computer(tmp_path, _PgConn({"gt_empty": []}))
    rec = {"af_metric": "f1", "answer": "ids 12345", "gt_sql_view": "gt_empty"}
    assert mc._af_for_record(rec) is None


def test_af_returns_a_score_when_ground_truth_exists(tmp_path):
    # Only checks that a question WITH ground truth is scored, not that the score is
    # meaningful: see the note on AF matching in docs/RERUN_PATCH_NOTES.md.
    mc = _computer(tmp_path, _PgConn({"gt_ok": [1234, 5678]}))
    rec = {"af_metric": "f1", "answer": "the ids are 1234 and 5678", "gt_sql_view": "gt_ok"}
    assert mc._af_for_record(rec) is not None


def test_compute_af_excludes_missing_ground_truth_and_reports_coverage(tmp_path):
    mc = _computer(tmp_path, _PgConn({"gt_ok": [1234]}))
    mc._systems = ["S"]
    recs = [
        {"system_name": "S", "query_class_label": "1", "af_metric": "f1",
         "answer": "1234", "gt_sql_view": "gt_ok"},
        {"system_name": "S", "query_class_label": "4a", "af_metric": "f1",
         "answer": "1234", "gt_sql_view": "gt_missing"},
    ]
    out = mc._compute_af(recs)
    assert out["S"]["1"] is not None
    assert out["S"]["4a"] is None
    assert out["S"]["aggregate"] == out["S"]["1"]
    cov = mc.data_quality["af_ground_truth"]["S"]
    assert cov["4a"] == {"scored": 0, "no_ground_truth": 1}
    assert cov["1"] == {"scored": 1, "no_ground_truth": 0}


# ------------------------------------------------------- question selection

def test_select_questions_filters_and_keeps_benchmark_order():
    from heterorag.evaluation.benchmark_runner import load_benchmark_questions, select_questions
    qs = load_benchmark_questions()
    out = select_questions(qs, ["c5_q01", "c1_q05"])
    assert [q.question_id for q in out] == ["c1_q05", "c5_q01"]


def test_select_questions_none_returns_everything():
    from heterorag.evaluation.benchmark_runner import load_benchmark_questions, select_questions
    qs = load_benchmark_questions()
    assert select_questions(qs, None) == qs
    assert len(select_questions(qs, [])) == len(qs)


def test_select_questions_rejects_unknown_id():
    from heterorag.evaluation.benchmark_runner import load_benchmark_questions, select_questions
    raised = False
    try:
        select_questions(load_benchmark_questions(), ["c1_q05", "c9_q99"])
    except ValueError as e:
        raised = "c9_q99" in str(e)
    assert raised


# ------------------------------------------------------------ model selection

def _with_env(updates, fn):
    import os
    saved = {k: os.environ.get(k) for k in updates}
    try:
        for k, v in updates.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return fn()
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_default_model_is_not_the_retired_one():
    from heterorag.llm_provider import AnthropicProvider
    assert AnthropicProvider.DEFAULT_MODEL != "claude-sonnet-4-20250514"


def test_resolved_model_uses_override_without_provider_var():
    from heterorag.llm_provider import resolved_model_name
    out = _with_env({"HETERORAG_LLM_PROVIDER": None, "HETERORAG_LLM_MODEL": "claude-sonnet-4-6"},
                    resolved_model_name)
    assert out == "claude-sonnet-4-6"


def test_resolved_model_falls_back_to_provider_default():
    from heterorag.llm_provider import resolved_model_name, AnthropicProvider
    out = _with_env({"HETERORAG_LLM_PROVIDER": None, "HETERORAG_LLM_MODEL": None},
                    resolved_model_name)
    assert out == AnthropicProvider.DEFAULT_MODEL


def test_model_env_var_alone_reaches_the_translation_llm():
    # Setting only HETERORAG_LLM_MODEL used to be ignored (the retired default was used).
    try:
        import anthropic  # noqa: F401
    except ImportError:
        stub = types.ModuleType("anthropic")
        stub.Anthropic = lambda **kw: object()
        sys.modules["anthropic"] = stub
    from heterorag.layer2.translation_llm import TranslationLLM

    def build():
        llm = TranslationLLM(api_key="test-key")
        inner = llm._provider
        while hasattr(inner, "_inner") or hasattr(inner, "_provider"):
            inner = getattr(inner, "_inner", None) or getattr(inner, "_provider")
        return getattr(inner, "model", None)

    out = _with_env({"HETERORAG_LLM_PROVIDER": None, "HETERORAG_LLM_MODEL": "claude-sonnet-4-6",
                     "ANTHROPIC_API_KEY": "test-key"}, build)
    assert out == "claude-sonnet-4-6"


# -------------------------------------------------- temperature not accepted

class _Resp:
    class _U:
        input_tokens, output_tokens = 10, 3
    usage = _U()

    class _C:
        text = " ok "
    content = [_C()]


class _FakeMessages:
    def __init__(self, mode):
        self.mode, self.calls = mode, []

    def create(self, **kw):
        self.calls.append(dict(kw))
        if "temperature" in kw:
            if self.mode == "typeerror":
                raise TypeError("Messages.create() got an unexpected keyword argument 'temperature'")
            if self.mode == "http400":
                err = RuntimeError("temperature is not supported for this model")
                err.status_code = 400
                raise err
            if self.mode == "other400":
                err = RuntimeError("max_tokens too large")
                err.status_code = 400
                raise err
        return _Resp()


def _provider_with(mode):
    try:
        import anthropic  # noqa: F401
    except ImportError:
        stub = types.ModuleType("anthropic")
        stub.Anthropic = lambda **kw: object()
        sys.modules["anthropic"] = stub
    from heterorag.llm_provider import AnthropicProvider
    p = AnthropicProvider(model="claude-sonnet-5-5", api_key="test-key")
    p._client = types.SimpleNamespace(messages=_FakeMessages(mode))
    return p


def test_provider_retries_without_temperature_on_typeerror():
    p = _provider_with("typeerror")
    out = p.complete("hi", max_tokens=5, temperature=0.0)
    calls = p._client.messages.calls
    assert out.text == "ok" and len(calls) == 2
    assert "temperature" in calls[0] and "temperature" not in calls[1]


def test_provider_retries_without_temperature_on_http_400():
    p = _provider_with("http400")
    p.complete("hi")
    assert p.temperature_supported is False


def test_provider_does_not_send_temperature_again_after_refusal():
    p = _provider_with("typeerror")
    p.complete("one")
    p.complete("two")
    calls = p._client.messages.calls
    assert len(calls) == 3                      # first try, retry, then one clean call
    assert "temperature" not in calls[2]


def test_provider_does_not_swallow_unrelated_400():
    p = _provider_with("other400")
    raised = False
    try:
        p.complete("hi")
    except RuntimeError:
        raised = True
    assert raised and p.temperature_supported is True


# ------------------------------------------------ thinking blocks in responses

class _Block:
    def __init__(self, type_, text=None):
        self.type = type_
        if text is not None:
            self.text = text


class _Details:
    thinking_tokens = 40


class _Usage:
    input_tokens, output_tokens = 100, 120
    output_tokens_details = _Details()


class _ThinkingResp:
    def __init__(self, blocks, stop_reason="end_turn"):
        self.content, self.usage, self.stop_reason = blocks, _Usage(), stop_reason


def _provider_returning(resp):
    p = _provider_with("ok")
    p._client = types.SimpleNamespace(
        messages=types.SimpleNamespace(create=lambda **kw: resp))
    return p


def test_text_is_taken_from_text_block_after_a_thinking_block():
    resp = _ThinkingResp([_Block("thinking"), _Block("text", " the answer ")])
    out = _provider_returning(resp).complete("q")
    assert out.text == "the answer"
    assert out.thinking_tokens == 40 and out.reply_tokens == 120


def test_multiple_text_blocks_are_joined_in_order():
    resp = _ThinkingResp([_Block("text", "a"), _Block("text", "b")])
    assert _provider_returning(resp).complete("q").text == "ab"


def test_response_with_only_thinking_raises_a_clear_error():
    resp = _ThinkingResp([_Block("thinking")], stop_reason="max_tokens")
    msg = ""
    try:
        _provider_returning(resp).complete("q")
    except RuntimeError as e:
        msg = str(e)
    assert "no text block" in msg and "max_tokens" in msg


def test_empty_content_still_returns_empty_text():
    resp = _ThinkingResp([])
    assert _provider_returning(resp).complete("q").text == ""


# ------------------------------------------------ reasoning exhausting max_tokens

def _sequence_provider(responses, seen):
    p = _provider_with("ok")
    it = iter(responses)

    def create(**kw):
        seen.append(kw["max_tokens"])
        return next(it)

    p._client = types.SimpleNamespace(messages=types.SimpleNamespace(create=create))
    return p


def test_thinking_only_reply_is_retried_once_with_a_larger_budget():
    seen = []
    first = _ThinkingResp([_Block("thinking")], stop_reason="max_tokens")
    second = _ThinkingResp([_Block("thinking"), _Block("text", "SELECT 1")])
    out = _sequence_provider([first, second], seen).complete("q", max_tokens=512)
    assert seen == [512, 2048]
    assert out.text == "SELECT 1"
    # both attempts are billed, so both are counted
    assert out.reply_tokens == 240 and out.prompt_tokens == 200 and out.thinking_tokens == 80


def test_no_retry_when_the_budget_is_already_at_the_ceiling():
    seen = []
    resp = _ThinkingResp([_Block("thinking")], stop_reason="max_tokens")
    p = _sequence_provider([resp], seen)
    raised = False
    try:
        p.complete("q", max_tokens=p.MAX_TOKENS_CEILING)
    except RuntimeError:
        raised = True
    assert raised and len(seen) == 1


def test_no_retry_for_other_stop_reasons():
    seen = []
    resp = _ThinkingResp([_Block("thinking")], stop_reason="end_turn")
    raised = False
    try:
        _sequence_provider([resp], seen).complete("q", max_tokens=512)
    except RuntimeError:
        raised = True
    assert raised and seen == [512]


def test_default_token_limits_leave_room_for_reasoning():
    from heterorag.layer2 import translation_llm
    from heterorag.layer4 import generation
    assert translation_llm._DEFAULT_MAX_TOKENS >= 2048
    assert generation._MAX_TOKENS >= 4096


# ------------------------------------------------ abstain_short_circuit default and guard

def _bare_runner(tmp_path, flag):
    from heterorag.evaluation.benchmark_runner import BenchmarkRunner
    return BenchmarkRunner(systems={}, questions=[], output_dir=tmp_path,
                           abstain_short_circuit=flag)


def test_run_meta_records_the_abstain_setting(tmp_path):
    import json
    r = _bare_runner(tmp_path, True)
    r._write_run_meta()
    meta = json.loads((tmp_path / "run_meta.jsonl").read_text().splitlines()[-1])
    assert meta["abstain_short_circuit"] is True


def test_resume_into_a_directory_with_a_different_setting_is_refused(tmp_path):
    import json
    (tmp_path / "run_meta.jsonl").write_text(json.dumps({"git_commit": "x"}) + "\n")  # old: no key
    r = _bare_runner(tmp_path, True)
    raised = False
    try:
        r._check_config_not_mixed({("q", "s", 0)})
    except RuntimeError as e:
        raised = "new --output-dir" in str(e)
    assert raised


def test_same_setting_or_empty_results_may_resume(tmp_path):
    import json
    (tmp_path / "run_meta.jsonl").write_text(json.dumps({"abstain_short_circuit": True}) + "\n")
    _bare_runner(tmp_path, True)._check_config_not_mixed({("q", "s", 0)})
    _bare_runner(tmp_path, False)._check_config_not_mixed(set())   # nothing completed yet


def test_cli_default_is_on_and_can_be_switched_off():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--abstain-short-circuit", action=argparse.BooleanOptionalAction, default=True)
    assert p.parse_args([]).abstain_short_circuit is True
    assert p.parse_args(["--no-abstain-short-circuit"]).abstain_short_circuit is False
    src = open("evaluation/run_benchmark.py").read()
    assert "BooleanOptionalAction" in src and "default=True" in src
