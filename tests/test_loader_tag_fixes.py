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
