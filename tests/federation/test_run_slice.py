"""Vertical-slice runner: plan equality, outcome classes, per-question rows, summary."""
import json

import pytest

from benchmark_v2 import run_slice
from benchmark_v2.run_slice import classify, plans_equal, run_one, signature
from heterorag.federation import Executor, FetchResult, Guarantee, Plan, Store
from heterorag.federation.answer import Answer, abstain
from heterorag.federation.pipeline import Federation
from heterorag.federation.planner import Planner
from heterorag.federation.translator import Translator
from tests.benchmark_v2.test_questions import world  # noqa: F401  (fixture)
from tests.federation.fake_descriptors import DESCRIPTORS, catalogue
from tests.federation.test_planner_translator import DOC_Q, GRAPH_Q, metered
from tests.federation.test_reference_and_validation import records_for


def P(*steps):
    return Plan.from_dict({"steps": list(steps)})


def R(i, store, entity="question", **kw):
    d = {"id": i, "op": "retrieve", "store": store, "question": "q", "output": entity}
    if store == "doc":
        d["mode"] = "filter"
    d.update(kw)
    return d


def C(i, op, ins, entity="question"):
    return {"id": i, "op": op, "inputs": ins, "output": entity}


# ------------------------------------------------------------------ plan equality

def test_signature_ignores_names_wording_and_input_order_of_symmetric_operators():
    a = P(R("a", "graph"), R("b", "doc"), C("c", "intersect", ["a", "b"]))
    b = P(R("x", "doc", question="other words"), R("y", "graph"), C("z", "intersect", ["x", "y"]))
    assert signature(a) == signature(b) and plans_equal(b, a)


def test_signature_distinguishes_structure():
    base = P(R("a", "graph"), R("b", "doc"), C("c", "intersect", ["a", "b"]))
    assert not plans_equal(P(R("a", "sql"), R("b", "doc"), C("c", "intersect", ["a", "b"])), base)
    assert not plans_equal(P(R("a", "graph"), R("b", "doc"), C("c", "union", ["a", "b"])), base)
    assert not plans_equal(P(R("a", "graph"), R("b", "doc", output="answer"), C("c", "intersect", ["a", "b"], "answer")), base)
    assert not plans_equal(None, base)
    sub1 = P(R("a", "graph"), R("b", "doc"), C("c", "subtract", ["a", "b"]))
    sub2 = P(R("a", "graph"), R("b", "doc"), C("c", "subtract", ["b", "a"]))
    assert not plans_equal(sub1, sub2)                    # order matters for difference
    dep = P(R("a", "sql", "user"), R("b", "graph", inputs=["a"], bind={"input": "a", "entity": "user"}))
    assert signature(dep)[6] == ("user",) and signature(dep)[-1][0][0] == "retrieve"   # the bind and the dependency are part of the structure


# ------------------------------------------------------------------ outcome classes

@pytest.mark.parametrize("ans,rec,want", [
    (Answer("ids", [1, 2], Guarantee.EXACT), {"answer_type": "ids", "gold": [2, 1]}, "correct"),
    (Answer("ids", [1, 3], Guarantee.EXACT), {"answer_type": "ids", "gold": [1, 2]}, "wrong"),
    (Answer("ids", [1], Guarantee.SUBSET), {"answer_type": "ids", "gold": [1, 2]}, "partial_valid"),
    (Answer("ids", [1, 9], Guarantee.SUBSET), {"answer_type": "ids", "gold": [1, 2]}, "partial_violated"),
    (Answer("ids", [1, 2, 3], Guarantee.SUPERSET), {"answer_type": "ids", "gold": [1, 2]}, "partial_valid"),
    (Answer("ranked", [3, 1], Guarantee.EXACT), {"answer_type": "ranked", "gold": [3, 1]}, "correct"),
    (Answer("ranked", [1, 3], Guarantee.EXACT), {"answer_type": "ranked", "gold": [3, 1]}, "wrong"),
    (Answer("scalar", 5, Guarantee.EXACT), {"answer_type": "scalar", "gold": 5}, "correct"),
    (Answer("scalar", 4, Guarantee.LOWER), {"answer_type": "scalar", "gold": 5}, "partial_valid"),
    (Answer("scalar", 6, Guarantee.LOWER), {"answer_type": "scalar", "gold": 5}, "partial_violated"),
    (abstain("incomplete"), {"answer_type": "ids", "gold": [1]}, "abstain_unexpected"),
    (abstain("premise_failed"), {"answer_type": "ids", "gold": [], "expect_abstain": True}, "abstain_expected"),
])
def test_classify(ans, rec, want):
    assert classify(ans, rec) == want


# ------------------------------------------------------------------ end to end with scripted model and fake stores

class Fixed:
    def __init__(self, rows):
        self.rows = rows

    def fetch(self, step, query, binds, limit):
        return FetchResult(list(self.rows), True)


def federation_for(rec, plan_json):
    gold = list(rec["gold"])
    pm, _ = metered([plan_json])
    tm, _ = metered(lambda p: DOC_Q if "Elasticsearch" in p else GRAPH_Q)
    clients = {Store.GRAPH: Fixed(gold + [900001]), Store.DOC: Fixed(gold + [900002]), Store.SQL: Fixed([])}
    return Federation(Planner(pm, DESCRIPTORS), Translator(tm, DESCRIPTORS, catalogue()), Executor(clients))


def test_run_one_scores_each_stage(world):
    rec = records_for(world, ["P01"], per=1)[0]
    row = run_one(rec, federation_for(rec, json.dumps({"steps": rec["plan"]})))
    assert row["plan_valid"] and row["plan_equal"] and row["translation_ok"] and row["outcome"] == "correct"
    assert row["usage"]["planner"]["calls"] == 1 and row["usage"]["translator"]["calls"] == 2
    assert row["native_calls"] == 2 and row["gold_plan"] == rec["plan"]

    swapped = [dict(s) for s in rec["plan"]]
    swapped[0]["store"] = "doc"; swapped[0]["mode"] = "filter"                 # wrong store for step a
    row = run_one(rec, federation_for(rec, json.dumps({"steps": swapped})))
    assert row["plan_valid"] and not row["plan_equal"]


def test_test_split_rows_carry_no_gold(world):
    rec = dict(records_for(world, ["P01"], per=1)[0], split="test")
    row = run_one(rec, federation_for(rec, json.dumps({"steps": rec["plan"]})))
    assert "gold_plan" not in row and "got" not in row


def test_main_writes_rows_and_prints_summary(world, tmp_path, capsys):
    recs = records_for(world, ["P01", "P04"], per=2)
    qf = tmp_path / "q.jsonl"
    qf.write_text("\n".join(json.dumps(r) for r in recs))
    by_t = {r["template"]: r for r in recs}

    class Router:                       # answers with the annotated plan of the question asked
        def answer(self, text):
            rec = next(r for r in recs if r["text"] == text)
            return federation_for(rec, json.dumps({"steps": rec["plan"]})).answer(text)

    out = tmp_path / "o.json"
    rc = run_slice.main(["--questions", str(qf), "--templates", "P01", "--limit", "1", "--out", str(out)],
                        federation=Router(), models=None)
    assert rc == 0
    data = json.loads(out.read_text())
    assert data["meta"]["n"] == 1 and data["rows"][0]["template"] == "P01"
    text = capsys.readouterr().out
    assert "plan_eq" in text and "P01" in text and "all" in text
    assert run_slice.main(["--questions", str(qf), "--templates", "ZZ", "--out", str(out)], federation=Router()) == 2


def test_a_crash_is_recorded_not_hidden(world, tmp_path):
    recs = records_for(world, ["P01"], per=1)
    qf = tmp_path / "q.jsonl"
    qf.write_text(json.dumps(recs[0]))

    class Boom:
        def answer(self, text):
            raise RuntimeError("api down")

    out = tmp_path / "o.json"
    run_slice.main(["--questions", str(qf), "--out", str(out)], federation=Boom())
    row = json.loads(out.read_text())["rows"][0]
    assert row["outcome"] == "abstain_unexpected" and "api down" in row["reason"]
