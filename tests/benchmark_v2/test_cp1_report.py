"""The CP1 report: outcome categories, F1 definitions, cost, repeats, intervals and the decision rule."""

import json

import pytest

from benchmark_v2 import cp1_report as cp
from benchmark_v2.run_slice import answer_scores
from heterorag.federation.answer import Answer
from heterorag.federation.guarantee import Guarantee


def pipe(qid, outcome, f1=None, template="D01", stratum="s", tin=1000, tout=100):
    return {"question_id": qid, "template": template, "stratum": stratum, "outcome": outcome, "f1": f1,
            "usage": {"planner": {"prompt_tokens": tin, "reply_tokens": tout},
                      "translator": {"prompt_tokens": tin, "reply_tokens": tout}},
            "rows_read": 10, "total_s": 2.0}


def agent(qid, outcome, f1=None, template="D01", stratum="s", tin=4000, tout=400):
    return {"question_id": qid, "template": template, "stratum": stratum, "outcome": outcome, "f1": f1,
            "usage": {"prompt_tokens": tin, "reply_tokens": tout, "calls": 3, "seconds": 1.0},
            "rows_read": 50, "total_s": 5.0}


def write(tmp_path, name, rows):
    p = tmp_path / name
    p.write_text(json.dumps({"meta": {}, "rows": rows}))
    return str(p)


def test_categories_follow_the_definitions():
    assert cp.category({"outcome": "correct"}) == "correct"
    assert cp.category({"outcome": "wrong"}) == "silent"
    assert cp.category({"outcome": "partial_violated"}) == "silent"          # the label did not hold
    assert cp.category({"outcome": "partial_valid"}) == "flagged"
    for o in ("abstain_expected", "abstain_unexpected", "no_answer"):
        assert cp.category({"outcome": o}) == "abstained"
    with pytest.raises(ValueError):
        cp.category({"outcome": "surprise", "question_id": "x"})


def test_f1_definitions_for_each_outcome():
    assert cp.f1_of({"outcome": "correct", "f1": None}) == 1.0               # an exact scalar has no F1 in the row
    assert cp.f1_of({"outcome": "wrong", "f1": 0.5}) == 0.5
    assert cp.f1_of({"outcome": "wrong", "f1": None}) == 0.0                 # a wrong scalar
    assert cp.f1_of({"outcome": "abstain_unexpected", "f1": None}) == 0.0
    flagged = {"outcome": "partial_valid", "f1": 0.8}
    assert cp.f1_of(flagged) == 0.8 and cp.f1_of(flagged, strict=True) == 0.0


def test_cost_counts_both_roles_of_the_pipeline_and_the_agent_alike():
    r = cp.run_record(pipe("a", "correct", tin=1000, tout=100), 1.0, 5.0)
    assert r["cost"] == pytest.approx((2000 * 1.0 + 200 * 5.0) / 1e6)
    r = cp.run_record(agent("a", "correct", tin=4000, tout=400), 1.0, 5.0)
    assert r["cost"] == pytest.approx((4000 + 400 * 5.0) / 1e6)


def test_repeats_are_averaged_inside_a_question(tmp_path):
    f1 = write(tmp_path, "r1.json", [pipe("q1", "correct"), pipe("q2", "wrong", f1=0.5)])
    f2 = write(tmp_path, "r2.json", [pipe("q1", "partial_valid", f1=0.8), pipe("q2", "wrong", f1=0.5)])
    s = cp.System("x", [f1, f2])
    assert s.repeats == 2 and s.qids == ["q1", "q2"]
    assert s.v["q1"]["correct"] == 0.5 and s.v["q1"]["flagged"] == 0.5
    assert s.v["q1"]["f1"] == pytest.approx(0.9) and s.v["q1"]["f1_strict"] == pytest.approx(0.5)
    assert s.v["q2"]["silent"] == 1.0


def test_a_repeat_over_other_questions_or_with_duplicates_is_refused(tmp_path):
    a = write(tmp_path, "a.json", [pipe("q1", "correct"), pipe("q2", "correct")])
    b = write(tmp_path, "b.json", [pipe("q1", "correct"), pipe("q3", "correct")])
    with pytest.raises(ValueError, match="different set"):
        cp.System("x", [a, b])
    c = write(tmp_path, "c.json", [pipe("q1", "correct"), pipe("q1", "correct")])
    with pytest.raises(ValueError, match="duplicate"):
        cp.System("x", [c])


def test_bootstrap_is_deterministic_and_the_interval_contains_the_point():
    import random
    ids = [str(i) for i in range(40)]
    vals = {q: (1.0 if int(q) % 4 else 0.0) for q in ids}
    stat = lambda xs: 100 * sum(vals[q] for q in xs) / len(xs)
    a = cp.boot(ids, stat, random.Random(1), 500)
    b = cp.boot(ids, stat, random.Random(1), 500)
    assert a == b and a[1] <= a[0] <= a[2] and a[0] == pytest.approx(75.0)


def _systems(tmp_path, n=30, agent_wrong=10):
    ours = write(tmp_path, "o.json", [pipe(f"q{i}", "correct") for i in range(n)])
    ag = write(tmp_path, "a.json", [agent(f"q{i}", "wrong" if i < agent_wrong else "correct",
                                          f1=0.5 if i < agent_wrong else 1.0) for i in range(n)])
    return cp.System("cost", [ours]), cp.System("plain", [ag])


def test_decision_rule_not_met_when_the_agent_is_worse_and_dearer(tmp_path):
    ours, plain = _systems(tmp_path)
    import random
    text = "\n".join(cp.decision(ours, [plain], ours.qids, random.Random(1)))
    assert "criterion not met" in text
    assert "within 5 points: **no**" in text and "within 1.5 times: **no**" in text


def test_decision_rule_met_when_the_agent_matches_on_f1_and_cost(tmp_path):
    ours = cp.System("cost", [write(tmp_path, "o.json", [pipe(f"q{i}", "correct") for i in range(20)])])
    ag = cp.System("plain", [write(tmp_path, "a.json", [agent(f"q{i}", "correct", tin=1500, tout=100) for i in range(20)])])
    import random
    text = "\n".join(cp.decision(ours, [ag], ours.qids, random.Random(1)))
    assert "criterion met" in text and "reposition" in text
    assert "the agent matches): **yes**" in text and "benchmark and analysis paper" in text


def test_report_end_to_end_with_stratum_table_and_output_file(tmp_path, capsys):
    ours = write(tmp_path, "o.json", [pipe(f"q{i}", "correct", template="D04", stratum="1-2" if i < 4 else "3-10")
                                     for i in range(8)])
    plain = write(tmp_path, "p.json", [agent(f"q{i}", "wrong" if i % 2 else "correct", f1=0.4, template="D04",
                                            stratum="1-2" if i < 4 else "3-10") for i in range(8)])
    py = write(tmp_path, "y.json", [agent(f"q{i}", "no_answer", template="D04", stratum="1-2" if i < 4 else "3-10")
                                   for i in range(8)])
    out = tmp_path / "rep.md"
    rc = cp.main(["--system", f"cost={ours}", "--system", f"plain={plain}", "--system", f"python={py}",
                  "--ours", "cost", "--agents", "plain,python", "--out", str(out)])
    assert rc == 0
    text = out.read_text()
    assert "## D04 by stratum" in text and "| 1-2 |" in text and "| 3-10 |" in text
    assert "better agent arm by F1 on the whole set: **plain**" in text
    assert "cost minus plain" in text and "plain minus python" in text
    assert capsys.readouterr().out.startswith("# CP1 report: 8 questions")


def test_unknown_system_and_mismatched_question_sets_are_refused(tmp_path):
    a = write(tmp_path, "a.json", [pipe("q1", "correct")])
    b = write(tmp_path, "b.json", [agent("q2", "correct")])
    assert cp.main(["--system", f"cost={a}", "--system", f"plain={b}", "--ours", "cost", "--agents", "plain"]) == 2
    assert cp.main(["--system", f"cost={a}", "--ours", "cost", "--agents", "nope"]) == 2


def test_answer_scores_of_the_pipeline_match_the_agent_definition():
    rec = {"answer_type": "ids", "gold": [1, 2, 3, 4]}
    s = answer_scores(Answer("ids", [1, 2, 9], Guarantee.SUBSET), rec)
    assert s["precision"] == pytest.approx(2 / 3, abs=1e-4) and s["recall"] == 0.5 and s["f1"] == pytest.approx(0.5714, abs=1e-4)
    assert answer_scores(Answer("ids", [], Guarantee.SUBSET), rec)["f1"] == 0.0
    assert answer_scores(Answer("abstain", None, Guarantee.NONE, "x"), rec)["f1"] is None
    assert answer_scores(Answer("scalar", 5, Guarantee.EXACT), {"answer_type": "scalar", "gold": 5})["f1"] is None


def test_overall_table_has_a_reliable_column_that_counts_flagged_partial_answers(tmp_path):
    import random
    rows = [pipe("q1", "correct"), pipe("q2", "partial_valid", f1=0.5), pipe("q3", "wrong", f1=0.0), pipe("q4", "abstain_unexpected")]
    sys_ = cp.System("cost", [write(tmp_path, "o.json", rows)])
    table = cp.table_overall([sys_], sys_.qids, random.Random(1))
    assert "reliable (correct + flagged)" in table[0]
    assert table[2].split("|")[7].strip().startswith("50.0%")          # q1 and q2 of four
