import json

import pytest

from benchmark_v2 import repeat_summary as rs


def _write(tmp_path, name, outcomes):
    p = tmp_path / name
    p.write_text(json.dumps({"rows": [{"question_id": f"Q{i}", "outcome": o} for i, o in enumerate(outcomes)]}))
    return str(p)


def test_group_maps_outcomes():
    assert rs.group("correct") == "correct"
    assert rs.group("wrong") == "silent"
    assert rs.group("partial_violated") == "silent"
    assert rs.group("partial_valid") == "flagged"
    assert rs.group("abstain_unexpected") == "abstained"
    assert rs.group("no_answer") == "abstained"


def test_stability_counts(tmp_path):
    f1 = _write(tmp_path, "a.json", ["correct", "correct", "wrong", "abstain_unexpected"])
    f2 = _write(tmp_path, "b.json", ["correct", "wrong", "wrong", "abstain_unexpected"])
    f3 = _write(tmp_path, "c.json", ["correct", "correct", "wrong", "correct"])
    s = rs.stability(rs.load([f1, f2, f3]))
    assert s == {"questions": 4, "correct_all": 1, "correct_some": 2, "correct_none": 1,
                 "silent_any": 2, "changed_category": 2}


def test_pooled_equals_mean_of_repeat_rates(tmp_path):
    import random
    f1 = _write(tmp_path, "a.json", ["correct", "correct", "wrong", "wrong"])
    f2 = _write(tmp_path, "b.json", ["correct", "wrong", "wrong", "wrong"])
    m, lo, hi = rs.pooled(rs.load([f1, f2]), "correct", random.Random(1))
    assert m == pytest.approx(37.5)
    assert lo <= m <= hi


def test_mismatched_questions_rejected(tmp_path):
    f1 = _write(tmp_path, "a.json", ["correct", "wrong"])
    f2 = _write(tmp_path, "b.json", ["correct"])
    with pytest.raises(ValueError):
        rs.load([f1, f2])


def test_render_runs(tmp_path):
    f1 = _write(tmp_path, "a.json", ["correct", "wrong"])
    f2 = _write(tmp_path, "b.json", ["correct", "correct"])
    out = rs.render([f1, f2], rs.load([f1, f2]))
    assert "2 repeats, 2 questions" in out and "pooled" in out
