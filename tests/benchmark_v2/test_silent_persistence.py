import json

from benchmark_v2 import silent_persistence as sp
from benchmark_v2.repeat_summary import load


def _write(tmp_path, name, outcomes):
    p = tmp_path / name
    p.write_text(json.dumps({"rows": [{"question_id": f"Q{i}", "outcome": o} for i, o in enumerate(outcomes)]}))
    return str(p)


def test_transitions_and_pooling(tmp_path):
    a = _write(tmp_path, "a.json", ["wrong", "wrong", "partial_violated", "correct"])
    b = _write(tmp_path, "b.json", ["wrong", "correct", "abstain_unexpected", "wrong"])
    runs = load([a, b])
    tr = sp.transitions(runs)
    assert tr[(0, 1)] == {"silent": 3, "still_silent": 1, "correct": 1, "other": 1}
    assert tr[(1, 0)] == {"silent": 2, "still_silent": 1, "correct": 1, "other": 0}
    assert sp.pooled(tr) == {"silent": 5, "still_silent": 2, "correct": 2, "other": 1}


def test_render_and_no_silent(tmp_path):
    a = _write(tmp_path, "a.json", ["correct", "correct"])
    b = _write(tmp_path, "b.json", ["correct", "wrong"])
    out = sp.render([a, b], load([a, b]))
    assert "no silent errors in the first run" in out and "pooled" in out


def test_needs_two_files(tmp_path, capsys):
    a = _write(tmp_path, "a.json", ["correct"])
    assert sp.main([a]) == 2
