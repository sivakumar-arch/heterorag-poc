import json

from benchmark_v2 import agreement_replay as ar
from benchmark_v2.repeat_summary import load


def _write(tmp_path, name, outcomes):
    p = tmp_path / name
    p.write_text(json.dumps({"rows": [{"question_id": f"Q{i}", "outcome": o} for i, o in enumerate(outcomes)]}))
    return str(p)


def test_pair_counts(tmp_path):
    a = _write(tmp_path, "a.json", ["correct", "correct", "wrong", "wrong", "abstain_unexpected", "partial_valid"])
    b = _write(tmp_path, "b.json", ["correct", "wrong", "wrong", "correct", "correct", "partial_valid"])
    r = load([a, b])
    c = ar.pair_counts(r[0], r[1])
    assert c == {"n": 6, "correct": 1, "silent_worst": 1, "abstained_or_disagree": 4}


def test_render_and_needs_two(tmp_path, capsys):
    a = _write(tmp_path, "a.json", ["correct", "wrong"])
    b = _write(tmp_path, "b.json", ["correct", "correct"])
    assert "pair 1+2" in ar.render([a, b], load([a, b]))
    assert ar.main([a]) == 2
