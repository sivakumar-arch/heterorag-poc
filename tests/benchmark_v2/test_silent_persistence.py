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


def test_clustered_interval_resamples_questions_and_is_wider_than_a_pair_binomial(tmp_path):
    # 3 runs, 6 questions: Q0 and Q1 silent in every run, Q2 silent once, the rest correct
    outs = [["wrong", "wrong", "wrong", "correct", "correct", "correct"],
            ["wrong", "wrong", "correct", "correct", "correct", "correct"],
            ["wrong", "wrong", "correct", "correct", "correct", "correct"]]
    runs = load([_write(tmp_path, f"r{i}.json", o) for i, o in enumerate(outs)])
    pq = sp.per_question(runs)
    assert pq["Q0"] == (6, 0, 0) and pq["Q2"] == (2, 2, 2) and pq["Q3"] == (0, 0, 0)
    m, lo, hi, nq = sp.clustered(runs, 1)
    assert nq == 3 and abs(m - 100.0 * 2 / 14) < 1e-9          # 2 of 14 ordered pairs became non-silent
    assert lo <= m <= hi and lo == 0.0 and hi > 25.0            # whole questions are resampled: the interval is wide
    assert sp.clustered(runs, 2)[0] == m                        # here every conversion is to correct


def test_clustered_with_no_silent_errors_is_nan(tmp_path):
    runs = load([_write(tmp_path, "a.json", ["correct"]), _write(tmp_path, "b.json", ["correct"])])
    assert sp.clustered(runs, 1)[3] == 0
