import json

from benchmark_v2.risk_coverage import main, points


def rows(**counts):
    out = []
    for oc, k in counts.items():
        out += [{"question_id": f"{oc}{i}", "outcome": oc} for i in range(k)]
    return out


def test_points_for_a_system_with_labels():
    p = points(rows(correct=6, wrong=1, partial_valid=2, partial_violated=1, abstain_unexpected=10))
    assert p["n"] == 20
    assert p["exact only"] == {"answered": 7, "silent": 1, "correct": 6, "flagged": 0}
    assert p["all answers"] == {"answered": 10, "silent": 2, "correct": 6, "flagged": 2}


def test_points_for_a_system_without_labels_coincide():
    p = points(rows(correct=3, wrong=5, no_answer=2))
    assert p["exact only"] == p["all answers"] == {"answered": 8, "silent": 5, "correct": 3, "flagged": 0}


def test_command_line_prints_both_points_and_handles_no_answers(tmp_path, capsys):
    f, g = tmp_path / "a.json", tmp_path / "b.json"
    f.write_text(json.dumps({"rows": rows(correct=4, wrong=1, abstain_unexpected=5)}))
    g.write_text(json.dumps({"rows": rows(no_answer=3)}))
    assert main([str(f), str(g)]) == 0
    out = capsys.readouterr().out
    assert "coverage 50.0%" in out and "silent among answered 20.0%" in out
    assert "n/a (no answers)" in out
