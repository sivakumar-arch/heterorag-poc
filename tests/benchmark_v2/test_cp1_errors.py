"""Reading the failures of a pipeline system across repeats."""
import json

from benchmark_v2 import cp1_errors as ce


def row(qid, outcome, template="D04", stratum="1-2", eq=True, guarantee="exact"):
    return {"question_id": qid, "template": template, "stratum": stratum, "outcome": outcome,
            "plan_equal": eq, "guarantee": guarantee}


def write(tmp_path, name, rows):
    p = tmp_path / name
    p.write_text(json.dumps({"meta": {}, "rows": rows}))
    return str(p)


def test_systematic_and_intermittent_failures_are_told_apart(tmp_path):
    f1 = write(tmp_path, "1.json", [row("a", "wrong"), row("b", "wrong"), row("c", "correct"), row("d", "partial_valid")])
    f2 = write(tmp_path, "2.json", [row("a", "wrong"), row("b", "correct"), row("c", "correct"), row("d", "partial_valid")])
    text = ce.render(ce.load([f1, f2]))
    assert "a            D04" in text and "WW" in text and "WC" in text and "PP" in text
    assert "c            D04" not in text                                   # always correct: not listed
    assert "questions with a silent error (W or V) in at least one repeat: 2" in text
    assert "in every repeat (systematic): 1" in text and "in some repeats (intermittent): 1" in text
    assert "P=2, W=3" in text and "C=" not in text.split("by kind:")[1].split("\n")[0]


def test_a_violated_label_counts_as_silent_and_is_shown_separately_from_wrong(tmp_path):
    f = write(tmp_path, "1.json", [row("a", "partial_violated", guarantee="subset", eq=False)])
    text = ce.render(ce.load([f]))
    assert "V" in text and "x" in text and "subset" in text and "silent error (W or V) in at least one repeat: 1" in text
