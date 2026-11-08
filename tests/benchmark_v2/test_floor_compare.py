import json

import pytest

from benchmark_v2.floor_compare import hidden_list, hidden_replies, main, mcnemar_exact, verdict_l1


def row(qid, outcome, transcript=None):
    return {"question_id": qid, "template": "D01", "stratum": "s", "outcome": outcome,
            "usage": {"prompt_tokens": 10, "reply_tokens": 5}, "transcript": transcript or []}


def write(path, outcomes, transcripts=None):
    rows = [row(f"Q{i:03d}", o, (transcripts or {}).get(i)) for i, o in enumerate(outcomes)]
    path.write_text(json.dumps({"meta": {"complete": True}, "rows": rows}))
    return str(path)


def test_mcnemar_exact_known_values():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(10, 0) == pytest.approx(2 / 1024)
    assert mcnemar_exact(0, 10) == pytest.approx(2 / 1024)


def test_hidden_replies_counts_last_and_anywhere():
    empty_big = {"tool": "_text", "text": "", "reply_tokens": 90}
    empty_small = {"tool": "_text", "text": "", "reply_tokens": 3}
    said = {"tool": "_text", "text": "I cannot", "reply_tokens": 90}
    call = {"tool": "es", "input": "{}"}
    rows = [row("a", "no_answer", [call, empty_big]), row("b", "no_answer", [empty_big, call, said]),
            row("c", "no_answer", [empty_small]), row("d", "no_answer", [])]
    assert hidden_replies(rows) == (1, 2)


def test_l1_verdicts():
    assert verdict_l1(40, 102, (30.0, 10.0, 50.0))[0].startswith("pipeline 39.2% correct is below 50%")
    assert verdict_l1(91, 102, (30.0, 10.0, 50.0))[-1] == "L1: MET"
    assert verdict_l1(91, 102, (12.0, 2.0, 22.0))[-1] == "L1: NOT MET"
    assert verdict_l1(91, 102, (20.0, -1.0, 40.0))[-1] == "L1: NOT MET"
    assert verdict_l1(80, 102, (30.0, 10.0, 50.0))[-1] == "L1: NOT MET"       # 78.4 percent is under 80


def test_command_line_end_to_end(tmp_path, capsys):
    n = 20
    pipe = write(tmp_path / "p.json", ["correct"] * 18 + ["wrong", "abstain_unexpected"])
    py = write(tmp_path / "y.json", ["correct"] * 4 + ["no_answer"] * 14 + ["wrong"] * 2,
               {5: [{"tool": "_text", "text": "", "reply_tokens": 90}]})
    pl = write(tmp_path / "l.json", ["correct"] * 6 + ["no_answer"] * 12 + ["wrong"] * 2)
    assert main(["--pipeline", pipe, "--python", py, "--plain", pl]) == 0
    out = capsys.readouterr().out
    assert "20 questions" in out and "Better agent arm (more correct answers): plain" in out
    assert "L1: MET" in out and "python: last reply 1/20 = 5.0% (below the 10% trigger)" in out


def test_hidden_list_says_where_in_the_run_the_reply_was():
    empty = {"tool": "_text", "text": "", "reply_tokens": 90}
    call = {"tool": "es", "input": "{}"}
    rows = [row("a", "no_answer", [empty, empty]), row("b", "no_answer", [call, call, empty]),
            row("c", "no_answer", [empty, call])]
    got = hidden_list(rows)
    assert [(g["question_id"], g["calls_before"], g["replies"]) for g in got] == [("a", 0, 2), ("b", 2, 1)]


def test_list_hidden_flag_prints_the_sample(tmp_path, capsys):
    pipe = write(tmp_path / "p.json", ["correct"] * 4)
    t = {"tool": "_text", "text": "", "reply_tokens": 50}
    py = write(tmp_path / "y.json", ["no_answer"] * 4, {1: [t, t]})
    pl = write(tmp_path / "l.json", ["no_answer"] * 4)
    assert main(["--pipeline", pipe, "--python", py, "--plain", pl, "--list-hidden"]) == 0
    out = capsys.readouterr().out
    assert "python arm: 1 questions, 1 with no tool call before it" in out and "Q001  D01  reply_tokens=50" in out
