import json

from benchmark_v2.text_answer_sensitivity import main, reread, tolerant_answer

PROSE_SET = 'The only user is 77222.\n\nfinal_answer:\n{\n  "identifiers": {77222}\n}'


def test_a_set_written_with_braces_and_an_invented_key_is_read():
    r = tolerant_answer(PROSE_SET, "set")
    assert r.kind == "set" and r.values == ["77222"]


def test_strings_and_numbers_are_values_and_keys_are_dropped():
    r = tolerant_answer('final_answer: {"kind": "set", "values": ["r", "anova", 12]}', "set")
    assert r.values == ["r", "anova", "12"]


def test_scalar_needs_exactly_one_number_and_prose_without_the_marker_is_not_read():
    assert tolerant_answer("final answer: {number: 17}", "scalar").number == 17.0
    assert tolerant_answer("final_answer: {number: 17, other: 3}", "scalar") is None
    assert tolerant_answer("The answer is 17.", "scalar") is None
    assert tolerant_answer("final_answer: no braces here", "set") is None


def test_an_abstention_is_read_as_one():
    assert tolerant_answer('final_answer: {"kind": "abstain", "reason": "x"}', "set").kind == "abstain"


def row(qid, text, outcome="no_answer", stop="no_tool"):
    return {"question_id": qid, "outcome": outcome, "stop": stop,
            "transcript": [{"tool": "_text", "text": text}, {"tool": "_text", "text": text}]}


QUESTIONS = {"A": {"question_id": "A", "answer_type": "set", "gold": [77222]},
             "B": {"question_id": "B", "answer_type": "set", "gold": [1, 2]},
             "C": {"question_id": "C", "answer_type": "scalar", "gold": 17}}


def test_reread_scores_with_the_scorer_of_the_run():
    rows = [row("A", PROSE_SET), row("B", PROSE_SET), row("C", "no marker here")]
    got = reread(rows, QUESTIONS)
    assert got["A"]["outcome"] == "correct" and got["B"]["outcome"] == "wrong" and "C" not in got


def test_command_line_prints_registered_and_tolerant_readings(tmp_path, capsys):
    f, q, p = tmp_path / "a.json", tmp_path / "q.jsonl", tmp_path / "p.json"
    f.write_text(json.dumps({"rows": [row("A", PROSE_SET), row("B", PROSE_SET),
                                      {"question_id": "C", "outcome": "correct", "stop": "answered"}]}))
    q.write_text("\n".join(json.dumps(v) for v in QUESTIONS.values()))
    p.write_text(json.dumps({"rows": [{"question_id": k, "outcome": "correct"} for k in "ABC"]}))
    assert main(["--file", str(f), "--questions", str(q), "--pipeline", str(p), "--show"]) == 0
    out = capsys.readouterr().out
    assert "no_tool runs: 2; last text reply readable under the tolerant rule: 2; of those: would be correct 1, wrong 1" in out
    assert "tolerant reading: correct 66.7% " in out and "pipeline minus this arm, tolerant: +33.3" in out
