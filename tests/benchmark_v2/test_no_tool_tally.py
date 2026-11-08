import json

from benchmark_v2.no_tool_tally import bucket, calls_made, classify, main, tally


def t(text, tokens=0):
    return {"tool": "_text", "text": text, "reply_tokens": tokens}


def test_categories_in_order():
    assert classify(t("", 400)) == "hidden"
    assert classify(t("", 3)) == "empty"
    assert classify(None) == "empty"
    assert classify(t('Here is it. final_answer: {"kind": "set"}')) == "final_marker"
    assert classify(t("```sql\nSELECT id FROM users\n```")) == "tool_as_text"
    assert classify(t('{"query": "MATCH (u:User) RETURN u"}')) == "tool_as_text"
    assert classify(t("I looked at the services and the answer is clear.")) == "prose"


def test_a_marker_wins_over_a_query_in_the_same_text():
    assert classify(t("SELECT 1 then final_answer: {}")) == "final_marker"


def test_prose_that_merely_contains_es_is_not_a_call():
    assert classify(t("The model does (it) well, goes on.")) == "prose"


def row(qid, stop, entries, tpl="D01"):
    return {"question_id": qid, "stop": stop, "template": tpl, "transcript": entries}


ROWS = [
    row("A", "no_tool", [t("hello"), t("still words")]),
    row("B", "no_tool", [{"tool": "sql", "input": "x"}, {"tool": "sql", "input": "y"}, t("", 500), t("", 500)]),
    row("C", "answered", [{"tool": "final_answer"}]),
    row("D", "no_tool", [t("final_answer: {number: 3}")]),
]


def test_calls_buckets_and_tally():
    assert calls_made(ROWS[1]) == 2 and bucket(0) == "none" and bucket(2) == "1 to 3" and bucket(4) == "4 or more"
    total, by_calls, recs = tally(ROWS)
    assert total == {"prose": 1, "hidden": 1, "final_marker": 1}
    assert by_calls["1 to 3"]["hidden"] == 1 and by_calls["none"]["prose"] == 1
    assert [r["question_id"] for r in recs] == ["A", "B", "D"]


def test_command_line(tmp_path, capsys):
    f = tmp_path / "a.json"
    f.write_text(json.dumps({"rows": ROWS}))
    assert main([str(f), "--show", "prose"]) == 0
    out = capsys.readouterr().out
    assert "3 ended no_tool" in out and "still words" in out
