import json

from benchmark_v2.no_tool_tally import bucket, calls_made, classify, main, first_to_last, repeated, tally, token_excess


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


def rec(text, tokens, qid="X"):
    return {"question_id": qid, "category": "prose", "text": text, "raw_chars": len(text), "tokens": tokens}


def test_repeated_replies_are_listed_only_when_they_recur():
    r = [rec("same", 5), rec("same", 5), rec("other", 5)]
    assert repeated(r) == [("same", 2)]


def test_token_excess_flags_a_reply_that_generated_far_more_than_it_shows():
    normal = [rec("a" * 100, 25, f"n{i}") for i in range(8)]          # 0.25 tokens per character
    dropped = rec("Here is the corrected query:" + " " * 2, 400, "d")  # short text, 400 tokens
    cut = rec("b" * 600, 900, "c")                                    # at the transcript limit: not usable
    use, med, flagged = token_excess(normal + [dropped, cut])
    assert use == 9 and abs(med - 0.25) < 1e-9 and [x["question_id"] for x in flagged] == ["d"]


def test_token_excess_needs_enough_replies():
    use, med, flagged = token_excess([rec("a" * 50, 12)])
    assert use == 1 and flagged == []


def test_first_to_last_crosses_the_two_replies_of_a_no_tool_run():
    rows = [
        row("A", "no_tool", [t("", 300), t("I currently have no tools")]),
        row("B", "no_tool", [t("", 300), t("I currently have no tools")]),
        row("C", "no_tool", [t("words"), t("final_answer: {}")]),
        row("D", "answered", [t("words"), t("more")]),
    ]
    c = first_to_last(rows)
    assert c[("hidden", "prose")] == 2 and c[("prose", "final_marker")] == 1 and sum(c.values()) == 3
