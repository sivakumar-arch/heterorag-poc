import json

from benchmark_v2 import silent_rows as sr

GOLD = [{"id": "s1", "op": "retrieve", "store": "graph"},
        {"id": "s2", "op": "retrieve", "store": "sql", "bind": {"input": "s1", "entity": "user"}},
        {"id": "s3", "op": "intersect", "inputs": ["s1", "s2"]}]
RAN = {"steps": [{"id": "s1", "op": "retrieve", "store": "sql", "question": "users with reputation over 5"}]}


def test_plan_line_shows_store_bind_and_inputs():
    assert sr.plan_line(GOLD) == "s1:graph  s2:sql[bind s1]  s3:intersect(s1,s2)"
    assert sr.plan_line(None) == "(no plan)"


def test_only_silent_rows_are_printed_with_both_plans_and_the_queries(tmp_path, capsys):
    rows = [{"question_id": "Q1", "template": "D01", "outcome": "wrong", "plan_equal": False, "text": "who?",
             "gold_plan": GOLD, "plan": RAN, "queries": {"s1": "SELECT  id\nFROM users"}, "got": [1, 2]},
            {"question_id": "Q2", "template": "D01", "outcome": "correct", "plan_equal": True, "text": "ok",
             "gold_plan": GOLD, "plan": RAN, "queries": {}}]
    f = tmp_path / "r.json"
    f.write_text(json.dumps({"rows": rows}))
    assert sr.main([str(f), "--width", "30"]) == 0
    out = capsys.readouterr().out
    assert "Q1" in out and "Q2" not in out and "gold: s1:graph" in out and "ran:  s1:sql" in out
    assert "query: SELECT id FROM users" in out and "1 silent answers shown" in out
    assert sr.main([str(tmp_path / "none.json")]) == 2
