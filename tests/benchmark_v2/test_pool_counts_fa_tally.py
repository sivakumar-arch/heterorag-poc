"""Batch 100 (build item 1): the pool-count script and the false-alarm tally script.

Pool counts run on the synthetic oracle (built by ``oracle.build`` and opened by ``union.open_union``, as on the
real machine) with the generator's real D templates at the scaled thresholds of the class D tests (T = 10,
H = 40). The tally runs on hand-made result rows.
"""
import json

from benchmark_v2 import catalogue_d, pool_counts, questions
from benchmark_v2 import fa_tally
from tests.benchmark_v2.test_questions import world  # noqa: F401  (fixture)

T, H = 10, 40


def _write_jsonl(path, records):
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return str(path)


def test_pool_counts_exclude_used_questions_and_add_up(world, tmp_path):
    _, db, db_path = world
    records, _ = questions.build_records(catalogue_d.TEMPLATES, db, seed=11, per_template=4, t_independent=T,
                                         h_cap=H)
    keys, notes = pool_counts.excluded_keys([_write_jsonl(tmp_path / "used.jsonl", records)],
                                            questions.oracle_fingerprint(db))
    assert "OTHER ORACLE" not in notes[0]
    used = {}
    for r in records:
        used.setdefault(r["template"], []).append(r)
    for t in catalogue_d.TEMPLATES:
        c = pool_counts.count(t, questions.Ctx(db, 5, T, H), keys.get(t.id, set()), T, H)
        assert c["pool"] + c["excluded"] == c["passing_check_gold"] <= c["distinct"] <= c["candidates"]
        assert sum(c["pool_by_stratum"].values()) == c["pool"]
        if t.id == "D05":          # one random word per (tag, band): excluded by (tag, band), whatever the word
            assert 0 < c["excluded"] <= len(used[t.id])
        else:                      # every generated question is a candidate with the same parameters
            assert c["excluded"] == len(used[t.id]), (t.id, c)


def test_pool_counts_main_reports_the_oracle_and_a_foreign_fingerprint(world, tmp_path, capsys):
    _, _, db_path = world
    foreign = [{"template": "D04", "params": {"badge": "b1"}, "stratum": "x", "oracle": "0000000000000000"}]
    out = tmp_path / "pools.json"
    rc = pool_counts.main(["--db", str(db_path), "--exclude", _write_jsonl(tmp_path / "f.jsonl", foreign),
                           "--t-independent", str(T), "--h-cap", str(H), "--out", str(out)])
    assert rc == 0
    text = capsys.readouterr().out
    assert "fingerprint" in text and "sha256" in text and "OTHER ORACLE FINGERPRINT" in text
    data = json.loads(out.read_text())
    assert [t["template"] for t in data["templates"]] == ["D01", "D04", "D05", "D07"]
    assert len(data["oracle_sha256"]) == 64


def test_pool_counts_main_rejects_a_missing_oracle(tmp_path):
    assert pool_counts.main(["--db", str(tmp_path / "none.sqlite")]) == 2


def test_d05_key_ignores_the_random_word():
    a = pool_counts.key("D05", {"tag": "r", "word": "model"}, "df 10-40")
    b = pool_counts.key("D05", {"tag": "r", "word": "data", "stratum": "df 10-40"})
    assert a == b and pool_counts.key("D04", {"badge": "x", "stratum": "s1"}) == pool_counts.key("D04", {"badge": "x"}, "s2")


# ---------------------------------------------------------------- false-alarm tally

PLAN = {"steps": [
    {"id": "s1", "op": "retrieve", "store": "graph", "question": "askers of the tag", "output": "user"},
    {"id": "s2", "op": "retrieve", "store": "sql", "question": "these users with reputation at least 5",
     "inputs": ["s1"], "bind": {"input": "s1", "entity": "user"}, "output": "user"}]}
CLEAN = {"s1": 'MATCH (u:User)-[:ASKED]->(:Question)-[:TAGGED_WITH]->(:Tag {name: "r"}) RETURN DISTINCT u.id AS id',
         "s2": "SELECT id FROM users WHERE reputation >= 5 AND id = ANY(%(s1)s)"}
NEUTRAL = dict(CLEAN, s2="SELECT id FROM users WHERE reputation >= 5 AND (id = ANY(%(s1)s) OR TRUE)")


def _row(qid, queries, outcome="correct"):
    return {"question_id": qid, "text": "users who asked about r with reputation at least 5", "outcome": outcome,
            "plan": PLAN, "executed_plan": PLAN, "queries": queries}


def _file(tmp_path, name, meta, rows):
    p = tmp_path / name
    p.write_text(json.dumps({"meta": meta, "rows": rows}))
    return str(p)


def test_attribution_by_translator_model():
    assert fa_tally.attribution({"translator_model": "claude-haiku-5-5"}) == ("hosted", "claude-haiku-5-5")
    assert fa_tally.attribution({"planner_model": "ollama:gpt-oss:20b"}) == ("ladder", "ollama:gpt-oss:20b")
    assert fa_tally.attribution({}) == ("unattributed", None)


def test_tally_counts_correct_rows_and_firings_by_group(tmp_path, capsys):
    hosted = _file(tmp_path, "h.json", {"translator_model": "claude-haiku-5-5"},
                   [_row("A-1", CLEAN), _row("A-2", NEUTRAL), _row("A-3", NEUTRAL, outcome="wrong")])
    ladder = _file(tmp_path, "l.json", {"translator_model": "ollama:gpt-oss:20b"},
                   [_row("B-1", CLEAN), _row("B-2", {})])                       # B-2: no queries, not scorable
    other = _file(tmp_path, "u.json", {}, [_row("C-1", NEUTRAL)])
    assert fa_tally.main([hosted, ladder, other]) == 0
    out = capsys.readouterr().out
    assert "h.json [hosted, claude-haiku-5-5]: correct 2, scored 2; fires: S 1 ['A-2']" in out
    assert "l.json [ladder, ollama:gpt-oss:20b]: correct 2, scored 1; fires: S 0" in out
    assert "u.json [unattributed, model not recorded]: correct 1, scored 1; fires: S 1 ['C-1']" in out
    assert "attributed   correct 4, scored 3; S 1, R7 0, S|R7 1" in out


def test_tally_reports_an_unreadable_file(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{")
    assert fa_tally.main([str(p)]) == 2


# ---------------------------------------------------------------- Batch 101

def test_attribution_by_model_name():
    a = fa_tally.attribution
    assert a({"translator_model": "ollama:gpt-oss:20b"}) == ("ladder", "ollama:gpt-oss:20b")
    assert a({"translator_model": "ollama:mistral-small3.2-nosys"})[0] == "ladder"
    assert a({"translator_model": "ollama:qwen3:30b-instruct"})[0] == "ladder"
    assert a({"translator_model": "ollama:gemma4:31b"}) == ("gemma", "ollama:gemma4:31b")
    assert a({"translator_model": "ollama:granite4.2:30b"})[0] == "granite"
    assert a({"translator_model": "ollama:llama3.1:8b"}) == ("other-local", "ollama:llama3.1:8b")
    assert a({"translator_model": "ollama:qwen3:30b"})[0] == "other-local"     # not the registered rung
    assert a({"translator_model": "gpt-4o"}) == ("unattributed", "gpt-4o")


def test_gemma_and_smoke_are_not_in_the_ladder_or_attributed_total(tmp_path, capsys):
    ladder = _file(tmp_path, "l.json", {"translator_model": "ollama:gpt-oss:20b"}, [_row("B-1", CLEAN)])
    gemma = _file(tmp_path, "g.json", {"translator_model": "ollama:gemma4:31b"}, [_row("G-1", NEUTRAL)])
    smoke = _file(tmp_path, "s.json", {"translator_model": "ollama:llama3.1:8b"},
                  [_row("K-1", CLEAN), _row("K-2", CLEAN)])
    assert fa_tally.main([ladder, gemma, smoke]) == 0
    out = capsys.readouterr().out
    assert "g.json [gemma, ollama:gemma4:31b]: correct 1, scored 1; fires: S 1 ['G-1']" in out
    assert "  ladder       files 1, correct 1, scored 1; S 0, R7 0, S|R7 0" in out
    assert "  gemma        files 1, correct 1, scored 1; S 1, R7 0, S|R7 1" in out
    assert "  other-local  files 1, correct 2, scored 2; S 0, R7 0, S|R7 0" in out
    assert "attributed   correct 1, scored 1; S 0, R7 0, S|R7 0" in out


def test_a_file_given_twice_is_refused(tmp_path, capsys, monkeypatch):
    f = _file(tmp_path, "x.json", {"translator_model": "ollama:gpt-oss:20b"}, [_row("B-1", CLEAN)])
    monkeypatch.chdir(tmp_path)
    assert fa_tally.main([f, "x.json"]) == 2                     # same file, two spellings
    assert "given twice" in capsys.readouterr().err
    copy = tmp_path / "copy.json"
    copy.write_bytes((tmp_path / "x.json").read_bytes())
    assert fa_tally.main([f, str(copy)]) == 2                    # identical content under another name
    assert "identical content" in capsys.readouterr().err
    other = _file(tmp_path, "y.json", {"translator_model": "ollama:gpt-oss:20b"}, [_row("B-2", CLEAN)])
    assert fa_tally.main([f, other]) == 0


def test_pool_counts_refuses_a_test_split_file_by_name_before_opening(world, tmp_path, capsys):
    _, _, db_path = world
    missing = tmp_path / "questions_v2_test.jsonl"               # does not exist: refused before it is opened
    assert pool_counts.main(["--db", str(db_path), "--exclude", str(missing)]) == 2
    assert "names a test split, not opened" in capsys.readouterr().err


def test_pool_counts_stops_on_a_test_split_record(world, tmp_path, capsys):
    _, _, db_path = world
    f = _write_jsonl(tmp_path / "renamed.jsonl", [{"template": "S03", "params": {}, "split": "test"}])
    assert pool_counts.main(["--db", str(db_path), "--exclude", f, "--t-independent", str(T),
                             "--h-cap", str(H)]) == 2
    captured = capsys.readouterr()
    assert "holds a test-split record" in captured.err and "POOL" not in captured.out


def test_pool_counts_docstring_example_excludes_dev_files_only():
    assert "--exclude data/questions_v2_dev*.jsonl" in pool_counts.__doc__
    assert "questions_v2_*.jsonl" not in pool_counts.__doc__
