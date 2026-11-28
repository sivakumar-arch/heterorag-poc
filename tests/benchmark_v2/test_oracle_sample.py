"""Batch 106: the oracle-check sample (``oracle_sample``) and the generator report lines on thresholds."""

from __future__ import annotations

import json

import pytest

from benchmark_v2 import catalogue_fc as fc, oracle_sample as osm, questions
from benchmark_v2.questions import Gold, Report, removed_by_condition, render_report


def rec(qid, template, stratum="all", oracle="fp1", split="dev", **params):
    r = {"question_id": qid, "template": template, "stratum": stratum, "params": params, "oracle": oracle,
         "split": split}
    if template in osm.N1_TEMPLATES:
        r["control"] = "N1"
        r["empty_at"] = "final" if stratum == "N1 final" else "b"
    return r


def fresh_like():
    """A question file shaped like the fresh set: 30 per first-contact shape, N1 40, REPEAT 130."""
    out = []
    for i in range(20):
        out.append(rec(f"H1-{i:03d}", "H1", "over-cap", year=2013 + i % 10, badge=f"b{i}"))
    for i in range(10):
        out.append(rec(f"H1-{20 + i:03d}", "H1", "within-cap", year=(2010, 2011, 2012, 2024)[i % 4], badge=f"w{i}"))
    k = 0
    for yi, y in enumerate((2010, 2011, 2012, 2024)):
        for v in range(3):
            out.append(rec(f"H4-{k:03d}", "H4", year=y, rank=(5, 20, 40)[(v - yi) % 3], variant=v))
            k += 1
    for t in ("H2", "S2"):
        out += [rec(f"{t}-{i:03d}", t, tag=f"t{i}") for i in range(30)]
    out += [rec(f"H3-{i:03d}", "H3", f"rank {(5, 20, 40)[i % 3]:02d}", word=f"w{i}") for i in range(30)]
    out += [rec(f"S1-{i:03d}", "S1", f"rank {(2, 5, 10, 20)[i % 4]:02d}", tag=f"t{i}") for i in range(30)]
    for t in osm.N1_TEMPLATES:
        out += [rec(f"{t}-{i:03d}", t, "N1 final" if i < 10 else "N1 intermediate", tag=f"t{i}") for i in range(20)]
    for t, n in (("D01", 50), ("D05", 50), ("D07", 25), ("D04", 5)):
        out += [rec(f"{t}-{i:03d}", t, f"s{i % 2}", tag=f"t{i}") for i in range(n)]
    return out


def test_the_sample_has_78_questions_split_32_16_30_by_checker():
    chosen, notes = osm.select(fresh_like())
    assert len(chosen) == 78 and notes == []
    by = {}
    for r in chosen:
        by[r["oracle_check"]["checker"]] = by.get(r["oracle_check"]["checker"], 0) + 1
    assert by == {osm.OUTSIDE_2: 32, osm.SEEMA: 16, osm.SIVA: 30}
    shapes = {}
    for r in chosen:
        shapes[r["oracle_check"]["shape"]] = shapes.get(r["oracle_check"]["shape"], 0) + 1
    assert shapes == {"H1": 8, "H2": 8, "H3": 8, "H4": 8, "S1": 8, "S2": 8, "N1": 10,
                      "D01": 5, "D05": 5, "D07": 5, "D04": 5}


def test_h4_takes_two_per_year_and_covers_every_rank_and_variant():
    for seed in range(20):
        chosen, _ = osm.select(fresh_like(), seed)
        h4 = [r for r in chosen if r["template"] == "H4"]
        years = {}
        for r in h4:
            years[r["params"]["year"]] = years.get(r["params"]["year"], 0) + 1
        assert sorted(years.values()) == [2, 2, 2, 2]
        assert {r["params"]["rank"] for r in h4} == {5, 20, 40}
        assert {r["params"]["variant"] for r in h4} == {0, 1, 2}


def test_h4_reports_when_no_choice_covers_every_rank():
    recs = [rec(f"H4-{i:03d}", "H4", year=y, rank=5, variant=i % 3) for i, y in enumerate((2010, 2010, 2011, 2011))]
    recs.append(rec("H4-009", "H4", year=2012, rank=20, variant=0))
    notes = []
    got = osm.select_h4(recs, 8, osm.rng_for(1, "H4"), notes)
    assert any("year 2012 has 1" in n for n in notes)
    assert {r["params"]["rank"] for r in got} == {5, 20}


def test_h1_takes_five_over_cap_years_all_different_and_three_within_cap():
    for seed in range(20):
        chosen, _ = osm.select(fresh_like(), seed)
        h1 = [r for r in chosen if r["template"] == "H1"]
        over = [r for r in h1 if r["stratum"] == "over-cap"]
        assert len(over) == 5 and len({r["params"]["year"] for r in over}) == 5
        assert sum(r["stratum"] == "within-cap" for r in h1) == 3


def test_n1_takes_five_per_cell_and_spreads_over_both_templates():
    for seed in range(20):
        chosen, _ = osm.select(fresh_like(), seed)
        assert len({r["question_id"] for r in chosen}) == len(chosen) == 78        # no question twice
        n1 = [r for r in chosen if r["oracle_check"]["shape"] == "N1"]
        assert sum(r["empty_at"] == "final" for r in n1) == 5
        assert sum(r["empty_at"] != "final" for r in n1) == 5
        assert {r["template"] for r in n1 if r["empty_at"] == "final"} == {"D01E", "D05E"}


def test_ranked_shapes_are_spread_over_their_ranks():
    chosen, _ = osm.select(fresh_like())
    s1 = sorted(r["stratum"] for r in chosen if r["template"] == "S1")
    assert s1 == ["rank 02"] * 2 + ["rank 05"] * 2 + ["rank 10"] * 2 + ["rank 20"] * 2
    h3 = [r["stratum"] for r in chosen if r["template"] == "H3"]
    assert sorted(set(h3)) == ["rank 05", "rank 20", "rank 40"] and max(h3.count(s) for s in set(h3)) == 3


def test_the_selection_is_seeded_and_one_shape_does_not_move_another():
    a, _ = osm.select(fresh_like(), 7)
    b, _ = osm.select(fresh_like(), 7)
    assert [r["question_id"] for r in a] == [r["question_id"] for r in b]
    c, _ = osm.select([r for r in fresh_like() if r["template"] != "H2"], 7)
    keep = lambda rs: [r["question_id"] for r in rs if r["template"] != "H2"]
    assert keep(a) == keep(c)
    pos = lambda rs, t: sorted(int(r["question_id"][-3:]) for r in rs if r["template"] == t)
    assert pos(a, "H2") != pos(a, "S2")          # two shapes of the same form draw from different streams
    d, _ = osm.select(fresh_like(), 8)
    assert [r["question_id"] for r in a] != [r["question_id"] for r in d]


def test_a_short_shape_gives_what_it_has_and_is_not_topped_up():
    recs = [r for r in fresh_like() if not (r["template"] == "D04" and r["question_id"] > "D04-002")]
    chosen, notes = osm.select(recs)
    assert sum(r["template"] == "D04" for r in chosen) == 3
    assert any(n.startswith("D04: 3 of 5") for n in notes)
    assert len(chosen) == 76


def test_the_input_is_refused_for_test_split_mixed_oracles_or_repeated_ids():
    with pytest.raises(osm.Refused):
        osm.select(fresh_like() + [rec("X-001", "H2", split="test")])
    with pytest.raises(osm.Refused):
        osm.select(fresh_like() + [rec("X-001", "H2", oracle="fp2")])
    recs = fresh_like()
    with pytest.raises(osm.Refused):
        osm.select(recs + [recs[0]])


def test_the_cli_writes_the_sample_and_refuses_a_test_file_name(tmp_path, capsys):
    q = tmp_path / "questions_fresh.jsonl"
    q.write_text("".join(json.dumps(r) + "\n" for r in fresh_like()))
    out = tmp_path / "sample.jsonl"
    assert osm.main(["--questions", str(q), "--out", str(out)]) == 0
    lines = [json.loads(x) for x in out.read_text().splitlines()]
    assert len(lines) == 78 and all(r["oracle_check"]["seed"] == osm.SEED for r in lines)
    text = capsys.readouterr().out
    assert "selected 78" in text and "H4: 8" in text
    bad = tmp_path / "questions_test.jsonl"
    bad.write_text(q.read_text())
    assert osm.main(["--questions", str(bad), "--out", str(out)]) == 2


# ---------------------------------------------------------------- generator report (Batch 106)

def test_removed_by_condition_is_the_condition_step_minus_the_answer():
    assert removed_by_condition(fc.H3, Gold([1, 2, 3], steps={"a": 9, "b": 5, "c": 100})) == 2
    assert removed_by_condition(fc.H2, Gold([1, 2], steps={"a": 5})) is None


def test_the_generator_report_counts_one_or_two_removed_and_number_marks():
    r = Report("H3", accepted=4, sizes=[5, 5, 20, 40], removed=[1, 2, 2, 7],
               marks={"rank 05": {"n": 2, 2: 1, 102: 0}, "rank 20": {"n": 2, 2: 0, 102: 1}})
    text = render_report([r], 4)
    assert "threshold removes 1 item: 1, 2 items: 2, either: 3 of 4" in text
    assert "rank 05: NUMBER = 2: 1 of 2, NUMBER = 102: 0 of 2" in text
    assert "rank 20: NUMBER = 2: 0 of 2, NUMBER = 102: 1 of 2" in text


def test_generate_template_fills_the_marks_and_the_removed_counts(monkeypatch):
    t = questions.Template(**{**fc.H3.__dict__})
    picks = [({"stratum": "rank 05", "word": "w1", "number": 2, "rank": 5}, Gold([1], steps={"a": 3, "b": 2, "c": 9})),
             ({"stratum": "rank 20", "word": "w2", "number": 102, "rank": 20}, Gold([1, 2], steps={"a": 5, "b": 9, "c": 9}))]
    monkeypatch.setattr(questions, "check_gold", lambda *a, **k: None)
    t = questions.Template(**{**t.__dict__, "sample": lambda c, n: [p for p, _ in picks],
                              "gold": lambda c, p: dict((x["word"], g) for x, g in picks)[p["word"]]})

    class C:
        notes = questions.Counter()
        t_independent, h_cap = 200, 10000

        def validator_caps(self):
            return None
    _, rep = questions.generate_template(t, C(), 2)
    assert sorted(rep.removed) == [1, 7]
    assert rep.marks == {"rank 05": {"n": 1, 2: 1, 102: 0}, "rank 20": {"n": 1, 2: 0, 102: 1}}


def test_pool_counts_report_thresholds_that_remove_one_or_two_items(monkeypatch):
    from benchmark_v2 import pool_counts
    monkeypatch.setattr(pool_counts, "check_gold", lambda *a, **k: None)
    from tests.benchmark_v2.test_fc_105 import T, H
    golds = {"x": Gold([1, 2], steps={"a": 3}), "y": Gold([1], steps={"a": 3}), "z": Gold([1], steps={"a": 9})}
    t = questions.Template(**{**fc.S1.__dict__, "sample": lambda c, n: [{"stratum": "rank 02", "tag": k, "number": 1}
                                                                       for k in golds],
                              "gold": lambda c, p: golds[p["tag"]]})

    class C:
        notes = questions.Counter()
        t_independent, h_cap = T, H

        def validator_caps(self):
            return {}
    r = pool_counts.count(t, C(), set(), T, H)
    assert r["removes_one_or_two"] == {"1": 1, "2": 1, "of": 3}


def test_the_extra_question_goes_to_strata_in_a_seeded_order_not_name_order():
    """Batch 107 (decision 4): D07-like, 5 over 3 strata; the strata that get 2 vary with the seed."""
    recs = [rec(f"D07-{i:03d}", "D07", ("a", "b", "c")[i % 3], tag=f"t{i}") for i in range(30)]
    singles = set()
    for seed in range(30):
        orders: dict = {}
        chosen, _ = osm.select(recs, seed, orders)
        got = [r["stratum"] for r in chosen if r["template"] == "D07"]
        counts = {s: got.count(s) for s in "abc"}
        assert sorted(counts.values()) == [1, 2, 2]
        single = next(s for s, k in counts.items() if k == 1)
        assert orders["D07"][-1] == single                    # the last in the seeded order gets one
        singles.add(single)
    assert singles == {"a", "b", "c"}
