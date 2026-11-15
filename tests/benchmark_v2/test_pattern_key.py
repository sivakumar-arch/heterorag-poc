"""The mechanical pattern key needs no reader; it uses sizes, plan structure and the final answer."""
import json

from benchmark_v2.pattern_key import pattern_key, patterns, render, wording_variant


def _row(**kw):
    base = {"question_id": "CP1B-D05-003", "template": "D05", "outcome": "wrong", "plan_equal": True,
            "executed_plan": {"steps": [
                {"id": "s1", "op": "retrieve", "store": "graph", "output": "question"},
                {"id": "s2", "op": "retrieve", "store": "doc", "output": "question"},
                {"id": "s3", "op": "intersect", "output": "question", "inputs": ["s1", "s2"]}]},
            "step_records": [{"step": "s1", "value_size": 0}, {"step": "s2", "value_size": 30}],
            "answer_digest": {"type": "ids", "size": 0}, "precision": 0.0, "recall": 0.0}
    base.update(kw)
    return base


Q = {"intermediates": {"graph": 12, "doc": 30}}


def test_empty_first_step_is_the_key_for_a_reversed_relationship():
    assert pattern_key(_row(), Q) == ("D05", "step1", "empty")


def test_equal_sizes_fall_through_to_the_final_answer():
    r = _row(step_records=[{"step": "s1", "value_size": 12}, {"step": "s2", "value_size": 30}],
             answer_digest={"type": "ids", "size": 9}, precision=0.5, recall=1.0)
    assert pattern_key(r, Q) == ("D05", "final", "superset")


def test_a_bound_step_is_not_compared_with_the_independent_size():
    plan = {"steps": [{"id": "s1", "op": "retrieve", "store": "graph", "output": "user"},
                      {"id": "s2", "op": "retrieve", "store": "sql", "output": "user", "bind": {"name": "s1"}}]}
    r = _row(executed_plan=plan, step_records=[{"step": "s1", "value_size": 12}, {"step": "s2", "value_size": 3}],
             answer_digest={"type": "ids", "size": 3}, precision=1.0, recall=0.5)
    assert pattern_key(r, Q) == ("D05", "final", "subset")


def test_a_different_plan_uses_the_structure_and_the_final_direction():
    r = _row(plan_equal=False, answer_digest={"type": "ids", "size": 40}, precision=0.2, recall=1.0)
    k = pattern_key(r, Q)
    assert k[0] == "D05" and k[1].startswith("plan:retrieve:graph:question:free|") and k[2] == "superset"


def test_wording_variant_is_derived_from_the_id_and_not_part_of_the_key():
    assert [wording_variant(f"CP1B-D05-{n:03d}") for n in (1, 2, 3, 4)] == [0, 1, 2, 0]
    assert pattern_key(_row(question_id="CP1B-D05-005"), Q) == pattern_key(_row(question_id="CP1B-D05-003"), Q)


def test_patterns_count_across_files_and_render(tmp_path):
    rows = [_row(question_id="CP1B-D05-003"), _row(question_id="CP1B-D05-005"),
            _row(question_id="CP1B-D05-007", outcome="correct")]
    f = tmp_path / "t0_a_cost_1.json"
    f.write_text(json.dumps({"rows": rows}))
    pats = patterns([str(f)], {})
    assert len(pats) == 1 and sum(len(v) for v in pats.values()) == 2
    assert "1 patterns" in render(pats)
