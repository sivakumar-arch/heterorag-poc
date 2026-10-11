"""
The benchmark's gold-plan annotation and the system's plan are one schema (decision D18).

The benchmark package does not import ``heterorag``, so the schema exists twice. These tests
keep the two copies identical, and check that every annotated gold plan is a valid plan for
the system's own validator.
"""
import dataclasses

import pytest

from benchmark_v2 import catalogue_sp, questions
from heterorag.federation import Op, Plan
from heterorag.federation.guarantee import Kind
from heterorag.federation.plan import STEP_FIELDS, Mode, Store


def test_field_names_and_order_are_identical():
    assert tuple(f.name for f in dataclasses.fields(questions.Step)) == STEP_FIELDS
    assert tuple(f.name for f in dataclasses.fields(questions.Bind)) == ("input", "entity", "param")


def test_vocabularies_match():
    used_ops = {s.op for t in catalogue_sp.TEMPLATES for s in t.plan}
    assert used_ops <= {o.value for o in Op}
    assert {s.store for t in catalogue_sp.TEMPLATES for s in t.plan if s.store} <= {x.value for x in Store}
    assert {s.mode for t in catalogue_sp.TEMPLATES for s in t.plan if s.mode} <= {m.value for m in Mode}
    assert {s.kind for t in catalogue_sp.TEMPLATES for s in t.plan} <= {k.value for k in Kind}


@pytest.mark.parametrize("template", catalogue_sp.TEMPLATES, ids=lambda t: t.id)
def test_every_gold_plan_is_valid_for_the_system(template):
    k = 5
    plan_json = {"steps": [questions.step_record(s, k) for s in template.plan]}
    plan = Plan.from_dict(plan_json)
    assert plan.validate() == []
    assert plan.to_dict() == plan_json or [s.id for s in plan.steps] == [s["id"] for s in plan_json["steps"]]
    # the answer type of the template agrees with the kind of the final step
    sink = plan.sink()
    expected = {"ids": Kind.SET, "ranked": Kind.RANKED, "scalar": Kind.SCALAR}[template.answer_type]
    assert sink.kind is expected
    assert sink.output == ("number" if expected is Kind.SCALAR else template.entity)
