"""Planner, translator and pipeline with a scripted model (no API, no live stores)."""
import json
import re

import pytest

from heterorag.federation import Executor, FetchResult, Guarantee, Op, Plan, Store
from heterorag.federation import planner as planner_mod
from heterorag.federation import translator as translator_mod
from heterorag.federation.llm_config import DEFAULT_MODEL, MeteredProvider, RoleModels, Usage
from heterorag.federation.pipeline import Federation
from heterorag.federation.planner import Planner, extract_json
from heterorag.federation.translator import Translator, clean_reply
from heterorag.llm_provider import LLMProvider, LLMResponse
from tests.federation.fake_descriptors import DESCRIPTORS, catalogue


class Scripted(LLMProvider):
    """Replies from a list (in order) or a function of the prompt. Records the prompts."""
    model = "scripted"

    def __init__(self, replies):
        self.replies, self.prompts = replies, []

    def complete(self, prompt, *, max_tokens=512, temperature=0.0):
        self.prompts.append(prompt)
        r = self.replies(prompt) if callable(self.replies) else self.replies[min(len(self.prompts) - 1, len(self.replies) - 1)]
        return LLMResponse(text=r, prompt_tokens=100, reply_tokens=20, model="scripted")


def metered(replies):
    inner = Scripted(replies)
    return MeteredProvider(inner, "test"), inner


P01_PLAN = {"steps": [
    {"id": "s1", "op": "retrieve", "store": "graph", "question": "questions tagged python", "output": "question"},
    {"id": "s2", "op": "retrieve", "store": "doc", "mode": "filter", "question": "questions with regex in the title", "output": "question"},
    {"id": "s3", "op": "intersect", "inputs": ["s1", "s2"], "output": "question"}]}


# ------------------------------------------------------------------ configuration and metering

def test_role_models_default_to_the_cheapest_and_ignore_the_legacy_variable():
    assert RoleModels.from_env({}) == RoleModels(DEFAULT_MODEL, DEFAULT_MODEL)
    assert DEFAULT_MODEL == "claude-haiku-4-5-20251001"
    legacy = RoleModels.from_env({"HETERORAG_LLM_MODEL": "claude-sonnet-5-5"})
    assert legacy.planner == legacy.translator == DEFAULT_MODEL
    r = RoleModels.from_env({"HETERORAG_PLANNER_MODEL": "big", "HETERORAG_TRANSLATOR_MODEL": "small"})
    assert (r.planner, r.translator) == ("big", "small")


def test_metered_provider_counts_calls_and_tokens():
    m, _ = metered(["x"])
    _, u = m.call("p")
    m.complete("q")
    assert u.calls == 1 and u.prompt_tokens == 100 and m.total.calls == 2 and m.total.reply_tokens == 40
    assert m.model == "scripted"
    assert (Usage(1, 2, 3, 4, 0.5) + Usage(1, 1, 1, 1, 0.5)).to_dict()["calls"] == 2


# ------------------------------------------------------------------ planner

def test_prompt_examples_are_valid_plans_for_the_system():
    block = planner_mod._EXAMPLES
    dec = json.JSONDecoder()
    plans = []
    for part in block.split("EXAMPLE ")[1:]:
        text = part[part.index("\n{") + 1:]
        obj, _ = dec.raw_decode(text)
        if "steps" in obj:
            plans.append(obj)
    assert len(plans) == 3
    for d in plans:
        assert Plan.from_dict(d).validate() == []
    assert '{"abstain": "out_of_scope"}' in block


def test_planner_parses_fenced_json_and_validates():
    m, _ = metered(["```json\n" + json.dumps(P01_PLAN) + "\n```"])
    r = Planner(m, DESCRIPTORS).plan("which python questions mention regex in the title?")
    assert r.ok and r.repairs == 0 and r.usage.calls == 1
    assert [s.op for s in r.plan.steps] == [Op.RETRIEVE, Op.RETRIEVE, Op.INTERSECT]


def test_planner_abstains_out_of_scope():
    m, _ = metered(['{"abstain": "out_of_scope"}'])
    r = Planner(m, DESCRIPTORS).plan("average reputation of tag python experts")
    assert not r.ok and r.abstain == "out_of_scope" and r.usage.calls == 1


def test_planner_repairs_once_with_the_error_messages():
    bad = {"steps": [{"id": "s1", "op": "retrieve", "store": "sql", "output": "user"}]}      # no question
    m, inner = metered([json.dumps(bad), json.dumps(P01_PLAN)])
    r = Planner(m, DESCRIPTORS).plan("q")
    assert r.ok and r.repairs == 1 and r.usage.calls == 2
    assert "IT WAS REJECTED" in inner.prompts[1] and "needs a sub-question" in inner.prompts[1]


def test_planner_gives_up_after_one_repair():
    m, _ = metered(["not json at all"])
    r = Planner(m, DESCRIPTORS).plan("q")
    assert not r.ok and r.abstain == "invalid_plan" and r.usage.calls == 2 and r.errors


def test_planner_rejects_unknown_fields_and_values():
    bad = {"steps": [{"id": "s1", "op": "join", "store": "sql", "question": "x", "output": "user"}]}
    m, _ = metered([json.dumps(bad)])
    r = Planner(m, DESCRIPTORS).plan("q")
    assert r.abstain == "invalid_plan" and any("join" in e for e in r.errors)


def test_prompt_carries_service_ownership_and_never_data():
    m, inner = metered([json.dumps(P01_PLAN)])
    Planner(m, DESCRIPTORS).plan("which python questions mention regex?")
    p = inner.prompts[0]
    assert "User & Activity Service" in p and "Knowledge Graph Service" in p and "Content Service" in p
    assert "QUESTION: which python questions mention regex?" in p
    assert "Do not set must_exist" in p


def test_extract_json_tolerates_surrounding_words():
    assert extract_json('Here is the plan: {"a": 1} done') == {"a": 1}
    with pytest.raises(ValueError):
        extract_json("nothing")


# ------------------------------------------------------------------ translator

def step_of(plan_dict, sid):
    return Plan.from_dict(plan_dict).by_id()[sid]


GRAPH_Q = 'MATCH (q:Question)-[:TAGGED_WITH]->(t:Tag {name: "python"}) RETURN DISTINCT q.id'
DOC_Q = json.dumps({"query": {"bool": {"filter": [{"term": {"doc_type": "question"}},
                                                  {"match": {"title": {"query": "regex", "operator": "and"}}}]}}})


def tr(replies):
    m, inner = metered(replies)
    return Translator(m, DESCRIPTORS, catalogue()), inner


def test_translator_accepts_a_valid_query_and_strips_fences():
    t, _ = tr(["```cypher\n" + GRAPH_Q + "\n```"])
    r = t.translate(step_of(P01_PLAN, "s1"))
    assert r.ok and r.query == GRAPH_Q and r.attempts == 1 and r.usage.calls == 1


def test_translator_repairs_once_and_reports_usage():
    t, inner = tr(["MATCH (q:Question) RETURN q.id LIMIT 5", GRAPH_Q])
    r = t.translate(step_of(P01_PLAN, "s1"))
    assert r.ok and r.attempts == 2 and r.usage.calls == 2
    assert "remove LIMIT" in inner.prompts[1] and "YOUR PREVIOUS REPLY" in inner.prompts[1]


def test_translator_fails_after_one_repair_and_keeps_the_errors():
    t, _ = tr(["DELETE everything"])
    r = t.translate(step_of(P01_PLAN, "s1"))
    assert not r.ok and r.attempts == 2 and r.errors


def test_translator_cannot_answer_is_not_retried():
    t, _ = tr(["CANNOT_ANSWER"])
    r = t.translate(step_of(P01_PLAN, "s1"))
    assert not r.ok and r.attempts == 1 and "cannot answer" in r.errors[0]


def test_bound_prompt_names_the_placeholder_and_contains_no_identifiers():
    plan = {"steps": [
        {"id": "s1", "op": "retrieve", "store": "sql", "question": "users with reputation of at least 5000", "output": "user"},
        {"id": "s2", "op": "retrieve", "store": "graph", "question": "questions asked by these users", "inputs": ["s1"],
         "bind": {"input": "s1", "entity": "user"}, "output": "question"}]}
    t, inner = tr(["MATCH (u:User)-[:ASKED]->(q:Question) WHERE u.id IN $s1 RETURN DISTINCT q.id"])
    r = t.translate(step_of(plan, "s2"))
    assert r.ok
    p = inner.prompts[0]
    assert "$s1" in p and "BOUND IDENTIFIERS" in p and "you never see them" in p
    sql = translator_mod.build_prompt(step_of(plan, "s1"), DESCRIPTORS[Store.SQL])
    assert "BOUND IDENTIFIERS" not in sql and "ANY(" not in sql


def test_prompts_state_the_output_contract():
    plan = {"steps": [
        {"id": "s1", "op": "retrieve", "store": "sql", "question": "top 5 questions by views in 2019", "output": "question",
         "kind": "ranked", "k": 5}]}
    p = translator_mod.build_prompt(step_of(plan, "s1"), DESCRIPTORS[Store.SQL])
    assert "posts.id (restrict to post_type_id = 1)" in p and "top 5" in p and "LIMIT" in p
    sc = {"steps": [{"id": "s1", "op": "retrieve", "store": "sql", "question": "reputation of user 5", "output": "number", "kind": "scalar"}]}
    assert "one row with one column" in translator_mod.build_prompt(step_of(sc, "s1"), DESCRIPTORS[Store.SQL])
    d = translator_mod.build_prompt(step_of(P01_PLAN, "s2"), DESCRIPTORS[Store.DOC])
    assert "filter" in d and "operator" in d and "doc_type" in d


def test_clean_reply():
    assert clean_reply("```sql\nSELECT 1\n```") == "SELECT 1"
    assert clean_reply("SQL: SELECT 1") == "SELECT 1"
    assert clean_reply("  ") == ""


# ------------------------------------------------------------------ pipeline

class Store_:
    def __init__(self, rows, error=None):
        self.rows, self.error, self.seen = rows, error, []

    def fetch(self, step, query, binds, limit):
        self.seen.append(query)
        if self.error:
            raise ConnectionError(self.error)
        return FetchResult(list(self.rows)[:limit], len(self.rows) <= limit)


def pipeline(model_replies_plan, model_replies_tr, graph=(1, 2, 3, 4), doc=(3, 4, 5), **kw):
    pm, _ = metered(model_replies_plan)
    tm, _ = metered(model_replies_tr)
    clients = {Store.GRAPH: Store_(graph), Store.DOC: Store_(doc), Store.SQL: Store_([])}
    clients.update(kw)
    ex = Executor(clients)
    return Federation(Planner(pm, DESCRIPTORS), Translator(tm, DESCRIPTORS, catalogue()), ex), clients


def by_prompt(prompt):
    return DOC_Q if "Elasticsearch" in prompt else GRAPH_Q


def test_pipeline_answers_exactly_and_attributes_cost():
    f, clients = pipeline([json.dumps(P01_PLAN)], by_prompt)
    out = f.answer("python questions with regex in the title")
    assert out.answer.type == "ids" and out.answer.value == [3, 4] and out.answer.guarantee is Guarantee.EXACT
    assert out.planner_usage.calls == 1 and out.translator_usage.calls == 2
    d = out.to_dict()
    assert d["queries"].keys() == {"s1", "s2"} and d["steps"]["s3"]["status"] == "complete"
    assert d["usage"]["translator"]["calls"] == 2 and d["totals"]["calls"] == 2
    json.dumps(d)                                         # serialisable


def test_pipeline_out_of_scope_and_invalid_plan_abstain_without_touching_stores():
    f, clients = pipeline(['{"abstain": "out_of_scope"}'], by_prompt)
    out = f.answer("average")
    assert out.answer.abstained and out.answer.reason == "out_of_scope" and out.run is None
    f, clients = pipeline(["garbage"], by_prompt)
    out = f.answer("q")
    assert out.answer.reason == "invalid_plan" and not clients[Store.GRAPH].seen


def test_pipeline_translation_failure_abstains_and_executes_nothing():
    def replies(p):
        return "DROP TABLE x" if "Elasticsearch" in p else GRAPH_Q
    f, clients = pipeline([json.dumps(P01_PLAN)], replies)
    out = f.answer("q")
    assert out.answer.reason == "translation_failed" and out.run is None
    assert not clients[Store.GRAPH].seen and not clients[Store.DOC].seen
    assert out.to_dict()["translation_errors"]["s2"]


def test_pipeline_store_failure_is_incomplete_not_a_guess():
    f, _ = pipeline([json.dumps(P01_PLAN)], by_prompt, doc=())
    f2, _ = pipeline([json.dumps(P01_PLAN)], by_prompt, **{})
    out = f.answer("q")                                    # doc store returns nothing: exact empty intersection
    assert out.answer.type == "ids" and out.answer.value == [] and out.answer.guarantee is Guarantee.EXACT
    f3, _ = pipeline([json.dumps(P01_PLAN)], by_prompt, **{Store.DOC: Store_([], error="down")})
    out = f3.answer("q")
    assert out.answer.abstained and out.answer.reason == "incomplete"


# ------------------------------------------------------------------ stand-ins versus the real descriptors

def test_fake_descriptors_match_the_real_ones():
    """Runs where Layer 1 (pydantic) is installed. The stand-ins used by the other tests must
    describe the same ownership as the real descriptors, and the real ones must work with the
    planner and translator prompts."""
    pytest.importorskip("pydantic")
    from heterorag.federation.contract import poc_catalogue
    from heterorag.layer1.poc_descriptors import (CONTENT_SERVICE_DESCRIPTOR, KNOWLEDGE_GRAPH_DESCRIPTOR,
                                                  USER_ACTIVITY_DESCRIPTOR)
    assert poc_catalogue() == catalogue()
    real = {Store.SQL: USER_ACTIVITY_DESCRIPTOR, Store.GRAPH: KNOWLEDGE_GRAPH_DESCRIPTOR,
            Store.DOC: CONTENT_SERVICE_DESCRIPTOR}
    services = planner_mod.describe_services(real)
    for d in real.values():
        assert d.display_name in services
    for sid, store in (("s1", Store.GRAPH), ("s2", Store.DOC)):
        p = translator_mod.build_prompt(step_of(P01_PLAN, sid), real[store])
        assert real[store].display_name in p and "SUB-QUESTION" in p
    sql_step = step_of({"steps": [{"id": "s1", "op": "retrieve", "store": "sql", "question": "x", "output": "user"}]}, "s1")
    assert "TABLE users" in translator_mod.build_prompt(sql_step, real[Store.SQL])
