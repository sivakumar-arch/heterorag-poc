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


def test_metered_provider_remembers_the_models_the_api_reported():
    class Reports(LLMProvider):
        model = "configured"
        def complete(self, prompt, *, max_tokens=512, temperature=0.0):
            return LLMResponse(text="x", model="claude-served-1")
    m = MeteredProvider(Reports(), "planner")
    assert m.models_seen == []
    m.complete("a"); m.complete("b")
    assert m.models_seen == ["claude-served-1"]


def test_federation_reports_models_per_role():
    f, _ = pipeline([json.dumps(P01_PLAN)], by_prompt)
    f.answer("q")
    assert f.models_seen() == {"planner": ["scripted"], "translator": ["scripted"]}


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


def chain_plan(n):
    steps = [{"id": "s1", "op": "retrieve", "store": "graph", "question": "q1", "output": "question", "kind": "set"}]
    for i in range(2, n + 1):
        steps.append({"id": f"s{i}", "op": "retrieve", "store": "graph", "question": f"q{i}", "inputs": [f"s{i-1}"],
                      "bind": {"input": f"s{i-1}", "entity": "question"}, "output": "question", "kind": "set"})
    return {"steps": steps}


def test_a_depth_rejection_gets_a_merge_hint_and_other_rejections_do_not():
    m, inner = metered([json.dumps(chain_plan(4)), json.dumps(P01_PLAN)])
    r = Planner(m, DESCRIPTORS).plan("q")
    assert r.ok and r.repairs == 1
    assert "plan depth 4 exceeds 3" in inner.prompts[1] and "HINT" in inner.prompts[1]
    assert "merged into one retrieve step" in inner.prompts[1]
    bad = {"steps": [{"id": "s1", "op": "retrieve", "store": "sql", "output": "user"}]}
    m, inner = metered([json.dumps(bad), json.dumps(P01_PLAN)])
    Planner(m, DESCRIPTORS).plan("q")
    assert "HINT" not in inner.prompts[1]


def test_prompt_keeps_a_path_inside_one_service_as_one_step_without_a_benchmark_example():
    m, inner = metered([json.dumps(P01_PLAN)])
    Planner(m, DESCRIPTORS).plan("q")
    prompt = inner.prompts[0]
    assert "A path that stays inside one service is one retrieve step" in prompt
    rule = prompt[prompt.index("7. A service can follow"):prompt.index("8. Reply with")]
    assert not any(w in rule for w in ("badge", "tag", "reputation", "title", "user"))   # no template content


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


GRAPH_Q = "MATCH (q:Question)-[:TAGGED_WITH]->(t:Tag) WHERE toLower(t.name) = toLower('python') RETURN DISTINCT q.id"
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


def tr_dry(replies, dry):
    m, inner = metered(replies)
    return Translator(m, DESCRIPTORS, catalogue(), dry_run=dry), inner


def test_store_check_rejection_is_repaired_with_the_stores_message():
    calls = []

    def dry(step, query):
        calls.append(query)
        return ["Neo4j rejected the query: Invalid input ':' (line 2)"] if len(calls) == 1 else []
    bad = GRAPH_Q.replace("RETURN DISTINCT", "RETURN DISTINCT :")
    t, inner = tr_dry([bad, GRAPH_Q], {Store.GRAPH: dry})
    r = t.translate(step_of(P01_PLAN, "s1"))
    assert r.ok and r.attempts == 2 and r.store_rejections == 1 and r.query == GRAPH_Q
    assert "Invalid input" in inner.prompts[1]


def test_store_check_is_skipped_when_static_checks_fail_and_when_unavailable():
    asked = []
    t, _ = tr_dry(["DELETE x"], {Store.GRAPH: lambda s, q: asked.append(q) or []})
    assert not t.translate(step_of(P01_PLAN, "s1")).ok and asked == []      # the store is not asked about unsafe text

    def down(step, query):
        raise ConnectionError("store down")
    t, _ = tr_dry([GRAPH_Q], {Store.GRAPH: down})
    r = t.translate(step_of(P01_PLAN, "s1"))
    assert r.ok and r.attempts == 1 and "store check unavailable" in r.notes[0]


def test_no_store_check_for_a_store_without_one():
    t, _ = tr_dry([DOC_Q], {Store.GRAPH: lambda s, q: ["never asked"]})
    assert t.translate(step_of(P01_PLAN, "s2")).ok


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


def test_multi_step_prompt_examples_are_not_isomorphic_to_any_benchmark_template():
    """The examples were written by someone who had read the template catalogue. Their structure
    must not coincide with a benchmark plan (single-step plans are excluded: every one-step plan
    is isomorphic to another)."""
    from benchmark_v2 import questions
    from benchmark_v2.run_slice import signature

    def shape(plan):
        sig = signature(plan)

        def strip(x):                                  # keep operators, stores, modes, kinds and wiring;
            return (x[0], x[1], x[2], x[4], bool(x[6]), tuple(strip(i) for i in x[7]))   # drop entity types and k
        return strip(sig)

    template_shapes = set()
    for t in questions.all_templates():
        if len(t.plan) > 1:
            template_shapes.add(shape(Plan.from_dict({"steps": [questions.step_record(s, 5) for s in t.plan]})))
    dec = json.JSONDecoder()
    checked = 0
    for part in planner_mod._EXAMPLES.split("EXAMPLE ")[1:]:
        obj, _ = dec.raw_decode(part[part.index("\n{") + 1:])
        if "steps" in obj and len(obj["steps"]) > 1:
            assert shape(Plan.from_dict(obj)) not in template_shapes
            checked += 1
    assert checked == 2


def test_graph_bind_instruction_names_the_label_and_property_of_the_bound_entity():
    # regression (live pilot D07-003, D07-004): the model wrote MATCH (:User {id}) ... WHERE id IN $s1
    from heterorag.federation import Bind
    from heterorag.federation.plan import Step
    for entity, label, prop in (("user", "User", "id"), ("question", "Question", "id"), ("tag", "Tag", "name")):
        st = Step("s2", Op.RETRIEVE, Store.GRAPH, None, "q", inputs=("s1",), bind=Bind("s1", entity), output="tag")
        prompt = translator_mod.build_prompt(st, DESCRIPTORS[Store.GRAPH])
        assert f"MATCH (v:{label}) ... WHERE v.{prop} IN $s1" in prompt
        assert "never inside a property map" in prompt
    free = Step("s1", Op.RETRIEVE, Store.GRAPH, None, "q", output="tag")
    assert "BOUND IDENTIFIERS" not in translator_mod.build_prompt(free, DESCRIPTORS[Store.GRAPH])


def test_graph_bind_on_an_entity_without_an_identifier_contract_keeps_the_generic_instruction():
    from heterorag.federation import Bind
    from heterorag.federation.plan import Step
    st = Step("s2", Op.RETRIEVE, Store.GRAPH, None, "q", inputs=("s1",), bind=Bind("s1", "comment"), output="tag")
    assert "IN $s1" in translator_mod.build_prompt(st, DESCRIPTORS[Store.GRAPH])


def test_doc_bind_instruction_names_the_identifier_field_and_makes_the_clause_mandatory():
    # regression (live pilot D05-002): the restricted Elasticsearch query came back without the sentinel
    from heterorag.federation import Bind, Mode
    from heterorag.federation.plan import Step
    for entity, fld in (("question", "post_id"), ("answer", "post_id"), ("user", "user_id"), ("tag", "tag_name")):
        st = Step("s2", Op.RETRIEVE, Store.DOC, Mode.FILTER, "q", inputs=("s1",), bind=Bind("s1", entity), output=entity)
        prompt = translator_mod.build_prompt(st, DESCRIPTORS[Store.DOC])
        assert f'{{"terms": {{"{fld}": "$bind:s1"}}}}' in prompt
        assert "A query without this clause is rejected" in prompt
    # bound identifiers of another entity than the output: the contract field would be wrong, so none is named
    st = Step("s2", Op.RETRIEVE, Store.DOC, Mode.FILTER, "q", inputs=("s1",), bind=Bind("s1", "user"), output="question")
    assert '<identifier field>' in translator_mod.build_prompt(st, DESCRIPTORS[Store.DOC])


def test_graph_rules_state_how_to_bind_and_return_the_identifier():
    # regression (live S05 repeats): the model returned the node, a bare id, or a label used as a variable
    from heterorag.federation.plan import Step
    st = Step("s1", Op.RETRIEVE, Store.GRAPH, None, "q", output="answer")
    prompt = translator_mod.build_prompt(st, DESCRIPTORS[Store.GRAPH])
    assert "A label is not a variable" in prompt and "RETURN DISTINCT n.id, never the node itself" in prompt
    rule = next(l for l in prompt.splitlines() if l.startswith("7. Give every node"))
    assert "S05" not in rule and "Answer" not in rule and "Question" not in rule          # generic wording


def test_every_store_prompt_states_the_case_insensitive_rule_in_its_own_idiom():
    # regression (CP1, D04 and D07): the model wrote 'Reliability' for the badge "reliability" and the query found nothing
    from heterorag.federation.plan import Mode, Step
    sql_p = translator_mod.build_prompt(Step("s1", Op.RETRIEVE, Store.SQL, None, "q", output="user"), DESCRIPTORS[Store.SQL])
    assert "without regard to case" in sql_p and "lower(column) = lower('value')" in sql_p
    assert "strpos(lower(column), lower('text'))" in sql_p
    cy_p = translator_mod.build_prompt(Step("s1", Op.RETRIEVE, Store.GRAPH, None, "q", output="question"), DESCRIPTORS[Store.GRAPH])
    assert "without regard to case" in cy_p and "toLower(t.name) = toLower('x')" in cy_p and "property map" in cy_p
    es_p = translator_mod.build_prompt(Step("s1", Op.RETRIEVE, Store.DOC, Mode.FILTER, "q", output="tag"), DESCRIPTORS[Store.DOC])
    assert "without regard to case" in es_p and '"case_insensitive": true' in es_p
    for p in (sql_p, cy_p, es_p):
        rules = p.split("RULES")[-1].split("SUB-QUESTION")[0].lower()
        assert "reliability" not in rules and "badge" not in rules                      # generic wording, no benchmark value


def test_a_case_sensitive_comparison_is_returned_to_the_translator_with_the_fix():
    bad = "SELECT DISTINCT user_id FROM badges WHERE name = 'Reliability'"
    good = "SELECT DISTINCT user_id FROM badges WHERE lower(name) = lower('reliability')"
    t, inner = tr([bad, good])
    from heterorag.federation.plan import Step
    step = Step("s1", Op.RETRIEVE, Store.SQL, None, "users with the reliability badge", output="user")
    r = t.translate(step)
    assert r.ok and r.query == good and r.attempts == 2
    assert "without regard to case" in inner.prompts[1] and "lower(name) = lower('Reliability')" in inner.prompts[1]


def test_planner_prompt_defines_depth_as_a_chain_of_retrieves():
    prompt = planner_mod.build_prompt("q", "services")
    assert "depth at most 3" in prompt and "longest chain of retrieve steps" in prompt
    assert "run locally and add nothing" in prompt


# ------------------------------------------------------------------ consecutive steps on one service

def _step(sid, store, entity, **kw):
    d = {"id": sid, "op": "retrieve", "store": store, "question": f"q{sid}", "output": entity, "kind": "set"}
    d.update(kw)
    return d


def _bound(sid, store, src, src_entity, entity):
    return _step(sid, store, entity, inputs=[src], bind={"input": src, "entity": src_entity})


SPLIT_D07 = {"steps": [
    _step("s1", "sql", "user"), _bound("s2", "graph", "s1", "user", "question"),
    _bound("s3", "graph", "s2", "question", "tag"),
    {"id": "s4", "op": "count", "inputs": ["s3"], "output": "number", "kind": "scalar"}]}
MERGED_D07 = {"steps": [
    _step("s1", "sql", "user"), _bound("s2", "graph", "s1", "user", "tag"),
    {"id": "s3", "op": "count", "inputs": ["s2"], "output": "number", "kind": "scalar"}]}


def test_merge_advice_flags_a_bound_chain_inside_a_sql_or_graph_service_only():
    from heterorag.federation.planner import merge_advice
    assert len(merge_advice(Plan.from_dict(SPLIT_D07))) == 1 and "s2 and s3" in merge_advice(Plan.from_dict(SPLIT_D07))[0]
    assert merge_advice(Plan.from_dict(MERGED_D07)) == []
    sql_chain = {"steps": [_step("s1", "sql", "user"), _bound("s2", "sql", "s1", "user", "question")]}
    assert merge_advice(Plan.from_dict(sql_chain))
    doc_chain = {"steps": [_step("s1", "doc", "question", mode="filter"),
                           _bound("s2", "doc", "s1", "question", "answer") | {"mode": "filter"}]}
    assert merge_advice(Plan.from_dict(doc_chain)) == []             # a document store cannot follow relationships
    cross = {"steps": [_step("s1", "sql", "user"), _bound("s2", "graph", "s1", "user", "question"),
                       _bound("s3", "sql", "s2", "question", "question")]}
    assert merge_advice(Plan.from_dict(cross)) == []                  # different services alternate


def test_a_restricted_plan_with_two_steps_on_one_service_is_still_valid():
    from heterorag.federation.bindplan import restrict
    p = Plan.from_dict({"steps": [_step("a", "graph", "question"), _step("b", "graph", "question"),
                                  {"id": "c", "op": "intersect", "inputs": ["a", "b"], "output": "question",
                                   "kind": "set"}]})
    assert restrict(p, "b", "a").validate() == []                     # the system may restrict within a service


def test_a_split_plan_is_sent_back_once_with_the_merge_hint_and_the_merged_reply_is_used():
    m, inner = metered([json.dumps(SPLIT_D07), json.dumps(MERGED_D07)])
    r = Planner(m, DESCRIPTORS).plan("q")
    assert r.ok and r.repairs == 1 and r.errors == [] and len(r.plan.steps) == 3
    assert "consecutive steps on the same service" in inner.prompts[1] and "HINT" in inner.prompts[1]


def test_a_plan_without_consecutive_steps_is_not_sent_back():
    m, inner = metered([json.dumps(MERGED_D07)])
    r = Planner(m, DESCRIPTORS).plan("q")
    assert r.ok and r.repairs == 0 and r.errors == [] and len(inner.prompts) == 1


def test_if_the_second_reply_is_still_split_invalid_or_an_abstention_the_first_valid_plan_runs():
    for second in (json.dumps(SPLIT_D07), "not json at all", json.dumps({"abstain": "out_of_scope"}),
                   json.dumps({"steps": [{"id": "s1", "op": "retrieve", "store": "sql", "output": "user"}]})):
        m, _ = metered([json.dumps(SPLIT_D07), second])
        r = Planner(m, DESCRIPTORS).plan("q")
        assert r.ok and r.abstain is None and r.repairs == 1 and len(r.plan.steps) == 4, second
        assert r.errors and "consecutive steps on the same service" in r.errors[0]       # the advice is kept in the row


def test_an_invalid_first_reply_and_a_split_second_reply_is_accepted_not_abstained():
    bad = {"steps": [{"id": "s1", "op": "retrieve", "store": "sql", "output": "user"}]}
    m, _ = metered([json.dumps(bad), json.dumps(SPLIT_D07)])
    r = Planner(m, DESCRIPTORS).plan("q")
    assert r.ok and r.repairs == 1 and r.errors and len(r.plan.steps) == 4
