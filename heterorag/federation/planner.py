"""
heterorag/federation/planner.py
===============================
Turns a natural-language question into a typed plan over the closed operator set
(Interface Specification v2.1, section 3).

The model sees the capability of each service and the plan language, never any data. Its JSON
output is parsed and validated by code (``Plan.from_dict`` and ``Plan.validate``). A plan that
fails is returned to the model once with the full list of problems. A question outside the plan
language is declined by the model with ``{"abstain": "out_of_scope"}``.

The worked examples in the prompt are fixed and are not benchmark questions.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .contract import ID_CONTRACT
from .llm_config import MeteredProvider, Usage
from .plan import MAX_DEPTH, MAX_STEPS, Plan, PlanError, Store

_STORE_KEYS = {Store.SQL: "sql", Store.GRAPH: "graph", Store.DOC: "doc"}


@dataclass
class PlanResult:
    plan: Plan | None = None
    abstain: str | None = None            # out_of_scope | invalid_plan
    errors: list[str] = field(default_factory=list)
    repairs: int = 0
    usage: Usage = field(default_factory=Usage)
    raw: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.plan is not None


def describe_services(descriptors: dict) -> str:
    """The services block of the prompt, from the Layer 1 descriptors."""
    out = []
    for store in (Store.SQL, Store.GRAPH, Store.DOC):
        d = descriptors[store]
        ident = "; ".join(f"{e}: {s}" for e, s in ID_CONTRACT[store].items())
        out.append(f'store "{_STORE_KEYS[store]}" ({d.display_name})\n  {d.capability_summary or ''}\n  Identifiers: {ident}')
    return "\n\n".join(out)


_EXAMPLES = """EXAMPLE 1 (one service, one number)
Question: How many views does question 777 have?
{"steps": [
 {"id": "s1", "op": "retrieve", "store": "sql", "question": "the view count of question 777", "output": "number", "kind": "scalar"}
]}

EXAMPLE 2 (independent sub-questions on two services, combined in code)
Question: Which answers belong to question 4 or contain the word "monad"?
{"steps": [
 {"id": "s1", "op": "retrieve", "store": "graph", "question": "answers of question 4", "output": "answer", "kind": "set"},
 {"id": "s2", "op": "retrieve", "store": "doc", "mode": "filter", "question": "answers whose text contains the word monad", "output": "answer", "kind": "set"},
 {"id": "s3", "op": "union", "inputs": ["s1", "s2"], "output": "answer", "kind": "set"}
]}

EXAMPLE 3 (a later step needs the identifiers of an earlier step: dependent step with a bind, combined with an independent step)
Question: Which questions asked by users located in Berlin have the word "bayesian" in their title?
{"steps": [
 {"id": "s1", "op": "retrieve", "store": "sql", "question": "users whose location is Berlin", "output": "user", "kind": "set"},
 {"id": "s2", "op": "retrieve", "store": "graph", "question": "questions asked by these users", "inputs": ["s1"],
  "bind": {"input": "s1", "entity": "user"}, "output": "question", "kind": "set"},
 {"id": "s3", "op": "retrieve", "store": "doc", "mode": "filter", "question": "questions whose title contains the word bayesian", "output": "question", "kind": "set"},
 {"id": "s4", "op": "intersect", "inputs": ["s2", "s3"], "output": "question", "kind": "set"}
]}

EXAMPLE 4 (outside the plan language)
Question: What is the average reputation of users who answered question 5?
{"abstain": "out_of_scope"}"""


def build_prompt(question: str, services: str) -> str:
    return f"""You plan how to answer a question about Stack Overflow data that is spread over three autonomous services. You write a plan; code runs it. You never see data.

SERVICES
{services}

PLAN LANGUAGE
A plan is a JSON object {{"steps": [...]}}. At most {MAX_STEPS} steps and depth at most {MAX_DEPTH}. Exactly one step is the final step (no other step consumes it). Step fields:
- id: a short unique name (s1, s2, ...).
- op: one of
    retrieve   run one sub-question on one service
    intersect  entities in all inputs (two or more inputs)
    union      entities in any input (two or more inputs)
    subtract   entities in the first input and not in the second (exactly two inputs, in that order)
    count      the number of entities in one input (produces a number)
- store: "sql", "graph" or "doc" (retrieve only).
- mode: for "doc" only. "filter" for a condition that every returned document must satisfy (a word occurs, a phrase occurs); "search" only for a relevance ranking with an explicit k.
- question: the sub-question in plain English (retrieve only). It must be complete without the original question: include every value (ids, names, words, numbers, years) it needs, and say which entity type it returns.
- inputs: ids of the steps consumed (combine operators, and bound retrieves).
- bind: {{"input": <step id>, "entity": <entity type>}} on a retrieve that must be restricted to the entities produced by an earlier step.
- output: the entity type produced: user, question, answer, tag, comment, or "number".
- kind: "set" (distinct entities), "ranked" (an ordered list of exactly k entities, k given in the field k), or "scalar" (one number).
- k: the number of entities for ranked steps and for doc search steps.

RULES
1. Each service owns only what its description says. Send each sub-question to the service that owns the facts it needs. A condition on a number or date goes to sql, a relationship goes to graph, a condition on text goes to doc.
2. Use independent retrieve steps combined with intersect, union or subtract when each sub-question can be answered on its own.
3. Use bind when a sub-question can only be asked about the entities another step produced. The bound step must be a set (never ranked), and the entity type of the bind must equal the output of the input step. Prefer independent steps when they are enough.
4. combine operators take sets of one entity type and produce the same entity type. count produces a number.
5. A question that needs arithmetic over several numbers (averages, sums, ratios), recursion, data that no service owns, or an opinion is outside the plan language. Reply {{"abstain": "out_of_scope"}}.
6. Do not set must_exist.
7. Reply with the JSON object only, without explanation or code fences.

{_EXAMPLES}

QUESTION: {question}"""


def build_repair_prompt(base: str, previous: str, errors: list[str]) -> str:
    listed = "\n".join(f"- {e}" for e in errors)
    return f"""{base}

YOUR PREVIOUS REPLY
{previous}

IT WAS REJECTED FOR THESE REASONS
{listed}

Reply again with the corrected JSON object only."""


def extract_json(text: str):
    """Parse the JSON object in a reply, tolerating code fences and surrounding words."""
    t = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text.strip())
    try:
        return json.loads(t)
    except ValueError:
        pass
    a, b = t.find("{"), t.rfind("}")
    if a >= 0 and b > a:
        return json.loads(t[a:b + 1])
    raise ValueError("no JSON object found in the reply")


class Planner:
    def __init__(self, provider: MeteredProvider, descriptors: dict, *, max_tokens: int = 2000):
        self._llm, self._max = provider, max_tokens
        self._services = describe_services(descriptors)

    @property
    def llm(self) -> MeteredProvider:
        return self._llm

    def plan(self, question: str) -> PlanResult:
        res = PlanResult()
        base = build_prompt(question, self._services)
        prompt = base
        for attempt in (0, 1):
            resp, u = self._llm.call(prompt, max_tokens=self._max, temperature=0.0)
            res.usage = res.usage + u
            res.raw.append(resp.text)
            errs: list[str] = []
            try:
                data = extract_json(resp.text)
                if isinstance(data, dict) and "abstain" in data and "steps" not in data:
                    res.abstain, res.errors, res.repairs = "out_of_scope", [], attempt
                    return res
                plan = Plan.from_dict(data)
                errs = plan.validate()
                if not errs:
                    res.plan, res.errors, res.repairs = plan, [], attempt
                    return res
            except PlanError as ex:
                errs = list(ex.args[0])
            except ValueError as ex:
                errs = [f"the reply is not valid JSON: {ex}"]
            res.errors, res.repairs = errs, attempt
            prompt = build_repair_prompt(base, resp.text, errs)
        res.abstain = "invalid_plan"
        return res
