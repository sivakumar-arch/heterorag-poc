"""Everything a model reads is pinned to the state of the repository at the tag ``cp1b-freeze`` (commit 8d7b377).

The model sweep and the capability-floor runs compare models, not prompts. The fingerprints below were computed at that
commit and at the commit that added local models; they were identical. If a prompt or a tool definition is changed on
purpose, this test fails, the change has to be argued, and runs made before and after it are not comparable."""
import hashlib

from benchmark_v2 import agent_baseline as ab
from heterorag.federation import Plan, planner, translator
from tests.federation.fake_descriptors import DESCRIPTORS

STEPS = [
    {"id": "a", "op": "retrieve", "store": "sql", "question": "users with reputation at least 5000", "output": "user"},
    {"id": "b", "op": "retrieve", "store": "graph", "question": "tags used by those users", "output": "tag", "inputs": ["a"],
     "bind": {"input": "a", "entity": "user"}},
    {"id": "c", "op": "retrieve", "store": "doc", "mode": "filter", "question": "questions whose title has model", "output": "question"},
    {"id": "d", "op": "retrieve", "store": "sql", "question": "top five questions of 2015 by views", "output": "question",
     "kind": "ranked", "k": 5},
    {"id": "e", "op": "retrieve", "store": "sql", "question": "count badges of user 7", "output": "number", "kind": "scalar"},
    {"id": "f", "op": "retrieve", "store": "sql", "question": "badge holders", "output": "user", "inputs": ["a"],
     "bind": {"input": "a", "entity": "user"}},
]

PINS = {
    "planner": "34d9c97d23e9761a",
    "translator_a": "02071f6bfff51906", "translator_b": "7b23ba01581919c1", "translator_c": "ad17a82ac9f82e8a",
    "translator_d": "f6b735dad364f6b1", "translator_e": "f365d4d2acad94f9", "translator_f": "e895e98ca2c1efe8",
    "agent_system_plain": "199455d24a611e47", "agent_tools_plain": "e555c251171122f1",
    "agent_system_python": "6f373567419c30f3", "agent_tools_python": "ca7d66bb10f22ea2",
}


def h(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def fingerprints() -> dict:
    plan = Plan.from_dict({"steps": STEPS})
    out = {"planner": h(planner.build_prompt("What is the question?", planner.describe_services(DESCRIPTORS)))}
    for s in plan.steps:
        out["translator_" + s.id] = h(translator.build_prompt(s, DESCRIPTORS[s.store]))
    for arm in ("plain", "python"):
        out["agent_system_" + arm] = h(ab.build_system_prompt(DESCRIPTORS, arm, 10000, 10))
        out["agent_tools_" + arm] = h(repr(ab.tool_definitions(arm, 10000)))
    return out


def test_model_facing_prompts_and_tools_are_those_of_the_cp1b_freeze():
    assert fingerprints() == PINS
