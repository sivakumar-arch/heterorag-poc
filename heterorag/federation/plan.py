"""
heterorag/federation/plan.py
============================
The plan: a DAG of typed steps over a closed operator set
(Interface Specification v2.1, sections 2 and 3).

The same step fields are used by the benchmark annotation (``benchmark_v2.questions.Step``);
a test asserts that the two schemas stay identical.

Operators in this version: retrieve, intersect, union, subtract, count.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any

from .guarantee import Kind

MAX_STEPS = 6
MAX_DEPTH = 3


class Op(str, Enum):
    RETRIEVE = "retrieve"
    INTERSECT = "intersect"
    UNION = "union"
    SUBTRACT = "subtract"
    COUNT = "count"


COMBINE_OPS = (Op.INTERSECT, Op.UNION, Op.SUBTRACT, Op.COUNT)


class Store(str, Enum):
    SQL = "sql"
    GRAPH = "graph"
    DOC = "doc"


class Mode(str, Enum):
    FILTER = "filter"      # predicate: complete enumeration
    SEARCH = "search"      # relevance: explicit top-k


class PlanError(ValueError):
    """The plan is malformed. ``args[0]`` is the list of problems."""


@dataclass(frozen=True)
class Bind:
    input: str             # id of the step whose output is injected
    entity: str            # entity type of the injected identifiers
    param: str = ""        # placeholder name in the native query; defaults to the input id

    @property
    def name(self) -> str:
        return self.param or self.input


@dataclass(frozen=True)
class Step:
    id: str
    op: Op
    store: Store | None = None
    mode: Mode | None = None
    question: str = ""
    inputs: tuple[str, ...] = ()
    bind: Bind | None = None
    output: str = ""                  # entity type of the output ("number" for scalars)
    kind: Kind = Kind.SET
    k: int | None = None
    must_exist: bool = False          # an empty result means the question's premise fails


STEP_FIELDS = tuple(f.name for f in fields(Step))


@dataclass(frozen=True)
class Plan:
    steps: tuple[Step, ...]

    # ---- structure
    def by_id(self) -> dict[str, Step]:
        return {s.id: s for s in self.steps}

    def sink(self) -> Step:
        used = {i for s in self.steps for i in s.inputs}
        sinks = [s for s in self.steps if s.id not in used]
        if len(sinks) != 1:
            raise PlanError([f"plan must have exactly one final step, found {len(sinks)}"])
        return sinks[0]

    def layers(self) -> list[list[Step]]:
        """Steps grouped so that every step depends only on earlier groups."""
        by = self.by_id()
        placed: set[str] = set()
        out: list[list[Step]] = []
        remaining = list(self.steps)
        while remaining:
            layer = [s for s in remaining if all(i in placed for i in s.inputs)]
            if not layer:
                raise PlanError(["the plan contains a cycle"])
            out.append(layer)
            placed |= {s.id for s in layer}
            remaining = [s for s in remaining if s.id not in placed]
        assert len(placed) == len(by)
        return out

    def depth(self) -> int:
        """The longest chain of retrieve steps in which each consumes the one before it, that is the
        number of store round trips that must happen one after another. Combining and counting steps
        run locally and add no level (an intersect after a chain of three retrieves is still depth 3)."""
        level: dict[str, int] = {}
        for layer in self.layers():                      # raises PlanError on a cycle
            for s in layer:
                level[s.id] = max((level[i] for i in s.inputs), default=0) + (1 if s.op is Op.RETRIEVE else 0)
        return max(level.values(), default=0)

    # ---- validation
    def validate(self) -> list[str]:
        """Return every problem found; an empty list means the plan is well formed."""
        errs: list[str] = []
        if not self.steps:
            return ["plan has no steps"]
        ids = [s.id for s in self.steps]
        if len(set(ids)) != len(ids):
            errs.append("step ids are not unique")
        if len(self.steps) > MAX_STEPS:
            errs.append(f"plan has {len(self.steps)} steps, limit is {MAX_STEPS}")
        by = {s.id: s for s in self.steps}
        for s in self.steps:
            errs += self._check_step(s, by)
        if errs:
            return errs
        try:
            if self.depth() > MAX_DEPTH:
                errs.append(f"plan depth {self.depth()} exceeds {MAX_DEPTH}")
            self.sink()
        except PlanError as e:
            errs += list(e.args[0])
        return errs

    @staticmethod
    def _check_step(s: Step, by: dict[str, Step]) -> list[str]:
        e: list[str] = []
        tag = f"step {s.id}"
        for i in s.inputs:
            if i not in by:
                e.append(f"{tag}: unknown input {i!r}")
            elif i == s.id:
                e.append(f"{tag}: consumes itself")
        if any(i not in by for i in s.inputs):
            return e
        ins = [by[i] for i in s.inputs]
        if not s.output:
            e.append(f"{tag}: output entity type missing")

        if s.op is Op.RETRIEVE:
            if s.store is None:
                e.append(f"{tag}: retrieve needs a store")
            if not s.question.strip():
                e.append(f"{tag}: retrieve needs a sub-question")
            if (s.store is Store.DOC) != (s.mode is not None):
                e.append(f"{tag}: mode is required for the document store and only there")
            if s.mode is Mode.SEARCH and not s.k:
                e.append(f"{tag}: search mode needs an explicit k")
            if s.mode is Mode.FILTER and s.k is not None:
                e.append(f"{tag}: filter mode takes no k")
            if s.kind is Kind.RANKED and (not s.k or s.k < 1):
                e.append(f"{tag}: ranked retrieve needs k >= 1")
            if s.kind is not Kind.RANKED and s.mode is not Mode.SEARCH and s.k is not None:
                e.append(f"{tag}: k is only for ranked or search steps")
            if s.must_exist and s.kind is not Kind.SET:
                e.append(f"{tag}: must_exist applies to set steps only")
            if s.inputs and s.bind is None:
                e.append(f"{tag}: a retrieve consumes another step only through a bind")
            if s.bind is not None:
                if len(s.inputs) != 1 or s.bind.input != s.inputs[0]:
                    e.append(f"{tag}: bind must name the single input step")
                else:
                    src = ins[0]
                    if src.kind is not Kind.SET:
                        e.append(f"{tag}: can only bind a set, step {src.id} is {src.kind.value}")
                    if src.output != s.bind.entity:
                        e.append(f"{tag}: bind entity {s.bind.entity!r} differs from "
                                 f"output entity {src.output!r} of {src.id}")
                    if s.kind is Kind.RANKED:
                        e.append(f"{tag}: a bind into a ranked step is not allowed in version 1")
        else:
            if s.store is not None or s.mode is not None or s.question or s.bind or s.must_exist:
                e.append(f"{tag}: {s.op.value} takes no store, mode, question, bind or must_exist")
            if s.k is not None:
                e.append(f"{tag}: k is not used by {s.op.value} in this version")
            if any(i.kind is not Kind.SET for i in ins):
                e.append(f"{tag}: {s.op.value} takes sets")
            if s.op in (Op.INTERSECT, Op.UNION):
                if len(ins) < 2:
                    e.append(f"{tag}: {s.op.value} needs at least two inputs")
            elif s.op is Op.SUBTRACT:
                if len(ins) != 2:
                    e.append(f"{tag}: subtract needs exactly two inputs (minuend, subtrahend)")
            elif s.op is Op.COUNT:
                if len(ins) != 1:
                    e.append(f"{tag}: count needs exactly one input")
                if s.kind is not Kind.SCALAR:
                    e.append(f"{tag}: count produces a scalar")
            if s.op is not Op.COUNT:
                if s.kind is not Kind.SET:
                    e.append(f"{tag}: {s.op.value} produces a set")
                if len({i.output for i in ins}) > 1 or (ins and ins[0].output != s.output):
                    e.append(f"{tag}: {s.op.value} needs inputs and output of one entity type")
        return e

    # ---- (de)serialisation
    def to_dict(self) -> dict[str, Any]:
        out = []
        for s in self.steps:
            d: dict[str, Any] = {"id": s.id, "op": s.op.value, "output": s.output, "kind": s.kind.value}
            if s.store:
                d["store"] = s.store.value
            if s.mode:
                d["mode"] = s.mode.value
            if s.question:
                d["question"] = s.question
            if s.inputs:
                d["inputs"] = list(s.inputs)
            if s.bind:
                d["bind"] = {"input": s.bind.input, "entity": s.bind.entity, "param": s.bind.param}
            if s.k is not None:
                d["k"] = s.k
            if s.must_exist:
                d["must_exist"] = True
            out.append(d)
        return {"steps": out}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Plan":
        """Parse a plan (for example the LLM's JSON). Raises PlanError with every problem found,
        so one repair message can list them all."""
        errs: list[str] = []
        raw = data.get("steps") if isinstance(data, dict) else None
        if not isinstance(raw, list) or not raw:
            raise PlanError(["'steps' must be a non-empty list"])
        known = set(STEP_FIELDS)
        steps: list[Step] = []
        for n, d in enumerate(raw):
            tag = f"steps[{n}]"
            if not isinstance(d, dict):
                errs.append(f"{tag}: not an object")
                continue
            extra = set(d) - known
            if extra:
                errs.append(f"{tag}: unknown fields {sorted(extra)}")
            try:
                bind = d.get("bind")
                steps.append(Step(
                    id=str(d["id"]),
                    op=Op(d["op"]),
                    store=Store(d["store"]) if d.get("store") else None,
                    mode=Mode(d["mode"]) if d.get("mode") else None,
                    question=str(d.get("question", "")),
                    inputs=tuple(str(i) for i in d.get("inputs", ())),
                    bind=Bind(str(bind["input"]), str(bind["entity"]), str(bind.get("param", "")))
                    if bind else None,
                    output=str(d.get("output", "")),
                    kind=Kind(d.get("kind", "set")),
                    k=int(d["k"]) if d.get("k") is not None else None,
                    must_exist=bool(d.get("must_exist", False)),
                ))
            except (KeyError, ValueError, TypeError) as ex:
                errs.append(f"{tag}: {type(ex).__name__}: {ex}")
        if errs:
            raise PlanError(errs)
        return cls(tuple(steps))
