"""
heterorag/federation/bindplan.py
================================
The system, not the language model, decides whether an independent sub-query should become a
dependent one (design B; checkpoint document, DQ1).

The planner writes only the *logically* necessary dependencies (a bind where a store lacks the
predicate). Where the stores could answer each part alone and the parts are then combined by
intersection or difference, there is a second way to execute the same plan: run one part first
and restrict the other to its identifiers. Both ways return the same set (Proposition 2, below);
they differ in the rows read, the calls made and whether the hard cap truncates the result. This
module chooses between them in four levels:

    0  necessity      the planner's binds are kept as written (not decided here)
    1  eligibility    only a store-owned filter predicate over a set can be restricted by
                      identifiers without changing the result: a retrieve step that is an input
                      of an intersection, or the *subtrahend* of a difference. Never a union,
                      never a ranked, search or scalar step.
    2  feasibility    a probe counts the rows of each unbound step. A step whose independent
                      result exceeds the cap H must be bound if the set it is bound to fits in H;
                      a set that exceeds H cannot be injected exactly.
    3  profitability  when independent execution is exact, bind only if the estimated time is
                      lower: ``t_call`` per native call and ``t_row`` per row read, bound batches
                      of ``batch`` identifiers run one after another and after their source.

Proposition 2 (equivalence). Let X and Y be the true results of two filter queries on identifier
sets. Running Y, then X restricted to the identifiers of Y, returns X intersect Y, which is what
intersecting the independent results returns. For a difference, restricting the subtrahend B to
the minuend A returns B intersect A, and A minus (B intersect A) = A minus B. Restricting the
minuend of a difference would return A intersect B, which is wrong, so it is never done. The
property holds only when the restricted query is a pure filter on the store's own predicate
(a "set" step, filter mode); a top-k, a relevance search or an aggregate over the restricted set
differs from the restricted result of the unrestricted one.

The cost constants are parameters of the model. The defaults below are placeholders; they are
fitted on the development split and checked on the test split before any result is reported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum
from typing import Mapping

from .executor import Estimate
from .guarantee import Kind
from .plan import Bind, Mode, Op, Plan, PlanError, Step


class BindPolicy(str, Enum):
    NONE = "none"        # run the planner's plan as written (design A: the planner decides)
    ALWAYS = "always"    # bind every eligible step to its sibling (ablation)
    COST = "cost"        # design B


@dataclass(frozen=True)
class CostModel:
    t_call: float = 0.02       # seconds per native call (placeholder, to be fitted)
    t_row: float = 5e-5        # seconds per row read (placeholder, to be fitted)
    batch: int = 1000          # identifiers per bound call
    cap: int = 10_000          # hard cap H on the rows of one retrieve step
    margin: float = 1.0        # bind on profit only if t_bound < margin * t_independent

    def t_read(self, rows: int) -> float:
        return self.t_call + self.t_row * rows

    def t_bound(self, source_rows: int, result_rows: int) -> float:
        """Time of a restricted step once its source is available."""
        if source_rows == 0:
            return 0.0
        return math.ceil(source_rows / self.batch) * self.t_call + self.t_row * result_rows


@dataclass(frozen=True)
class Decision:
    step: str                      # the retrieve step considered for restriction
    source: str | None             # the step whose identifiers it would be restricted to
    combine: str                   # the combining step the pair belongs to
    action: str                    # "bind" | "independent"
    level: str                     # eligibility | feasibility | profitability | policy | missing
    reason: str
    t_independent: float | None = None
    t_bound: float | None = None

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in ("step", "source", "combine", "action", "level", "reason")}
        for k in ("t_independent", "t_bound"):
            v = getattr(self, k)
            d[k] = None if v is None else round(v, 4)
        return d


# ----------------------------------------------------------------------------- eligibility

def _eligible(step: Step) -> str | None:
    """Why this retrieve step cannot be restricted by identifiers, or None if it can."""
    if step.op is not Op.RETRIEVE:
        return "not a retrieve step"
    if step.bind is not None:
        return "already dependent"
    if step.kind is not Kind.SET:
        return f"a {step.kind.value} step is not a pure filter"
    if step.mode is Mode.SEARCH:
        return "a relevance search is not a pure filter"
    if step.must_exist:
        return "must_exist would test the restricted result"
    return None


def _pairs(plan: Plan):
    """(combine step, step to restrict, step it is restricted to) for every restriction that is
    equivalent by Proposition 2, plus (combine, None, None) for a union, recorded as ineligible."""
    by = plan.by_id()
    for c in plan.steps:
        if c.op is Op.INTERSECT:
            for x in c.inputs:
                for y in c.inputs:
                    if x != y:
                        yield c, by[x], by[y]
        elif c.op is Op.SUBTRACT:
            yield c, by[c.inputs[1]], by[c.inputs[0]]          # only the subtrahend, bound to the minuend
        elif c.op is Op.UNION:
            yield c, None, None


def restrict(plan: Plan, step_id: str, source_id: str) -> Plan:
    """The plan with ``step_id`` restricted to the identifiers of ``source_id``."""
    src = plan.by_id()[source_id]
    steps = tuple(replace(s, inputs=(source_id,), bind=Bind(source_id, src.output))
                  if s.id == step_id else s for s in plan.steps)
    return Plan(steps)


# ----------------------------------------------------------------------------- the decision

def decide(plan: Plan, estimates: Mapping[str, Estimate], policy: BindPolicy,
           cost: CostModel) -> list[Decision]:
    """One decision per combining step: restrict one of its inputs or keep them independent."""
    out: list[Decision] = []
    chosen: dict[str, Decision] = {}
    options: dict[str, list[Decision]] = {}
    by = plan.by_id()
    combines = {c.id: c for c in plan.steps}

    for c, x, y in _pairs(plan):
        if x is None:
            out.append(Decision("", None, c.id, "independent", "eligibility",
                                "a union is not restricted by identifiers: restricting one side changes the result"))
            continue
        why = _eligible(x)
        if why is None and by[y.id].op is not Op.RETRIEVE:
            why = f"its sibling {y.id} is not a single retrieve step"
        if why is None:
            try:
                restricted = restrict(plan, x.id, y.id)
                bad = restricted.validate()
                if bad:
                    why = "the restricted plan is invalid: " + bad[0]
            except PlanError as e:
                why = f"the restricted plan is invalid: {e.args[0][0]}"
        if why is not None:
            options.setdefault(c.id, []).append(
                Decision(x.id, y.id, c.id, "independent", "eligibility", why))
            continue
        options.setdefault(c.id, []).append(_assess(c, x, y, plan, estimates, policy, cost))

    for cid, ds in options.items():
        if cid in combines and combines[cid].op is Op.UNION:
            continue
        binds = [d for d in ds if d.action == "bind"]
        if binds:
            # prefer a source whose size is known exactly, then the cheaper restricted execution
            def key(d: Decision):
                e = estimates.get(d.source)
                return (0 if e is not None and e.basis == "count" else 1,
                        d.t_bound if d.t_bound is not None else 0.0, d.step)
            out.append(min(binds, key=key))
        else:
            # report the most informative reason (a cost-level one before eligibility) and, when
            # the pairs differ, what the other pairs said, so that no reason is hidden
            rank = {"profitability": 0, "feasibility": 1, "missing": 2, "policy": 3, "eligibility": 4}
            best = min(ds, key=lambda d: (rank.get(d.level, 9), d.step))
            others = [f"{d.step}<-{d.source}: {d.reason}" for d in ds if d is not best and d.reason != best.reason]
            if others:
                best = replace(best, reason=best.reason + " | also " + " | ".join(others))
            out.append(best)
    return out


def _assess(c: Step, x: Step, y: Step, plan: Plan, est: Mapping[str, Estimate], policy: BindPolicy,
            cost: CostModel) -> Decision:
    """Decide for one ordered pair: restrict ``x`` to the identifiers of ``y``."""
    if policy is BindPolicy.NONE:
        return Decision(x.id, y.id, c.id, "independent", "policy", "binds are left to the planner")
    if policy is BindPolicy.ALWAYS:
        # deterministic ablation: the second input is bound to the first (a difference: the subtrahend)
        first = c.inputs[0]
        if y.id == first and x.id == c.inputs[1]:
            return Decision(x.id, y.id, c.id, "bind", "policy", "always bind (ablation)")
        return Decision(x.id, y.id, c.id, "independent", "policy", "always-bind ablation uses input order")

    ex, ey = est.get(x.id), est.get(y.id)
    if ex is None or ey is None:
        missing = x.id if ex is None else y.id
        return Decision(x.id, y.id, c.id, "independent", "missing", f"no size estimate for step {missing}")
    siblings = [est.get(i) for i in c.inputs]
    if any(e is None for e in siblings):
        return Decision(x.id, y.id, c.id, "independent", "missing", "a sibling of the combining step has no estimate")

    cap = cost.cap
    known_over = any(e.n > cap or e.exceeds for e in siblings)        # an input is known to exceed the cap
    # exact only when every input was counted (not estimated by the store's planner) and fits the cap
    independent_exact = all(e.basis == "count" and e.n <= cap for e in siblings)
    t_ind = max(cost.t_read(e.n) for e in siblings)
    y_fits = ey.n <= cap and not ey.exceeds
    if not y_fits:
        return Decision(x.id, y.id, c.id, "independent", "feasibility",
                        f"{y.id} returns more than the cap ({cap}), so its identifiers cannot be injected exactly",
                        t_ind, None)
    result = min(ex.n, ey.n)
    others = [cost.t_read(est[i].n) for i in c.inputs if i not in (x.id, y.id)]
    t_bound = max([cost.t_read(ey.n) + cost.t_bound(ey.n, result)] + others)
    if not independent_exact:
        # either an input is known to exceed the cap, or a size is only the store planner's guess
        # (which can be far off): then independent execution cannot be shown to be exact, while the
        # restricted execution is bounded by the exactly or plausibly small set, so it is chosen
        why = (f"independent execution would exceed the cap ({cap}); {y.id} has {ey.n} identifiers" if known_over
               else f"independent execution cannot be shown to fit the cap ({cap}): a size is only the store "
                    f"planner's estimate; {y.id} has {ey.n} identifiers ({ey.basis})")
        return Decision(x.id, y.id, c.id, "bind", "feasibility", why, t_ind, t_bound)
    if t_bound < cost.margin * t_ind:
        return Decision(x.id, y.id, c.id, "bind", "profitability",
                        f"estimated {t_bound:.3f}s bound against {t_ind:.3f}s independent",
                        t_ind, t_bound)
    return Decision(x.id, y.id, c.id, "independent", "profitability",
                    f"estimated {t_bound:.3f}s bound is not below {t_ind:.3f}s independent", t_ind, t_bound)


def apply(plan: Plan, decisions: list[Decision]) -> Plan:
    """The plan with every 'bind' decision applied. Each restricted step becomes dependent."""
    for d in decisions:
        if d.action == "bind":
            plan = restrict(plan, d.step, d.source)
    bad = plan.validate()
    if bad:
        raise PlanError(bad)
    return plan
