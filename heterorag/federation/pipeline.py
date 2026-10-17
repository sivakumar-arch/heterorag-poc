"""
heterorag/federation/pipeline.py
================================
Question in, typed answer out: planner, translator, executor and emission policy
(Interface Specification v2.1, sections 3 to 7).

Reasons produced here: ``out_of_scope`` and ``invalid_plan`` (planner), ``translation_failed``
(translator). The executor and the emission policy produce ``incomplete`` and
``premise_failed``. A question for which any step cannot be translated is not answered from
the remaining steps: the plan is only valid as a whole.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from . import bindplan
from .answer import Answer, abstain
from .bindplan import BindPolicy, CostModel, Decision
from .executor import Estimate, Executor, RunRecord
from .llm_config import Usage
from .plan import Op, Plan
from .planner import PlanResult, Planner
from .translator import TranslationResult, Translator


@dataclass
class PipelineResult:
    question: str
    plan_result: PlanResult
    translations: dict[str, TranslationResult] = field(default_factory=dict)
    run: RunRecord | None = None
    answer: Answer = field(default_factory=lambda: abstain("invalid_plan"))
    # bind planning (design B)
    bind_policy: str = "none"
    estimates: dict[str, Estimate] = field(default_factory=dict)
    bind_decisions: list[Decision] = field(default_factory=list)
    bind_fallbacks: list[str] = field(default_factory=list)      # restricted translations that failed
    executed_plan: Plan | None = None                            # the plan as run (after restriction)
    probe_ms: float = 0.0
    discarded: list[TranslationResult] = field(default_factory=list)   # independent translations replaced

    @property
    def plan(self) -> Plan | None:
        return self.plan_result.plan

    @property
    def queries(self) -> dict[str, str]:
        return {sid: t.query for sid, t in self.translations.items()}

    @property
    def planner_usage(self) -> Usage:
        return self.plan_result.usage

    @property
    def translator_usage(self) -> Usage:
        total = Usage()
        for t in list(self.translations.values()) + self.discarded:
            total = total + t.usage
        return total

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "plan": self.plan.to_dict() if self.plan else None,
            "plan_errors": self.plan_result.errors,
            "plan_repairs": self.plan_result.repairs,
            "queries": self.queries,
            "translation_errors": {k: t.errors for k, t in self.translations.items() if t.errors},
            "translation_attempts": {k: t.attempts for k, t in self.translations.items()},
            "steps": {sid: r.to_dict() for sid, r in self.run.records.items()} if self.run else None,
            "answer": self.answer.to_dict(),
            "usage": {"planner": self.planner_usage.to_dict(), "translator": self.translator_usage.to_dict()},
            "bind_policy": self.bind_policy,
            "bind_decisions": [d.to_dict() for d in self.bind_decisions],
            "bind_fallbacks": self.bind_fallbacks,
            "estimates": {k: {"n": e.n, "basis": e.basis} for k, e in self.estimates.items()},
            "executed_plan": self.executed_plan.to_dict() if self.executed_plan else None,
            "probe_ms": round(self.probe_ms, 1),
            "totals": self.run.totals() if self.run else None,
        }


class Federation:
    def __init__(self, planner: Planner, translator: Translator, executor: Executor,
                 *, translate_workers: int = 4, bind_policy: BindPolicy = BindPolicy.NONE,
                 cost: CostModel | None = None):
        self._planner, self._translator, self._executor = planner, translator, executor
        self._workers = translate_workers
        self._policy = BindPolicy(bind_policy)
        self._cost = cost or CostModel(batch=executor.batch_size, cap=executor.cap)

    def models_seen(self) -> dict[str, list[str]]:
        """Model names the API reported, per role."""
        return {"planner": self._planner.llm.models_seen, "translator": self._translator.llm.models_seen}

    def answer(self, question: str) -> PipelineResult:
        pr = self._planner.plan(question)
        out = PipelineResult(question, pr)
        out.bind_policy = self._policy.value
        if not pr.ok:
            out.answer = abstain(pr.abstain or "invalid_plan")
            return out

        steps = [s for s in pr.plan.steps if s.op is Op.RETRIEVE]
        with ThreadPoolExecutor(max_workers=min(self._workers, max(1, len(steps)))) as pool:
            results = list(pool.map(self._translator.translate, steps))
        out.translations = {t.step_id: t for t in results}
        failed = [t for t in results if not t.ok]
        if failed:
            out.answer = abstain("translation_failed")
            return out

        plan, queries = self._plan_binds(pr.plan, out)
        out.executed_plan = plan
        out.run = self._executor.run(plan, queries)
        out.answer = out.run.answer
        return out

    # ------------------------------------------------------------------ design B

    def _plan_binds(self, plan: Plan, out: PipelineResult) -> tuple[Plan, dict[str, str]]:
        """Probe, decide, and translate the restricted variants. Any failure here falls back to
        the independent plan, which is always a correct plan: a restriction is an optimisation."""
        out.bind_policy = self._policy.value
        queries = out.queries
        if self._policy is BindPolicy.NONE:
            return plan, queries
        if self._policy is BindPolicy.COST:
            out.estimates, out.probe_ms = self._executor.probe(plan, queries)
        out.bind_decisions = bindplan.decide(plan, out.estimates, self._policy, self._cost)
        wanted = [d for d in out.bind_decisions if d.action == "bind"]
        if not wanted:
            return plan, queries

        restricted = {d.step: bindplan.restrict(plan, d.step, d.source).by_id()[d.step] for d in wanted}
        with ThreadPoolExecutor(max_workers=min(self._workers, len(wanted))) as pool:
            results = list(pool.map(self._translator.translate, restricted.values()))
        ok: list[Decision] = []
        for d, t in zip(wanted, results):
            if t.ok:
                ok.append(d)
                out.discarded.append(out.translations[d.step])
                out.translations[d.step] = t
            else:
                out.discarded.append(t)
                out.bind_fallbacks.append(f"{d.step}: {t.errors[0] if t.errors else 'translation failed'}")
        try:
            return bindplan.apply(plan, ok), out.queries
        except Exception as ex:                       # an invalid combination: keep the independent plan
            out.bind_fallbacks.append(f"restricted plan rejected: {ex}")
            for d in ok:                              # restore the independent translations
                i = next(i for i, t in enumerate(out.discarded) if t.step_id == d.step)
                original = out.discarded.pop(i)
                out.discarded.append(out.translations[d.step])      # its tokens were still spent
                out.translations[d.step] = original
            return plan, out.queries
