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

from .answer import Answer, abstain
from .executor import Executor, RunRecord
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
        for t in self.translations.values():
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
            "totals": self.run.totals() if self.run else None,
        }


class Federation:
    def __init__(self, planner: Planner, translator: Translator, executor: Executor,
                 *, translate_workers: int = 4):
        self._planner, self._translator, self._executor = planner, translator, executor
        self._workers = translate_workers

    def answer(self, question: str) -> PipelineResult:
        pr = self._planner.plan(question)
        out = PipelineResult(question, pr)
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

        out.run = self._executor.run(pr.plan, out.queries)
        out.answer = out.run.answer
        return out
