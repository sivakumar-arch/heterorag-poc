"""
heterorag/federation/answer.py
==============================
The typed answer and the emission policy (Interface Specification v2.1, sections 6 and 7).

An answer is computed by code from the final step. A language model may verbalise it for
presentation, but scoring and guarantees refer to the typed value only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

from .guarantee import Guarantee, Kind
from .plan import Plan

if TYPE_CHECKING:                                    # pragma: no cover
    from .executor import StepRecord

MAX_PARTIAL_SET = 50     # a superset larger than this is not a useful answer


@dataclass(frozen=True)
class Answer:
    type: str                          # ids | ranked | scalar | abstain
    value: Any = None                  # sorted list | list | number | None
    guarantee: Guarantee = Guarantee.NONE
    reason: str | None = None          # abstention reason

    @property
    def abstained(self) -> bool:
        return self.type == "abstain"

    @property
    def exact(self) -> bool:
        return not self.abstained and self.guarantee is Guarantee.EXACT

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "value": self.value, "guarantee": self.guarantee.value,
                "reason": self.reason}


def abstain(reason: str) -> Answer:
    return Answer("abstain", None, Guarantee.NONE, reason)


def emit(plan: Plan, records: "Mapping[str, StepRecord]") -> Answer:
    """Apply the emission policy to the plan's final step.

    Reasons produced here: ``premise_failed`` and ``incomplete``. The reasons ``out_of_scope``,
    ``invalid_plan`` and ``translation_failed`` are produced by the planner and the translator.
    """
    if any(r.premise_failed for r in records.values()):
        return abstain("premise_failed")
    sink = plan.sink()
    rec = records.get(sink.id)
    if rec is None or rec.value is None or rec.guarantee is Guarantee.NONE:
        return abstain("incomplete")

    g, v = rec.guarantee, rec.value
    if sink.kind is Kind.SCALAR:
        if g is Guarantee.EXACT or g is Guarantee.UPPER or (g is Guarantee.LOWER and v > 0):
            return Answer("scalar", v, g)
        return abstain("incomplete")
    if sink.kind is Kind.RANKED:
        return Answer("ranked", list(v), g) if g is Guarantee.EXACT else abstain("incomplete")

    items = sorted(v)
    if g is Guarantee.EXACT:
        return Answer("ids", items, g)                       # including a legitimate empty set
    if g is Guarantee.SUBSET and items:
        return Answer("ids", items, g)
    if g is Guarantee.SUPERSET and len(items) <= MAX_PARTIAL_SET:
        return Answer("ids", items, g)
    return abstain("incomplete")
