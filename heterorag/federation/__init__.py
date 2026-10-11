"""
heterorag.federation
====================
Federated query processing core: typed plans, dependent execution, guarantee levels.
See the Interface Specification v2.1 in the project documents.

This package does not use the earlier retrieve-and-generate layers (layer2, layer3, layer4),
which remain as the baseline system.
"""

from .answer import Answer, abstain, emit
from .executor import Executor, FetchResult, RunRecord, StepRecord, StoreClient
from .guarantee import Guarantee, Kind, Status
from .plan import Bind, Mode, Op, Plan, PlanError, Step, Store

__all__ = [
    "Answer", "abstain", "emit", "Executor", "FetchResult", "RunRecord", "StepRecord",
    "StoreClient", "Guarantee", "Kind", "Status", "Bind", "Mode", "Op", "Plan", "PlanError",
    "Step", "Store",
]
