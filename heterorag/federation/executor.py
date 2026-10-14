"""
heterorag/federation/executor.py
================================
DAG executor with dependent (bind) execution, batching, fetch-until-exhausted, a hard cap,
per-step records and guarantee propagation (Interface Specification v2.1, section 6).

The executor does not know how to talk to a database. It talks to a ``StoreClient`` per store,
which runs one native query and reports honestly whether it read everything. The real clients
(PostgreSQL, Neo4j, Elasticsearch) implement this protocol; tests use in-memory fakes.

Bound identifiers are passed to the client as parameters by this code. They never pass through
any language model (decision D16).
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from . import guarantee as G
from .answer import Answer, emit
from .guarantee import Guarantee, Kind, Status
from .plan import Op, Plan, PlanError, Step, Store

DEFAULT_BATCH = 1000
DEFAULT_CAP = 10_000


@dataclass
class FetchResult:
    """What a store client returns for one native query."""
    values: list                      # keys (set, ranked) or a one-element list (scalar)
    exhausted: bool = True            # False when more rows existed beyond ``limit``
    error: str | None = None          # set when reading stopped early because of an error
    bytes: int = 0


@dataclass(frozen=True)
class Estimate:
    """How many rows an unbound step would return, as far as the store can tell cheaply.

    basis: "count"   the exact number of rows (a bounded count finished within its limit)
           "lower"   at least ``n`` rows: the bounded count stopped at its limit
           "planner" the store's own planner estimate (a fallback; can be far off)"""
    n: int
    basis: str = "count"
    ms: float = 0.0

    @property
    def exceeds(self) -> bool:
        return self.basis == "lower"


class StoreClient(Protocol):
    def fetch(self, step: Step, query: str, binds: Mapping[str, Sequence],
              limit: int) -> FetchResult:
        """Run ``query`` and return at most ``limit`` values. Raise on connection errors."""

    # Optional, used by the bind planner (``bindplan``): ``estimate(step, query, limit) ->
    # Estimate | None`` counts the rows of an unbound set query cheaply (bounded by ``limit``),
    # or returns None when the store cannot say.


@dataclass
class StepRecord:
    step_id: str
    status: Status
    guarantee: Guarantee
    value: Any = None                 # frozenset (set), tuple (ranked), number (scalar), or None
    rows_read: int = 0
    calls: int = 0
    batches: int = 0
    failed_batches: int = 0
    bytes: int = 0
    wall_ms: float = 0.0
    error: str | None = None
    premise_failed: bool = False

    def to_dict(self) -> dict[str, Any]:
        v = self.value
        if isinstance(v, (set, frozenset)):
            v = sorted(v)
        elif isinstance(v, tuple):
            v = list(v)
        return {
            "step": self.step_id, "status": self.status.value, "guarantee": self.guarantee.value,
            "value_size": (len(v) if isinstance(v, list) else None), "rows_read": self.rows_read,
            "calls": self.calls, "batches": self.batches, "failed_batches": self.failed_batches,
            "bytes": self.bytes, "wall_ms": round(self.wall_ms, 3), "error": self.error,
            "premise_failed": self.premise_failed,
        }


@dataclass
class RunRecord:
    records: dict[str, StepRecord]
    answer: Answer
    wall_ms: float

    def totals(self) -> dict[str, float]:
        rs = self.records.values()
        return {"rows_read": sum(r.rows_read for r in rs), "calls": sum(r.calls for r in rs),
                "batches": sum(r.batches for r in rs), "bytes": sum(r.bytes for r in rs),
                "wall_ms": self.wall_ms}


class Executor:
    def __init__(self, clients: Mapping[Store, StoreClient], *, batch_size: int = DEFAULT_BATCH,
                 cap: int = DEFAULT_CAP, retries: int = 1, max_workers: int = 8):
        if batch_size < 1 or cap < 1:
            raise ValueError("batch_size and cap must be positive")
        self._clients = dict(clients)
        self._batch = batch_size
        self._cap = cap
        self._retries = retries
        self._workers = max_workers

    # ------------------------------------------------------------------ public

    @property
    def cap(self) -> int:
        return self._cap

    @property
    def batch_size(self) -> int:
        return self._batch

    def probe(self, plan: Plan, queries: Mapping[str, str]) -> tuple[dict[str, Estimate], float]:
        """Estimate the size of every unbound set retrieve step that has a query, in parallel.
        A store that cannot estimate, or fails, simply yields no entry: the bind planner then keeps
        that step independent. Returns the estimates and the wall time of the probes in ms."""
        t0 = time.perf_counter()
        todo = [s for s in plan.steps if s.op is Op.RETRIEVE and s.bind is None and s.kind is Kind.SET
                and queries.get(s.id) and getattr(self._clients.get(s.store), "estimate", None)]

        def one(s: Step):
            try:
                return s.id, self._clients[s.store].estimate(s, queries[s.id], self._cap)
            except Exception:
                return s.id, None

        out: dict[str, Estimate] = {}
        if todo:
            with ThreadPoolExecutor(max_workers=min(self._workers, len(todo))) as pool:
                for sid, est in pool.map(one, todo):
                    if est is not None:
                        out[sid] = est
        return out, (time.perf_counter() - t0) * 1000.0

    def run(self, plan: Plan, queries: Mapping[str, str]) -> RunRecord:
        problems = plan.validate()
        if problems:
            raise PlanError(problems)
        t0 = time.perf_counter()
        records: dict[str, StepRecord] = {}
        for layer in plan.layers():
            retrieves = [s for s in layer if s.op is Op.RETRIEVE]
            combines = [s for s in layer if s.op is not Op.RETRIEVE]
            if retrieves:
                with ThreadPoolExecutor(max_workers=min(self._workers, len(retrieves))) as pool:
                    futures = [(s, pool.submit(self._retrieve, s, queries.get(s.id), records))
                               for s in retrieves]
                    for s, f in futures:
                        records[s.id] = f.result()
            for s in combines:
                records[s.id] = self._combine(s, records)
        answer = emit(plan, records)
        return RunRecord(records, answer, (time.perf_counter() - t0) * 1000.0)

    # ------------------------------------------------------------------ retrieve

    def _retrieve(self, step: Step, query: str | None, records: Mapping[str, StepRecord]) -> StepRecord:
        t0 = time.perf_counter()
        rec = self._retrieve_inner(step, query, records)
        rec.wall_ms = (time.perf_counter() - t0) * 1000.0
        if step.must_exist and rec.guarantee is Guarantee.EXACT and not rec.value:
            rec.premise_failed = True
        return rec

    def _retrieve_inner(self, step: Step, query: str | None, records) -> StepRecord:
        def failed(msg: str) -> StepRecord:
            return StepRecord(step.id, Status.FAILED, Guarantee.NONE, error=msg)

        if not query or not query.strip():
            return failed("no native query for this step")
        client = self._clients.get(step.store)
        if client is None:
            return failed(f"no client for store {step.store.value}")

        if step.bind is None:
            res = self._call(client, step, query, {}, self._cap)
            return self._single(step, res)

        src = records[step.bind.input]
        if src.value is None or src.guarantee is Guarantee.NONE:
            return failed(f"bound input {src.step_id} has no usable result")
        keys = sorted(src.value)
        if step.kind is Kind.SCALAR:
            # An aggregate over a bound set cannot be split into batches or bounded in general.
            if src.guarantee is not Guarantee.EXACT:
                return failed("a scalar over an inexact bound set has no guarantee")
            if not keys:
                return failed("a scalar over an empty bound set is not defined for every aggregate")
            if len(keys) > self._batch:
                return failed("a scalar over a bound set larger than one batch is not supported")
            return self._single(step, self._call(client, step, query,
                                                 {step.bind.name: keys}, self._cap))
        if not keys:        # nothing to restrict to: the result is empty, no call needed
            return StepRecord(step.id, Status.COMPLETE, src.guarantee, value=frozenset())
        return self._batched(step, query, keys, src.guarantee, client)

    def _call(self, client: StoreClient, step: Step, query: str, binds, limit: int) -> "_Call":
        last: str | None = None
        calls = 0
        for _ in range(1 + self._retries):
            calls += 1
            try:
                return _Call(client.fetch(step, query, binds, limit), calls, None)
            except Exception as ex:                       # connection-level failure
                last = f"{type(ex).__name__}: {ex}"
        return _Call(None, calls, last)

    def _single(self, step: Step, call: "_Call") -> StepRecord:
        if call.result is None:
            return StepRecord(step.id, Status.FAILED, Guarantee.NONE, calls=call.calls, error=call.error)
        res = call.result
        n = len(res.values)
        base = dict(rows_read=n, calls=call.calls, batches=1, bytes=res.bytes)
        if step.kind is Kind.SCALAR:
            ok = res.exhausted and res.error is None and n == 1
            return StepRecord(step.id, Status.COMPLETE if ok else Status.FAILED,
                              Guarantee.EXACT if ok else Guarantee.NONE,
                              value=res.values[0] if ok else None, error=None if ok else
                              (res.error or "scalar step did not return exactly one value"), **base)
        complete = res.exhausted and res.error is None
        status = Status.COMPLETE if complete else (Status.PARTIAL if n else Status.FAILED)
        value = _as_value(step.kind, res.values)
        return StepRecord(step.id, status, G.seed(step.kind, status),
                          value=value if status is not Status.FAILED else None,
                          error=res.error, failed_batches=0 if complete else 1, **base)

    def _batched(self, step: Step, query: str, keys: list, src_g: Guarantee,
                 client: StoreClient) -> StepRecord:
        name = step.bind.name
        acc: set = set()
        calls = batches = failed = nbytes = 0
        capped = False
        errors: list[str] = []
        for i in range(0, len(keys), self._batch):
            batch = keys[i:i + self._batch]
            room = self._cap - len(acc)
            if room <= 0:
                capped = True
                break
            call = self._call(client, step, query, {name: batch}, room)
            calls += call.calls
            batches += 1
            if call.result is None:
                failed += 1
                errors.append(call.error or "error")
                continue
            res = call.result
            nbytes += res.bytes
            acc.update(res.values)
            if res.error is not None:
                failed += 1
                errors.append(res.error)
            elif not res.exhausted:
                capped = True
                break
        total_batches = -(-len(keys) // self._batch)
        complete = (not capped) and failed == 0 and batches == total_batches
        base = dict(rows_read=len(acc), calls=calls, batches=batches, failed_batches=failed,
                    bytes=nbytes, error="; ".join(errors) or ("result cap reached" if capped else None))
        if complete:
            status = Status.COMPLETE
        elif acc:
            status = Status.PARTIAL
        else:
            status = Status.FAILED
        own = G.seed(step.kind, status)
        g = G.retrieve_with_bind(src_g, own) if status is not Status.FAILED else Guarantee.NONE
        value = frozenset(acc) if status is not Status.FAILED else None
        return StepRecord(step.id, status, g, value=value, **base)

    # ------------------------------------------------------------------ combine

    def _combine(self, step: Step, records: Mapping[str, StepRecord]) -> StepRecord:
        ins = [records[i] for i in step.inputs]
        gs = [r.guarantee for r in ins]
        if any(r.value is None for r in ins) or Guarantee.NONE in gs:
            return StepRecord(step.id, Status.FAILED, Guarantee.NONE,
                              error="an input has no usable result")
        sets = [r.value for r in ins]
        if step.op is Op.INTERSECT:
            value, g = frozenset.intersection(*map(frozenset, sets)), G.intersect(gs)
        elif step.op is Op.UNION:
            value, g = frozenset().union(*sets), G.union(gs)
        elif step.op is Op.SUBTRACT:
            value, g = frozenset(sets[0]) - frozenset(sets[1]), G.subtract(gs[0], gs[1])
        elif step.op is Op.COUNT:
            value, g = len(sets[0]), G.count(gs[0])
        else:                                                    # pragma: no cover
            raise PlanError([f"unsupported operator {step.op}"])
        if g is Guarantee.NONE:
            return StepRecord(step.id, Status.FAILED, g, error="operands do not combine to a guarantee")
        status = Status.COMPLETE if g is Guarantee.EXACT else Status.PARTIAL
        return StepRecord(step.id, status, g, value=value)


@dataclass
class _Call:
    result: FetchResult | None
    calls: int
    error: str | None


def _as_value(kind: Kind, values: Sequence):
    if kind is Kind.RANKED:
        return tuple(values)
    return frozenset(values)

