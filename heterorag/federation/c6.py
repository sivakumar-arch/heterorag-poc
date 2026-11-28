"""C6, the empty-bind probe (pre-registration section 2; Batch 107).

For each retrieve step that is a SET step with a bind and is not ranked (the scope of R7), the final query is run once
against its store with the bound set replaced by the empty set and the row cap set to ``C6_ROWS``. The query passes
if no identifier comes back. This is the behavioural form of premise PM stated pointwise, I(empty set) = empty set:
it rejects a query whose result does not depend on its bound input, including every neutralised bind (``OR TRUE``,
``1=1``, ``OR id > 0``), whatever the syntax.

Empty means no identifier (decision 1 of 2026-11-28): the result is read through the same identifier extraction the
executor uses (the store client's ``fetch``, first column or identifier field), and the probe fires only if at least
one non-null identifier comes back. A null row, which an OPTIONAL MATCH or an outer join returns on a correct query,
does not fire.

If the store cannot be asked (it raises, or returns an error with no identifier), the probe is skipped: it passes,
and the reason is recorded and counted. A skip is never a firing.

The positive control (``control``) runs, per store, one query whose bind is neutralised (it must fire) and one
restricted query (it must pass), written with the bind placeholder each store's translator prompt asks for, through
the same ``probe`` function as the run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .guarantee import Kind
from .plan import Bind, Mode, Op, Step, Store

C6_ROWS = 1       # row cap of the probe (registered)
FIRED, PASSED, SKIPPED, OUT_OF_SCOPE = "fired", "passed", "skipped", "out_of_scope"

# One message for every store; no schema text (registered: check messages contain no schema-specific text).
C6_MESSAGE = ("the query returns identifiers even when its bound input is empty, so its result does not depend on "
              "the bound identifiers; every result must satisfy the restriction to the bound identifiers")


@dataclass(frozen=True)
class ProbeResult:
    status: str
    reason: str = ""
    identifiers: int = 0          # non-null identifiers seen (at most C6_ROWS)

    def to_dict(self) -> dict:
        return asdict(self)


def in_scope(step: Step) -> bool:
    """A retrieve step with a bind whose result is a set and is not ranked (the scope of R7)."""
    return (step.op is Op.RETRIEVE and step.bind is not None and step.kind is Kind.SET
            and step.mode is not Mode.SEARCH)


def _brief(text: Any, limit: int = 200) -> str:
    return " ".join(str(text).split())[:limit]


def probe(client: Any, step: Step, query: str, rows: int = C6_ROWS) -> ProbeResult:
    """Run ``query`` with the step's bound set empty. Fired: at least one non-null identifier came back."""
    if not in_scope(step):
        return ProbeResult(OUT_OF_SCOPE)
    try:
        res = client.fetch(step, query, {step.bind.name: []}, rows)
    except Exception as ex:                                  # the store could not be asked
        return ProbeResult(SKIPPED, f"{type(ex).__name__}: {_brief(ex)}")
    ids = [v for v in (res.values or []) if v is not None]
    if ids:
        return ProbeResult(FIRED, C6_MESSAGE, len(ids))
    if res.error:
        return ProbeResult(SKIPPED, f"store error: {_brief(res.error)}")
    return ProbeResult(PASSED)


# ----------------------------------------------------------------------------- positive control

_CONTROL_BIND = "c6"
_CONTROL = {
    # store: (entity, mode, vacuous query (must fire), restricted query (must pass)); placeholders as the translator's
    # prompts ask: SQL ``= ANY(%(name)s)``, Cypher ``IN $name``, document store ``"$bind:name"``.
    Store.SQL: ("user", None,
                "SELECT u.id FROM users u WHERE u.id = ANY(%(c6)s) OR TRUE",
                "SELECT u.id FROM users u WHERE u.id = ANY(%(c6)s)"),
    Store.GRAPH: ("user", None,
                  "MATCH (u:User) WHERE u.id IN $c6 OR true RETURN u.id",
                  "MATCH (u:User) WHERE u.id IN $c6 RETURN u.id"),
    Store.DOC: ("question", Mode.FILTER,
                '{"query": {"bool": {"should": [{"terms": {"post_id": "$bind:c6"}}, {"match_all": {}}], '
                '"minimum_should_match": 1}}}',
                '{"query": {"terms": {"post_id": "$bind:c6"}}}'),
}


def control_step(store: Store) -> Step:
    entity, mode, _, _ = _CONTROL[store]
    return Step(id="c6_control", op=Op.RETRIEVE, store=store, mode=mode, question="C6 positive control",
                inputs=("c6_input",), bind=Bind(input="c6_input", entity=entity, param=_CONTROL_BIND),
                output=entity, kind=Kind.SET)


def control(clients: Mapping[Store, Any]) -> dict:
    """Per store: the vacuous query must fire and the restricted one must pass. ``ok`` is True only if every store
    present passes both; a store without a client is reported as missing and fails the control."""
    out: dict = {"stores": {}, "ok": True}
    for store in (Store.SQL, Store.GRAPH, Store.DOC):
        client = clients.get(store)
        if client is None:
            out["stores"][store.value] = {"ok": False, "why": "no client for the store"}
            out["ok"] = False
            continue
        step = control_step(store)
        _, _, vacuous, restricted = _CONTROL[store]
        v, r = probe(client, step, vacuous), probe(client, step, restricted)
        ok = v.status == FIRED and r.status == PASSED
        why = "" if ok else (f"vacuous query: {v.status}{' (' + v.reason + ')' if v.reason and v.status != FIRED else ''}; "
                             f"restricted query: {r.status}{' (' + r.reason + ')' if r.reason else ''}")
        out["stores"][store.value] = {"ok": ok, "vacuous": v.status, "restricted": r.status, "why": why}
        out["ok"] = out["ok"] and ok
    return out
