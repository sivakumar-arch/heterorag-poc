"""
benchmark_v2/resume.py
======================
Long runs must survive a network loss or a sleeping laptop without turning that interruption
into a result. Two kinds of row can come out of an interrupted run, and they are kept apart:

    infrastructure failure   the model API could not be reached (connection, timeout, overload,
                             rate limit) after the client's own retries. The row carries
                             ``"infra": true`` and is not a result of the method: it is never
                             counted, and a resumed run redoes it.
    method failure           everything else, including a crash of our own code. It stays in the
                             results and is reported as it is.

A run writes its file after every question (atomically), marks it ``complete`` only when every
question has a row without an infrastructure failure, stops after ``MAX_CONSECUTIVE`` infrastructure
failures in a row (the network is down; going on would only burn through the question set) and then
returns ``EXIT_INTERRUPTED`` so that the calling script waits and starts again with ``--resume``.

A third kind of failure is a run that cannot work as configured: a missing or wrong API key, a model
name the account cannot use, a defect that makes every question crash before any model reply. Such a
row is not a result either, and waiting does not help. After ``MAX_SETUP`` of them in a row the run
stops, does not save those rows, and returns ``EXIT_SETUP`` (78, EX_CONFIG) with the reason printed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

EXIT_INTERRUPTED = 75          # EX_TEMPFAIL: try again later
EXIT_SETUP = 78                # EX_CONFIG: the run cannot work as configured; do not wait, fix it
MAX_CONSECUTIVE = 3
MAX_SETUP = 3

_NAMES = {"APIConnectionError", "APITimeoutError", "RateLimitError", "InternalServerError", "OverloadedError",
          "ServiceUnavailableError", "Timeout", "ReadTimeout", "ConnectTimeout", "ConnectError",
          "ConnectionError", "RemoteProtocolError", "TimeoutError"}


def is_infra_exception(ex: BaseException) -> bool:
    """True if the model API (or the network to it) failed, as opposed to a defect in our code."""
    try:
        from heterorag.llm_provider import is_retryable_error
        if is_retryable_error(ex):
            return True
    except Exception:                                      # the provider module is not importable here
        pass
    return any(c.__name__ in _NAMES for c in type(ex).__mro__)


class Guard:
    """Counts infrastructure failures in a row."""

    def __init__(self, limit: int = MAX_CONSECUTIVE):
        self.limit, self.n = limit, 0

    def add(self, row: dict) -> bool:
        """Record a finished row; True if the limit is reached."""
        self.n = self.n + 1 if row.get("infra") else 0
        return self.n >= self.limit


def is_setup_failure(row: dict) -> bool:
    """A question that crashed, or that got no model reply at all, for a reason other than an unreachable API."""
    if row.get("infra"):
        return False
    if str(row.get("reason") or "").startswith("crash:"):
        return True
    return row.get("stop") == "error" and not row.get("turns")


def screen(done: dict[str, dict], out: str, rerun: bool) -> tuple[dict[str, dict], list[str], str | None]:
    """Rows kept by a resume that failed before any model reply (a file written before the stop existed, or by
    hand). Such a row is not a result, but a resume must not change a results file unasked: without ``rerun`` the
    run refuses and says what to do; with it the rows are dropped and redone. Returns (rows to keep, dropped ids,
    refusal message or None)."""
    bad = sorted(q for q, r in done.items() if is_setup_failure(r))
    if not bad:
        return done, [], None
    if rerun:
        return {q: r for q, r in done.items() if q not in set(bad)}, bad, None
    sample = str(done[bad[0]].get("reason") or "unknown")[:200]
    return done, bad, (f"refusing to resume: {len(bad)} of {len(done)} rows in {out} failed before any model reply "
                       f"(for example: {sample}). They are not results. Check the API key and the model name, then "
                       f"either delete the file or add --rerun-setup-failures to redo only those rows.")


class SetupGuard:
    """Counts questions in a row that failed as ``is_setup_failure`` says."""

    def __init__(self, limit: int = MAX_SETUP):
        self.limit, self.n = limit, 0

    def add(self, row: dict) -> bool:
        """Record a finished row; True if the limit is reached."""
        self.n = self.n + 1 if is_setup_failure(row) else 0
        return self.n >= self.limit

    def stop_message(self, rows: list[dict], total: int, out: str) -> str:
        reason = str(rows[-1].get("reason") or "unknown")[:300]
        return (f"\nstopped: {self.n} questions in a row failed before any model reply, and not because the network "
                f"is down. Last reason: {reason}\nCheck the API key (ANTHROPIC_API_KEY in this shell), the model "
                f"name and the environment. Those {self.n} rows were not saved as results; {len(rows) - self.n} of "
                f"{total} results are saved in {out}; start again with --resume.")


def load(path: str) -> dict[str, dict]:
    """Rows of an earlier, interrupted run that are worth keeping, by question id."""
    p = Path(path)
    if not p.exists():
        return {}
    try:
        rows = json.loads(p.read_text())["rows"]
    except Exception:                                      # a damaged file is not trusted
        return {}
    return {r["question_id"]: r for r in rows if not r.get("infra")}


def save(path: str, meta: dict, rows: list[dict], complete: bool) -> None:
    """Atomic write: a reader never sees half a file, and an interrupted write leaves the old one."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps({"meta": dict(meta, complete=complete), "rows": rows}, indent=1, default=str))
    os.replace(tmp, p)


def pending(recs: list[dict], done: dict[str, dict], processed: int) -> list[dict]:
    """Rows of questions that an earlier run finished and that this run has not reached yet. A resumed run rebuilds
    its row list in question order; without these rows an interruption would drop every finished question after the
    current one from the file, and those questions would be run again."""
    return [done[r["question_id"]] for r in recs[processed:] if r["question_id"] in done]


def is_complete(rows: list[dict], n_questions: int) -> bool:
    return len(rows) == n_questions and not any(r.get("infra") for r in rows)
