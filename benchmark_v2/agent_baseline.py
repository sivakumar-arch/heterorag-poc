"""
benchmark_v2/agent_baseline.py
==============================
The free-form tool-use agent: the baseline that decides whether a federated query processor is
needed at all (checkpoint CP1).

One model (the same default as the planner and translator, decision D23) is given the same
service descriptions, schemas and identifier contract as our system, read-only access to the same
three stores through native query tools, a turn budget, and a typed final answer. It plans in
its own head, writes its own queries, reads results through its context and corrects itself
from the store's error messages. There is no plan language, no executor, no bind and no
guarantee level.

Two arms, so that the comparison is not made against a weak opponent:

* ``plain``   tools ``sql``, ``cypher``, ``es`` and ``final_answer``. Results come back as text
              (the first rows and a row count); identifiers can only move between stores by being
              read in the context and written into the next query.
* ``python``  plain plus a ``python`` tool whose helpers ``sql()``, ``cypher()`` and ``es()``
              return full result lists (up to the same cap H) into a Python process, so the agent
              can intersect and count in code. This is the strongest generic baseline: it is the
              executor written by the model.

What is held equal to the federation system: model, schemas and identifier contract, the hard cap
H on the rows of one native call, read-only stores, and the statement and transaction timeouts.
What differs, deliberately and as a disclosed asymmetry: our translator sees the store's EXPLAIN
check before a query runs and the executor's rules (no LIMIT, one statement); the agent sees the
store's error after the query ran and may retry within its turn budget.

    python -m benchmark_v2.agent_baseline --questions data/questions_v2_dev_classD_probe.jsonl \
        --templates D01,D04,D05,D07 --limit 6 --arm plain --out data/agent_plain.json
"""

from __future__ import annotations

import argparse
import builtins
import contextlib
import io
import json
import math
import os
import re
import signal
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from benchmark_v2 import resume

CAP = 10_000                 # hard cap H on the rows of one native call (same as the federation run)
SHOW_ROWS = 50               # rows of a result shown to the model; the answer cap, so a complete answer is always visible
SHOW_CHARS = 5000            # characters of a tool result shown to the model
PYTHON_SECONDS = 30
DEFAULT_MAX_TURNS = 10
DEFAULT_MAX_TOKENS = 4096     # a reply is not cut off in the middle of a tool call or of python code

ARMS = ("plain", "python")


# ----------------------------------------------------------------------------- stores

@dataclass
class RawResult:
    columns: list[str]
    rows: list[list]
    exhausted: bool = True         # False when more rows existed than were returned


class Stores(Protocol):
    def sql(self, query: str, limit: int) -> RawResult: ...
    def cypher(self, query: str, limit: int) -> RawResult: ...
    def es(self, body: dict, limit: int) -> RawResult: ...


def _plain(v: Any) -> Any:
    from decimal import Decimal
    if isinstance(v, Decimal):
        return int(v) if v == v.to_integral_value() else float(v)
    return v


class LiveStores:
    """Read-only access to the local docker stores, configured like ``live_clients``."""

    def __init__(self, env: Mapping[str, str] | None = None, *, pg_timeout_ms: int = 20_000,
                 neo_timeout_s: float = 20.0):
        e = env if env is not None else os.environ
        self._pg_dsn = e.get("PG_DSN", "postgresql://heterorag:heterorag_secret@localhost:5432/heterorag")
        self._pg_timeout, self._neo_timeout = pg_timeout_ms, neo_timeout_s
        from neo4j import READ_ACCESS, GraphDatabase
        self._driver = GraphDatabase.driver(e.get("NEO4J_URI", "bolt://localhost:7687"),
                                            auth=(e.get("NEO4J_USER", "neo4j"),
                                                  e.get("NEO4J_PASSWORD", "heterorag_secret")))
        self._read = READ_ACCESS
        from elasticsearch import Elasticsearch
        from heterorag.es_index import INDEX_NAME
        self._es = Elasticsearch(e.get("ES_URL", "http://localhost:9200"), request_timeout=60)
        self._index = INDEX_NAME

    def sql(self, query: str, limit: int) -> RawResult:
        import psycopg2
        conn = psycopg2.connect(self._pg_dsn)
        try:
            conn.set_session(readonly=True)
            with conn.cursor() as c0:
                c0.execute("SET LOCAL statement_timeout = %s", (self._pg_timeout,))
            with conn.cursor(name="agent_read") as cur:
                cur.execute(query)
                rows = cur.fetchmany(limit + 1)
                cols = [d[0] for d in (cur.description or [])]
            return RawResult(cols, [[_plain(v) for v in r] for r in rows[:limit]], len(rows) <= limit)
        finally:
            try:
                conn.rollback()
            finally:
                conn.close()

    def cypher(self, query: str, limit: int) -> RawResult:
        from neo4j import unit_of_work

        @unit_of_work(timeout=self._neo_timeout)
        def work(tx):
            res = tx.run(query)
            keys = list(res.keys())
            rows = []
            for rec in res:
                rows.append([_plain(v) for v in rec.values()])
                if len(rows) > limit:
                    break
            return keys, rows

        with self._driver.session(default_access_mode=self._read) as s:
            keys, rows = s.execute_read(work)
        return RawResult(keys, rows[:limit], len(rows) <= limit)

    def es(self, body: dict, limit: int) -> RawResult:
        body = dict(body)
        size = int(body.get("size", 10))
        if size > limit:
            raise ValueError(f"size {size} is above the cap {limit}")
        body.pop("pit", None)
        body.setdefault("track_total_hits", True)
        resp = self._es.search(index=self._index, body=body)
        hits = resp["hits"]["hits"]
        cols = sorted({k for h in hits for k in (h.get("_source") or {})})
        rows = [[(h.get("_source") or {}).get(c) for c in cols] for h in hits]
        total = resp["hits"].get("total", {})
        more = isinstance(total, dict) and total.get("value", 0) > len(hits)
        return RawResult(cols, rows, not more)


# ----------------------------------------------------------------------------- tools

def _qschema(desc: str) -> dict:
    return {"type": "object", "properties": {"query": {"type": "string", "description": desc}},
            "required": ["query"]}


def tool_definitions(arm: str, cap: int) -> list[dict]:
    tools = [
        {"name": "sql", "description": "Run one read-only PostgreSQL SELECT on the user-activity service. "
         f"Returns the columns, the first rows and the number of rows returned (at most {cap}).",
         "input_schema": _qschema("one SELECT statement")},
        {"name": "cypher", "description": "Run one read-only Cypher query on the knowledge-graph service. "
         f"Returns the columns, the first rows and the number of rows returned (at most {cap}).",
         "input_schema": _qschema("one Cypher query ending in RETURN")},
        {"name": "es", "description": "Run one Elasticsearch search body on the content service. Use "
         '"_source" to choose fields and "size" (at most ' + str(cap) + ") to get more than 10 hits.",
         "input_schema": {"type": "object", "properties": {"body": {"type": "object"}}, "required": ["body"]}},
    ]
    if arm == "python":
        tools.append({"name": "python", "description":
                      "Run Python code. Helpers sql(query), cypher(query) and es(body) return the full result "
                      f"as a list of rows, each row a tuple of the columns (at most {cap} rows; the list has "
                      ".truncated, True if more rows existed, and .columns). "
                      "Use print() to see output. Modules: collections, itertools, math, json, re, statistics. "
                      "Variables persist between calls for this question.",
                      "input_schema": {"type": "object", "properties": {"code": {"type": "string"}},
                                       "required": ["code"]}})
    tools.append({"name": "final_answer", "description":
                  "Give the final answer. kind 'set': values are distinct identifiers; 'ranked': values in order; "
                  "'number': put the number in 'number'; 'abstain': the question cannot be answered, give a reason.",
                  "input_schema": {"type": "object", "properties": {
                      "kind": {"type": "string", "enum": ["set", "ranked", "number", "abstain"]},
                      "values": {"type": "array", "items": {}},
                      "number": {"type": "number"},
                      "reason": {"type": "string"}}, "required": ["kind"]}})
    return tools


def build_system_prompt(desc: Mapping, arm: str, cap: int, max_turns: int, *, show_rows: int = SHOW_ROWS) -> str:
    from heterorag.federation.contract import ID_CONTRACT
    from heterorag.federation.plan import Store
    from heterorag.federation.translator import _doc_schema, _graph_schema, _sql_schema
    blocks = []
    for store, key, schema in ((Store.SQL, "sql", _sql_schema), (Store.GRAPH, "cypher", _graph_schema),
                               (Store.DOC, "es", _doc_schema)):
        d = desc[store]
        ident = "; ".join(f"{e}: {s}" for e, s in ID_CONTRACT[store].items())
        blocks.append(f"SERVICE {d.display_name} (tool: {key})\n{d.capability_summary or ''}\n"
                      f"Identifiers: {ident}\n\n{schema(d)}")
    py = ("\nThe python tool lets you keep whole result sets in variables and combine them in code; "
          "prefer it when a result is larger than the rows shown to you.\n" if arm == "python" else "")
    return f"""You answer questions about Stack Overflow data that is spread over three autonomous services, each with its own database. You have read-only query tools for each. No single query can join across services: combine results yourself.

{chr(10).join(blocks)}

NOTES
- Tag names are strings, other identifiers are integers. DUPLICATE_OF points from the duplicate to the question it duplicates. Match CO_OCCURS_WITH without direction.
- In PostgreSQL quote the column names class, date and count with double quotes. For a year use EXTRACT(YEAR FROM creation_date).
- Text values (names, tags, locations) are compared without regard to case: lower() in SQL, toLower() in Cypher, and "case_insensitive": true on a term over a keyword field in Elasticsearch.
- For a condition on words in text use an Elasticsearch match query with "operator": "and" and a term on doc_type, so that whole words match.
- A single call returns at most {cap} rows. You see only the first {show_rows} rows of a result and the number of rows returned.{py}
ANSWERING
- Finish by calling final_answer exactly once, with every identifier asked for (a set), the ordered identifiers (ranked) or the number. If a count is asked, answer with kind 'number'.
- You have at most {max_turns} turns, each turn may use several tools. Call final_answer before the budget is spent. If the question cannot be answered from these services, call final_answer with kind 'abstain'."""


# ----------------------------------------------------------------------------- python tool

class Rows(list):
    """A result list that remembers whether the cap truncated it."""
    truncated: bool = False
    columns: list = []


_ALLOWED_MODULES = {"collections", "itertools", "math", "json", "re", "statistics", "functools"}
_SAFE_BUILTINS = {n: getattr(builtins, n) for n in (
    "abs all any bool dict enumerate filter float frozenset int isinstance len list map max min "
    "next range reversed round set sorted str sum tuple zip print repr sorted divmod pow "
    "ValueError KeyError TypeError IndexError Exception StopIteration type hasattr getattr chr ord iter bytes "
    "format callable issubclass RuntimeError AttributeError NotImplementedError AssertionError ZeroDivisionError "
    "LookupError".split())}


def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.split(".")[0] not in _ALLOWED_MODULES:
        raise ImportError(f"import of {name!r} is not allowed")
    return __import__(name, globals, locals, fromlist, level)


class _Timeout(Exception):
    pass


@contextlib.contextmanager
def _alarm(seconds: int):
    """Interrupt runaway code with SIGALRM (main thread on Unix); elsewhere there is no timeout."""
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def handler(signum, frame):
        raise _Timeout(f"the code ran longer than {seconds} s")

    try:
        old = signal.signal(signal.SIGALRM, handler)
    except ValueError:                                   # not the main thread
        yield
        return
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def run_python(code: str, stores: Stores, cap: int, seconds: int = PYTHON_SECONDS,
               on_rows: Callable[[int], None] | None = None, ns: dict | None = None,
               max_chars: int = SHOW_CHARS) -> tuple[str, bool]:
    """Execute model-written code against read-only helpers. Returns (output text, error?).
    ``ns`` is the namespace of the question: variables set by one call are there in the next, as in
    a notebook. The restricted builtins guard against accidents; this is not a security boundary
    (do not run it on data an attacker can write to)."""
    def wrap(fn):
        def helper(arg):
            r = fn(arg, cap)
            if on_rows:
                on_rows(len(r.rows))
            rows = Rows(tuple(x) for x in r.rows)             # always tuples, as the tool description says
            rows.truncated, rows.columns = not r.exhausted, list(r.columns)
            return rows
        return helper

    ns = {} if ns is None else ns
    ns.update({"__builtins__": {**_SAFE_BUILTINS, "__import__": _restricted_import},
               "sql": wrap(stores.sql), "cypher": wrap(stores.cypher), "es": wrap(stores.es)})
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out), _alarm(seconds):
            exec(compile(code, "<agent>", "exec"), ns)
        err = False
        text = out.getvalue()
    except BaseException as ex:                         # model code may raise anything
        if isinstance(ex, (KeyboardInterrupt, SystemExit)) and not isinstance(ex, _Timeout):
            raise
        err = True
        text = out.getvalue() + f"\n{type(ex).__name__}: {ex}"
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n... (output truncated at {max_chars} characters)"
    return text.strip() or "(no output)", err


def render_rows(res: RawResult, show: int = SHOW_ROWS, max_chars: int = SHOW_CHARS) -> str:
    payload: dict[str, Any] = {"columns": res.columns, "rows": res.rows[:show],
                               "rows_returned": len(res.rows), "all_rows_returned": res.exhausted}
    if len(res.rows) > show:
        payload["note"] = f"only the first {show} rows are shown"
    text = json.dumps(payload, default=str)
    return text if len(text) <= max_chars else text[:max_chars] + " ... (truncated)"


# ----------------------------------------------------------------------------- the model

@dataclass
class Call:
    id: str
    name: str
    input: dict


@dataclass
class Turn:
    content: list[dict]            # assistant content blocks, in API format
    calls: list[Call]
    text: str = ""
    prompt_tokens: int = 0
    reply_tokens: int = 0
    seconds: float = 0.0
    model: str = ""
    stop_reason: str = ""


class AgentLLM(Protocol):
    def turn(self, system: str, messages: list[dict], tools: list[dict], max_tokens: int) -> Turn: ...


class AnthropicAgent:
    """Native tool use through the Anthropic Messages API. Retries transient errors like the
    federation's provider; sampling is the model default."""

    def __init__(self, model: str | None = None, client=None, retries: int = 4):
        from heterorag.federation.llm_config import DEFAULT_MODEL
        self.model = model or os.environ.get("HETERORAG_AGENT_MODEL") or DEFAULT_MODEL
        if client is None:
            import anthropic
            client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"),
                                         timeout=float(os.environ.get("HETERORAG_API_TIMEOUT", "180")))
        self._client, self._retries = client, retries
        self.models_seen: set[str] = set()

    def turn(self, system: str, messages: list[dict], tools: list[dict], max_tokens: int) -> Turn:
        from heterorag.llm_provider import is_retryable_error
        t0, last = time.perf_counter(), None
        for attempt in range(self._retries + 1):
            try:
                r = self._client.messages.create(model=self.model, max_tokens=max_tokens, system=system,
                                                 messages=messages, tools=tools)
                break
            except Exception as ex:
                last = ex
                if attempt == self._retries or not is_retryable_error(ex):
                    raise
                time.sleep(min(2 ** attempt, 20))
        else:                                                         # pragma: no cover
            raise last
        content, calls, text = [], [], []
        for b in r.content:
            if b.type == "text":
                content.append({"type": "text", "text": b.text})
                text.append(b.text)
            elif b.type == "tool_use":
                content.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
                calls.append(Call(b.id, b.name, dict(b.input)))
        if getattr(r, "model", None):
            self.models_seen.add(r.model)
        return Turn(content, calls, "".join(text), r.usage.input_tokens, r.usage.output_tokens,
                    time.perf_counter() - t0, getattr(r, "model", "") or "", r.stop_reason or "")


# ----------------------------------------------------------------------------- the loop

@dataclass
class AgentResult:
    kind: str | None = None            # set | ranked | number | abstain, as the agent gave it
    values: list = field(default_factory=list)
    number: float | None = None
    reason: str = ""
    stop: str = ""                     # answered | max_turns | no_tool | error
    infra: bool = False                # stop == "error" because the model API could not be reached
    turns: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    prompt_tokens: int = 0
    reply_tokens: int = 0
    llm_seconds: float = 0.0
    rows_read: int = 0
    store_ms: float = 0.0
    transcript: list[dict] = field(default_factory=list)

    @property
    def answered(self) -> bool:
        return self.stop == "answered" and self.kind is not None


def _brief(x: Any, n: int) -> str:
    s = x if isinstance(x, str) else json.dumps(x, default=str)
    return s if len(s) <= n else s[:n] + "..."


def run_agent(question: str, llm: AgentLLM, stores: Stores, system: str, tools: list[dict], *,
              arm: str, cap: int = CAP, max_turns: int = DEFAULT_MAX_TURNS,
              max_tokens: int = DEFAULT_MAX_TOKENS, show_rows: int = SHOW_ROWS,
              show_chars: int = SHOW_CHARS) -> AgentResult:
    res = AgentResult()
    messages: list[dict] = [{"role": "user", "content": question}]
    nudged = False
    py_ns: dict = {}                                       # the python tool's variables for this question

    def count_rows(n: int):
        res.rows_read += n

    for _ in range(max_turns):
        try:
            t = llm.turn(system, messages, tools, max_tokens)
        except Exception as ex:
            res.stop, res.reason = "error", f"{type(ex).__name__}: {str(ex)[:200]}"
            res.infra = resume.is_infra_exception(ex)
            return res
        res.turns += 1
        res.prompt_tokens += t.prompt_tokens
        res.reply_tokens += t.reply_tokens
        res.llm_seconds += t.seconds
        if t.stop_reason not in ("tool_use", "end_turn", ""):
            # e.g. "max_tokens": the reply was cut off, possibly in the middle of a tool call
            res.transcript.append({"tool": "_turn", "stop_reason": t.stop_reason,
                                   "calls": len(t.calls), "text": _brief(t.text, 200)})
        messages.append({"role": "assistant", "content": t.content})
        if not t.calls:
            if nudged:
                res.stop = "no_tool"
                return res
            nudged = True
            messages.append({"role": "user", "content": "Use the tools, and finish with final_answer."})
            continue
        results = []
        for c in t.calls:
            if c.name == "final_answer":
                res.stop = "answered"
                res.kind = str(c.input.get("kind", ""))
                res.values = list(c.input.get("values") or [])
                res.number = c.input.get("number")
                res.reason = str(c.input.get("reason") or "")
                res.transcript.append({"tool": "final_answer", "input": _brief(c.input, 300)})
                return res
            res.tool_calls += 1
            t0 = time.perf_counter()
            err = False
            seen: dict[str, Any] = {}          # what the store returned against what the model was shown
            try:
                if c.name in ("sql", "cypher", "es"):
                    if c.name == "es":
                        r = stores.es(dict(c.input["body"]), cap)
                    else:
                        r = (stores.sql if c.name == "sql" else stores.cypher)(str(c.input["query"]), cap)
                    res.rows_read += len(r.rows)
                    out = render_rows(r, show_rows, show_chars)
                    seen = {"rows": len(r.rows), "exhausted": bool(r.exhausted), "shown": min(show_rows, len(r.rows))}
                elif c.name == "python" and arm == "python":
                    out, err = run_python(str(c.input["code"]), stores, cap,
                                          on_rows=count_rows, ns=py_ns, max_chars=show_chars)
                else:
                    out, err = f"unknown tool {c.name!r}", True
            except Exception as ex:
                out, err = f"{type(ex).__name__}: {' '.join(str(ex).split())[:400]}", True
            res.store_ms += (time.perf_counter() - t0) * 1000.0
            res.tool_errors += err
            cut = "truncated" in out[-80:]
            res.transcript.append({"tool": c.name, "input": _brief(c.input, 600),
                                   "result": _brief(out, 600 if err else 300), "error": err,
                                   "chars": len(out), "cut": cut, **seen})
            results.append({"type": "tool_result", "tool_use_id": c.id, "content": out, "is_error": err})
        messages.append({"role": "user", "content": results})
    res.stop = "max_turns"
    return res


# ----------------------------------------------------------------------------- scoring

def _norm(v: Any) -> str:
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return str(v).strip()


def score(rec: dict, res: AgentResult) -> dict:
    """Outcome and set-level precision, recall and F1 against the gold answer."""
    at, gold = rec["answer_type"], rec["gold"]
    out = {"outcome": "no_answer", "precision": None, "recall": None, "f1": None}
    if not res.answered:
        return out
    if res.kind == "abstain":
        out["outcome"] = "abstain_unexpected"
        return out
    if at == "scalar":
        want = {"number"}
        got_ok = res.kind in want and res.number is not None
        if not got_ok:
            out["outcome"] = "wrong"
            return out
        ok = math.isclose(float(res.number), float(gold), rel_tol=0, abs_tol=1e-9)
        out.update(outcome="correct" if ok else "wrong")
        return out
    if res.kind not in ("set", "ranked"):
        out["outcome"] = "wrong"
        return out
    got = [_norm(v) for v in res.values]
    want = [_norm(v) for v in gold]
    gs, ws = set(got), set(want)
    tp = len(gs & ws)
    p = tp / len(gs) if gs else 0.0
    r = tp / len(ws) if ws else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    exact = (got == want) if at == "ranked" else (gs == ws and len(got) == len(gs))
    out.update(outcome="correct" if exact else "wrong", precision=round(p, 4), recall=round(r, 4),
               f1=round(f1, 4))
    return out


# ----------------------------------------------------------------------------- one question, report

def run_one(rec: dict, llm: AgentLLM, stores: Stores, system: str, tools: list[dict], *, arm: str,
            cap: int, max_turns: int, max_tokens: int = DEFAULT_MAX_TOKENS, show_rows: int = SHOW_ROWS,
            show_chars: int = SHOW_CHARS) -> dict:
    t0 = time.perf_counter()
    res = run_agent(rec["text"], llm, stores, system, tools, arm=arm, cap=cap, max_turns=max_turns,
                    max_tokens=max_tokens, show_rows=show_rows, show_chars=show_chars)
    row = {"question_id": rec["question_id"], "template": rec["template"], "class": rec.get("class"),
           "split": rec["split"], "stratum": rec.get("stratum"), "text": rec["text"], "arm": arm,
           **score(rec, res),
           "answer_kind": res.kind, "stop": res.stop, "reason": res.reason, "turns": res.turns,
           "tool_calls": res.tool_calls, "tool_errors": res.tool_errors,
           "usage": {"prompt_tokens": res.prompt_tokens, "reply_tokens": res.reply_tokens,
                     "calls": res.turns, "seconds": round(res.llm_seconds, 2)},
           "rows_read": res.rows_read, "store_ms": round(res.store_ms, 1),
           "total_s": round(time.perf_counter() - t0, 2)}
    if res.infra:
        row["infra"] = True                                   # not a result of the method: redone on resume
    if rec["split"] == "dev":
        row["transcript"] = res.transcript
        if row["outcome"] == "wrong":
            row["got"] = res.number if rec["answer_type"] == "scalar" else res.values[:20]
    return row


def summarise(rows: list[dict]) -> str:
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by[r["template"]].append(r)
    lines = [f"{'template':8s} {'n':>3s} {'correct':>7s} {'wrong':>5s} {'abstain':>7s} {'no_ans':>6s} "
             f"{'mean_f1':>7s} {'turns':>5s} {'calls':>5s} {'tool_err':>8s} {'tok_in':>8s} {'tok_out':>7s} {'sec':>6s}"]

    def line(name, rs):
        n = len(rs)
        c = lambda f: sum(1 for r in rs if f(r))
        f1s = [r["f1"] for r in rs if r["f1"] is not None]
        m = lambda k: sum(r[k] for r in rs) / n
        return (f"{name:8s} {n:3d} {c(lambda r: r['outcome'] == 'correct'):7d} {c(lambda r: r['outcome'] == 'wrong'):5d} "
                f"{c(lambda r: r['outcome'] == 'abstain_unexpected'):7d} {c(lambda r: r['outcome'] == 'no_answer'):6d} "
                f"{(statistics.mean(f1s) if f1s else 0.0):7.3f} {m('turns'):5.1f} {m('tool_calls'):5.1f} "
                f"{m('tool_errors'):8.1f} {sum(r['usage']['prompt_tokens'] for r in rs):8d} "
                f"{sum(r['usage']['reply_tokens'] for r in rs):7d} {m('total_s'):6.1f}")

    for t in sorted(by):
        lines.append(line(t, by[t]))
    lines.append(line("all", rows))
    return "\n".join(lines)


# ----------------------------------------------------------------------------- main

def main(argv: list[str] | None = None, llm: AgentLLM | None = None, stores: Stores | None = None,
         desc: Mapping | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--templates")
    ap.add_argument("--ids")
    ap.add_argument("--limit", type=int, default=0, help="at most this many questions per template")
    ap.add_argument("--arm", choices=ARMS, default="plain")
    ap.add_argument("--cap", type=int, default=CAP)
    ap.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    ap.add_argument("--show-rows", type=int, default=SHOW_ROWS,
                    help="rows of a result shown to the model (default 50, the answer cap of the benchmark)")
    ap.add_argument("--show-chars", type=int, default=SHOW_CHARS,
                    help="characters of a tool result shown to the model (default 5000)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run: keep the rows already in --out, redo rows that failed "
                         "because the model API was unreachable, and write the file after every question")
    args = ap.parse_args(argv)

    recs = [json.loads(l) for l in Path(args.questions).read_text().splitlines() if l.strip()]
    if args.templates:
        recs = [r for r in recs if r["template"] in set(args.templates.split(","))]
    if args.ids:
        recs = [r for r in recs if r["question_id"] in set(args.ids.split(","))]
    if args.limit:
        seen: dict[str, int] = defaultdict(int)
        kept = []
        for r in recs:
            if seen[r["template"]] < args.limit:
                kept.append(r)
                seen[r["template"]] += 1
        recs = kept
    if not recs:
        print("no questions selected", file=sys.stderr)
        return 2

    if desc is None:
        from heterorag.federation.plan import Store
        from heterorag.layer1.poc_descriptors import (CONTENT_SERVICE_DESCRIPTOR, KNOWLEDGE_GRAPH_DESCRIPTOR,
                                                      USER_ACTIVITY_DESCRIPTOR)
        desc = {Store.SQL: USER_ACTIVITY_DESCRIPTOR, Store.GRAPH: KNOWLEDGE_GRAPH_DESCRIPTOR,
                Store.DOC: CONTENT_SERVICE_DESCRIPTOR}
    llm = llm or AnthropicAgent()
    stores = stores or LiveStores()
    system = build_system_prompt(desc, args.arm, args.cap, args.max_turns, show_rows=args.show_rows)
    tools = tool_definitions(args.arm, args.cap)
    print(f"agent model: {getattr(llm, 'model', '?')}   arm: {args.arm}   questions: {len(recs)}   "
          f"max turns: {args.max_turns}   cap: {args.cap}   shown: {args.show_rows} rows / {args.show_chars} chars", flush=True)

    done = resume.load(args.out) if args.resume else {}
    if done:
        print(f"resuming: {len(done)} of {len(recs)} questions already have a result in {args.out}", flush=True)
    try:
        from importlib.metadata import version
        sdk = version("anthropic")
    except Exception:
        sdk = None

    def make_meta(rows):
        return {"model": getattr(llm, "model", None), "models_returned": sorted(getattr(llm, "models_seen", []) or []),
                "arm": args.arm, "cap": args.cap, "max_turns": args.max_turns, "show_rows": args.show_rows,
                "show_chars": args.show_chars, "anthropic_sdk": sdk,
                "questions_file": args.questions, "n": len(rows), "resumed_rows": len(done)}

    guard, setup, rows = resume.Guard(), resume.SetupGuard(), []
    for i, rec in enumerate(recs, 1):
        if rec["question_id"] in done:
            rows.append(done[rec["question_id"]])
            continue
        try:
            row = run_one(rec, llm, stores, system, tools, arm=args.arm, cap=args.cap, max_turns=args.max_turns,
                          show_rows=args.show_rows, show_chars=args.show_chars)
        except Exception as ex:
            row = {"question_id": rec["question_id"], "template": rec["template"], "class": rec.get("class"),
                   "split": rec["split"], "stratum": rec.get("stratum"), "text": rec["text"], "arm": args.arm,
                   "outcome": "no_answer", "precision": None, "recall": None, "f1": None, "answer_kind": None,
                   "stop": "error", "reason": f"crash: {type(ex).__name__}: {ex}", "turns": 0, "tool_calls": 0,
                   "tool_errors": 0, "usage": {"prompt_tokens": 0, "reply_tokens": 0, "calls": 0, "seconds": 0.0},
                   "rows_read": 0, "store_ms": 0.0, "total_s": 0.0}
            if resume.is_infra_exception(ex):
                row["infra"] = True
        rows.append(row)
        tag = "  [infrastructure failure: not a result]" if row.get("infra") else ""
        print(f"[{i}/{len(recs)}] {row['question_id']:9s} {row['outcome']:18s} turns={row['turns']} "
              f"calls={row['tool_calls']} stop={row['stop']}{tag}", flush=True)
        if args.resume or row.get("infra"):
            resume.save(args.out, make_meta(rows), rows, complete=False)
        if guard.add(row):
            kept = [r for r in rows if not r.get("infra")]
            resume.save(args.out, make_meta(kept), kept, complete=False)
            print(f"\nstopped: {guard.n} infrastructure failures in a row (the model API is unreachable). "
                  f"{len(kept)} of {len(recs)} results are saved in {args.out}; start again with --resume.",
                  file=sys.stderr)
            return resume.EXIT_INTERRUPTED
        if setup.add(row):
            kept = [r for r in rows[:-setup.n] if not r.get("infra")]
            resume.save(args.out, make_meta(kept), kept, complete=False)
            print(setup.stop_message(rows, len(recs), args.out), file=sys.stderr)
            return resume.EXIT_SETUP

    finished = resume.is_complete(rows, len(recs))
    resume.save(args.out, make_meta(rows), rows, complete=finished)
    if not finished:
        print(f"\n{sum(1 for r in rows if r.get('infra'))} question(s) failed because the model API was unreachable; "
              f"start again with --resume.", file=sys.stderr)
    print("\n" + summarise(rows))
    print(f"\nrows written to {args.out}")
    return 0 if finished else resume.EXIT_INTERRUPTED


if __name__ == "__main__":
    sys.exit(main())
