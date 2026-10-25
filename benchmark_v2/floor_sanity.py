"""
benchmark_v2/floor_sanity.py
============================
Plumbing check for one capability-floor model before its full run (novelty gate G0, rule L). It reads the
three small result files of a sanity run (pipeline, python agent, plain agent; a few questions each) and
reports PASS / WARN / FAIL / INFO per check. The checks are fixed here, before any model is run, and they test
whether the run can be trusted as a run (the right model answered, nothing crashed, tool calls were read,
reasoning was handled), never how accurate the model is. A sanity result is not a result: its files are not
kept as evidence, it is never used to choose or drop a model, and a FAIL leads to a fix of the plumbing,
disclosed in the log, not to a change of a prompt.

A FAIL means the run cannot be trusted until its cause has been read from the files (plan_errors, plan_raw and
the last ``_text`` of an agent question). The cause is either plumbing, which is fixed and disclosed, or the
model, which cannot meet the budgets every model runs under; that is recorded as a floor result with its
evidence and is never dropped silently. Two checks exist because a first version passed a run in which the
model reasoned in its visible replies (the setting for reasoning off was not applied by the server) and no
question got a plan or an answer: reasoning markup in replies, and a stage in which nothing worked at all.

    python -m benchmark_v2.floor_sanity --model qwen3:30b --cost A.json --python B.json --plain C.json

Exit status 0 if there is no FAIL, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

FULL_QUESTIONS = 102
RETRY_SHARE_WARN = 0.25          # questions with at least one HTTP retry
NO_ANSWER_WARN = 0.5             # agent questions without any answer
EMPTY_REPLY_FAIL = 0.25          # share of pipeline questions with an empty model reply that stops the run
PROJECTION_SPREAD = (1.0, 4.0)   # a sample of easy questions underestimates the full run (gpt-oss:20b: 3.7 times)
MIN_ROWS_FOR_FAIL = 4            # a stage in which nothing worked fails only when the sample is large enough to say so
PLAN_WARN = 0.5                  # share of questions with a valid plan below which the sample is flagged


def reply_shape(text: str) -> str:
    """How a planner reply is built: a bare object, prose before an object, prose only, cut off, or empty."""
    t = (text or "").strip()
    if not t:
        return "empty"
    if t.count("{") > t.count("}"):
        return "unbalanced braces (cut off)"
    if t.startswith("{") or t.startswith("```"):
        return "object"
    if t.endswith("}") or t.endswith("```"):
        return "prose then object"
    return "prose only"


def has_think_markup(x) -> bool:
    """A think tag, opening or closing, in visible text (a closing tag alone is what a template that opens the
    reasoning block itself leaves behind)."""
    s = json.dumps(x, default=str).lower()
    return "<think" in s or "</think" in s


def _agent_texts(rows: list[dict]) -> list[str]:
    return [str(t.get("text") or "") for r in rows for t in (r.get("transcript") or []) if t.get("tool") == "_text"]


def last_reply_kind(row: dict) -> str:
    """What the last reply without a tool call of an unanswered question looks like."""
    texts = [str(t.get("text") or "") for t in (row.get("transcript") or []) if t.get("tool") == "_text"]
    if not texts:
        return "no text reply"
    last = texts[-1]
    if not last.strip():
        return "empty"
    if "final_answer" in last:
        return "final_answer written as markup"
    if has_think_markup(last):
        return "reasoning markup"
    return "prose"


def _rows(d: dict) -> list[dict]:
    return d.get("rows") or []


def _meta(d: dict) -> dict:
    return d.get("meta") or {}


def _names(meta: dict, *keys: str) -> list[str]:
    out: list[str] = []
    for k in keys:
        v = meta.get(k)
        if isinstance(v, (list, tuple, set)):
            out.extend(str(x) for x in v)
        elif v:
            out.append(str(v))
    return out


def _sum_adapter(rows: list[dict], key: str) -> int:
    return sum(int((r.get("adapter") or {}).get(key, 0)) for r in rows)


def check(model: str, cost: dict, python: dict, plain: dict) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    add = lambda status, name, detail: out.append((status, name, detail))
    systems = {"pipeline": cost, "python": python, "plain": plain}

    # complete, no infrastructure rows, no crash of our own code
    for name, d in systems.items():
        rows = _rows(d)
        bad_infra = sum(1 for r in rows if r.get("infra"))
        crashes = sum(1 for r in rows if str(r.get("reason") or "").startswith("crash:"))
        if not rows:
            add("FAIL", f"{name}: rows", "the file has no rows")
        elif not _meta(d).get("complete") or bad_infra or crashes:
            add("FAIL", f"{name}: finished cleanly",
                f"complete={_meta(d).get('complete')} infrastructure rows={bad_infra} crashes={crashes}")
        else:
            add("PASS", f"{name}: finished cleanly", f"{len(rows)} rows, no infrastructure row, no crash")

    # the model that answered is the model asked for
    for name, d in systems.items():
        meta = _meta(d)
        seen = _names(meta, "model", "planner_model", "translator_model", "models_returned")
        ok = bool(seen) and all(model in s for s in seen)
        add("PASS" if ok else "FAIL", f"{name}: model identity", f"asked {model}; recorded {sorted(set(seen))}")

    # pipeline: the model replied, no reasoning markup leaked into a plan or a query
    prow = _rows(cost)
    silent = [r["question_id"] for r in prow
              if not (((r.get("usage") or {}).get("planner") or {}).get("calls", 0) >= 1
                      and ((r.get("usage") or {}).get("planner") or {}).get("reply_tokens", 0) > 0)]
    add("FAIL" if silent else "PASS", "pipeline: planner replied", f"questions without a planner reply: {silent or 'none'}")
    leaked = [r["question_id"] for r in prow if has_think_markup([r.get("plan_raw"), r.get("translation_raw")])]
    add("FAIL" if leaked else "PASS", "pipeline: no reasoning markup in plan or query", f"questions with a think tag: {leaked or 'none'}")
    p_off = str(_meta(cost).get("ollama_think")).lower() == "false"
    shown = Counter(reply_shape(x) for r in prow for x in (r.get("plan_raw") or []))
    prose = shown["prose only"] + shown["prose then object"] + shown["unbalanced braces (cut off)"]
    if p_off and prose:
        add("WARN", "pipeline: reasoning off was requested", f"{prose} of {sum(shown.values())} planner replies are not a bare "
            "object; check with a direct request to the server that the setting is applied for this model")
    empty_plans = [r["question_id"] for r in prow if any(not (x or "").strip() for x in (r.get("plan_raw") or []))]
    empty_tr = [r["question_id"] for r in prow
                if any(not (x or "").strip() for xs in (r.get("translation_raw") or {}).values() for x in xs)]
    n_empty = len(set(empty_plans) | set(empty_tr))
    add("FAIL" if prow and n_empty / len(prow) > EMPTY_REPLY_FAIL else ("WARN" if n_empty else "PASS"),
        "pipeline: no empty replies",
        f"questions with an empty planner or translator reply: {n_empty} of {len(prow)} (a model that reasons "
        f"at length can spend the whole token budget before it writes the answer; see HETERORAG_OLLAMA_THINK)")
    executed = sum(1 for r in prow if r.get("native_calls"))
    add("PASS" if executed else "WARN", "pipeline: queries reached the stores", f"{executed} of {len(prow)} questions ran at least one native query")
    shapes = Counter(reply_shape(x) for r in prow for x in (r.get("plan_raw") or []))
    cut = shapes["unbalanced braces (cut off)"]
    add("WARN" if cut else "INFO", "pipeline: how planner replies are built",
        f"{dict(shapes)} over all attempts (a model that writes its reasoning into the reply can use the whole "
        f"planner budget before it writes the plan)")
    c = Counter(("plan" if r.get("plan_valid") else ("abstain" if r.get("plan_abstain") == "out_of_scope" else "invalid"))
                for r in prow)
    detail = f"valid {c['plan']}, deliberate abstain {c['abstain']}, invalid {c['invalid']} of {len(prow)}"
    if prow and len(prow) >= MIN_ROWS_FOR_FAIL and c["plan"] == 0:
        add("FAIL", "pipeline: plans", detail + ": no question got a valid plan, so nothing downstream was tested; "
            "read plan_errors and plan_raw before any full run")
    elif prow and c["plan"] / len(prow) < PLAN_WARN:
        add("WARN", "pipeline: plans", detail + ": fewer than half of the sample")
    else:
        add("INFO", "pipeline: plans", detail + " (a result, not a plumbing check)")

    # agents
    for name in ("python", "plain"):
        rows = _rows(systems[name])
        if not rows:
            continue
        no_reply = [r["question_id"] for r in rows if not r.get("turns")]
        add("FAIL" if no_reply else "PASS", f"{name}: model replied", f"questions without a reply: {no_reply or 'none'}")
        calls = sum(int(r.get("tool_calls") or 0) for r in rows)
        add("PASS" if calls else "FAIL", f"{name}: tool calls were read", f"{calls} tool calls over {len(rows)} questions")
        unread = _sum_adapter(rows, "unreadable_arguments")
        add("WARN" if unread else "PASS", f"{name}: arguments readable", f"unreadable arguments {unread}, given as text "
            f"{_sum_adapter(rows, 'arguments_as_text')}, values as text {_sum_adapter(rows, 'values_as_text')}")
        add("INFO", f"{name}: reasoning handled", f"turns with reasoning sent back {_sum_adapter(rows, 'thinking_turns')} of "
            f"{sum(int(r.get('turns') or 0) for r in rows)} turns")
        src = Counter((r.get("answer_source") or "none") for r in rows)
        none = src["none"] / len(rows)
        detail = f"tool call {src['tool']}, text {src['text']}, none {src['none']} ({none:.0%} of {len(rows)})"
        if len(rows) >= MIN_ROWS_FOR_FAIL and src["none"] == len(rows):
            add("FAIL", f"{name}: how answers arrived", detail + ": no question was answered, so the answer path was "
                "not tested; read the last reply of a question (the line below) before any full run")
        else:
            add("WARN" if none > NO_ANSWER_WARN else "INFO", f"{name}: how answers arrived", detail)
        ended = Counter(last_reply_kind(r) for r in rows if not r.get("answer_source"))
        if ended:
            add("INFO", f"{name}: how an unanswered question ended", f"last reply was {dict(ended)}")
        marked = sum(1 for r in rows if has_think_markup(_agent_texts([r])))
        off = str(_meta(systems[name]).get("ollama_think")).lower() == "false"
        add("FAIL" if (marked and off) else ("WARN" if marked else "PASS"), f"{name}: no reasoning markup in replies",
            f"questions with a think tag in a reply: {marked} of {len(rows)}" +
            ("; reasoning off was requested, so the server or the model did not apply the setting" if marked and off else ""))
        stops = Counter(r.get("stop") or "?" for r in rows)
        cut = sum(1 for r in rows for t in (r.get("transcript") or []) if t.get("tool") == "_turn" and t.get("stop_reason") == "max_tokens")
        add("WARN" if cut else "PASS", f"{name}: replies cut at the token limit",
            f"{cut} turns ended at max_tokens; stop reasons {dict(stops)}")
        retried = sum(1 for r in rows if (r.get("adapter") or {}).get("retries"))
        add("WARN" if retried / len(rows) > RETRY_SHARE_WARN else "INFO", f"{name}: HTTP retries",
            f"{_sum_adapter(rows, 'retries')} retries in {retried} of {len(rows)} questions")
        errs = Counter(t["tool"] for r in rows for t in (r.get("transcript") or []) if t.get("error"))
        tot = Counter(t["tool"] for r in rows for t in (r.get("transcript") or []) if t.get("tool") in ("sql", "cypher", "es", "python"))
        add("INFO", f"{name}: tool errors", ", ".join(f"{t} {errs[t]}/{tot[t]}" for t in sorted(tot)) or "no tool calls")

    # settings recorded with the files
    think = {n: d.get("meta", {}).get("ollama_think", "not recorded") for n, d in systems.items()}
    add("INFO", "settings", f"reasoning mode (HETERORAG_OLLAMA_THINK) per system: {think}; unset means the model's own mode")

    # time and a projection to the full question set
    for name, d in systems.items():
        rows = _rows(d)
        if rows:
            mean = sum(float(r.get("total_s") or 0) for r in rows) / len(rows)
            lo, hi = (mean * FULL_QUESTIONS * f / 3600 for f in PROJECTION_SPREAD)
            add("INFO", f"{name}: time", f"{mean:.0f} s per question on {len(rows)}; full run of {FULL_QUESTIONS} "
                f"between {lo:.1f} and {hi:.1f} hours (the sample is easier than the whole set)")
    return out


def render(results: list[tuple[str, str, str]]) -> str:
    return "\n".join(f"{s:5s} {n}: {d}" for s, n, d in results)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="model name as Ollama knows it, e.g. qwen3:30b")
    ap.add_argument("--cost", required=True)
    ap.add_argument("--python", required=True)
    ap.add_argument("--plain", required=True)
    a = ap.parse_args(argv)
    res = check(a.model, *(json.loads(Path(p).read_text()) for p in (a.cost, a.python, a.plain)))
    print(render(res))
    fails = sum(1 for s, _, _ in res if s == "FAIL")
    warns = sum(1 for s, _, _ in res if s == "WARN")
    print(f"\n{fails} FAIL, {warns} WARN. " + ("Not ready: read the cause from the files. Plumbing is fixed and disclosed, "
          "then the sanity check is repeated; a model that cannot meet the fixed budgets is recorded as a floor result, "
          "never dropped silently." if fails else "No FAIL: read the WARN lines, then start the full run."))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
