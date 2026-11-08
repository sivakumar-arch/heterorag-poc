"""
benchmark_v2/text_answer_sensitivity.py
=======================================
Sensitivity of an agent arm to the strictness of the text-final reader. The registered reader
(`agent_baseline.parse_text_answer`) accepts a reply only if the WHOLE reply is one valid JSON object with the
keys of the final_answer tool. A model that writes `final_answer:` followed by an object that is not valid JSON
(a set written with braces, an invented key such as `identifiers`) or wraps it in prose is not read, and the run
ends as `no_tool` with no answer.

This script re-reads the last text reply of every `no_tool` run with a tolerant rule and scores the result
against the gold answer with the same scorer as the run. Nothing is written to the result files and the
registered scores are NOT replaced: the tolerant numbers are a sensitivity analysis, reported beside them.

Tolerant rule (fixed here before any Mistral number was read from it): the reply must contain `final_answer`
(any case, with an underscore or a space); the text between the first `{` after it and the last `}` is the
payload; key names (kind, values, number, identifiers, ids, reason) are dropped; for a scalar question the
payload must hold exactly one number; for a set question every quoted string and every number is a value; a
payload that names abstain is an abstention. A reply that states an answer in prose without `final_answer`
is not read.

    python -m benchmark_v2.text_answer_sensitivity --file data/floor_mistral_small32_24b_plain_text_1.json \
        --questions data/questions_v2_dev_classD_cp1b.final.jsonl \
        --pipeline data/floor_mistral_small32_24b_cost_1.json [--show]
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

from benchmark_v2.agent_baseline import AgentResult, score
from benchmark_v2.floor_compare import SEED, paired_bootstrap, pct

_KEYS = re.compile(r"""["']?\b(kind|values|number|identifiers|ids|reason|answer)\b["']?\s*:""", re.I)
_DROP_PAIR = re.compile(r"""["']?\b(kind|reason)\b["']?\s*:\s*(?:"[^"]*"|'[^']*'|[\w-]+)""", re.I)
_TOKEN = re.compile(r'"([^"]*)"|\'([^\']*)\'|(-?\d+(?:\.\d+)?)')


def tolerant_answer(text: str, answer_type: str) -> AgentResult | None:
    m = re.search(r"final[_ ]answer", text or "", re.I)
    if not m:
        return None
    tail = text[m.end():]
    lb, rb = tail.find("{"), tail.rfind("}")
    if lb < 0 or rb < lb:
        return None
    body = tail[lb + 1:rb]
    if re.search(r"abstain", body, re.I):
        return AgentResult(stop="answered", kind="abstain")
    body = _DROP_PAIR.sub(" ", body)
    clean = _KEYS.sub(" ", body)
    toks = [(a or b or c).strip() for a, b, c in _TOKEN.findall(clean)]
    toks = [t for t in toks if t != ""]
    if answer_type == "scalar":
        nums = re.findall(r"-?\d+(?:\.\d+)?", clean)
        if len(nums) != 1:
            return None
        return AgentResult(stop="answered", kind="number", number=float(nums[0]))
    return AgentResult(stop="answered", kind="set", values=toks)


def last_text(row: dict) -> str:
    for t in reversed(row.get("transcript") or []):
        if t.get("tool") == "_text" and str(t.get("text") or "").strip():
            return str(t["text"])
    return ""


def reread(rows: list[dict], questions: dict[str, dict]) -> dict[str, dict]:
    """question id -> {"result": outcome of the tolerant reading, "text": the reply} for `no_tool` runs whose
    last text reply could be read."""
    out = {}
    for r in rows:
        if r.get("stop") != "no_tool" or r.get("outcome") != "no_answer":
            continue
        rec = questions.get(r["question_id"])
        if rec is None:
            continue
        txt = last_text(r)
        res = tolerant_answer(txt, rec["answer_type"])
        if res is None:
            continue
        out[r["question_id"]] = {"outcome": score(rec, res)["outcome"], "text": txt,
                                 "values": res.values if res.kind == "set" else res.number}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", required=True, help="an agent result file")
    ap.add_argument("--questions", required=True, help="the question file with the gold answers (dev split)")
    ap.add_argument("--pipeline", help="the pipeline file of the same model, for the paired gap")
    ap.add_argument("--show", action="store_true", help="print each tolerant reading")
    args = ap.parse_args(argv)
    questions = {j["question_id"]: j for j in (json.loads(l) for l in Path(args.questions).read_text().splitlines() if l.strip())}
    rows = json.loads(Path(args.file).read_text())["rows"]
    n = len(rows)
    base = Counter("correct" if r["outcome"] == "correct" else "wrong" if r["outcome"] == "wrong" else "other" for r in rows)
    notool = [r for r in rows if r.get("stop") == "no_tool" and r.get("outcome") == "no_answer"]
    got = reread(rows, questions)
    c = Counter(v["outcome"] for v in got.values())
    print(f"{Path(args.file).name}: {n} runs; registered reading: correct {base['correct']}, wrong {base['wrong']}, "
          f"no answer or abstained {base['other']}")
    print(f"  no_tool runs: {len(notool)}; last text reply readable under the tolerant rule: {len(got)}; "
          f"of those: would be correct {c['correct']}, wrong {c['wrong']}, abstain {c['abstain_unexpected']}")
    adj_correct = base["correct"] + c["correct"]
    adj_wrong = base["wrong"] + c["wrong"]
    print(f"  tolerant reading: correct {pct(adj_correct, n)}; wrong answers {pct(adj_wrong, n)}")
    if args.pipeline:
        prows = json.loads(Path(args.pipeline).read_text())["rows"]
        pc = {r["question_id"]: float(r["outcome"] == "correct") for r in prows}
        ids = sorted(pc)
        reg = {r["question_id"]: float(r["outcome"] == "correct") for r in rows}
        tol = {q: max(reg[q], float(got.get(q, {}).get("outcome") == "correct")) for q in ids}
        rng = random.Random(SEED)
        for label, vec in (("registered", reg), ("tolerant", tol)):
            p, lo, hi = paired_bootstrap(pc, vec, ids, rng)
            print(f"  pipeline minus this arm, {label}: {p:+.1f} points [{lo:+.1f}, {hi:+.1f}]")
    if args.show:
        for q, v in sorted(got.items()):
            print(f"\n{q}  tolerant: {v['outcome']}  read: {v['values']}\n    {' '.join(v['text'].split())[:240]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
