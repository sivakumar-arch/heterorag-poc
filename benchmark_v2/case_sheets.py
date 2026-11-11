"""
benchmark_v2/case_sheets.py
===========================
Blind case sheets for the independent reader (Reader Kit, jobs 2 and 3).

    python -m benchmark_v2.case_sheets data/floor_*_off.json --questions data/questions_v2_dev_classD_cp1c.final.jsonl \
        --kind classify --out-prefix data/sheets_classify
    python -m benchmark_v2.case_sheets data/floor_*_off.json --questions ... --kind wording --correct 20 --out-prefix data/sheets_wording

Writes three files next to --out-prefix:

    <prefix>.md        the sheets, for the reader
    <prefix>_form.csv  the blank answer form, for the reader
    <prefix>.key.json  case id -> source file, question id, outcome and template: KEEP, do not send to the reader

What a sheet shows, and what it must not show. A ``classify`` sheet shows, per silent case (outcome wrong or
partial_violated): the question, the plan of the answer key and the plan that ran, for each retrieve step what it was
meant to do and the final query, the label the system gave, and the answer next to the correct answer. It does not
show the model, the file, the run, the sampling, any attempt that was rejected before the final query, any step error
or any output of the semantic checks. Result files made with the checks on are refused: their final queries were
chosen under the checks, so a sheet from them would carry the checks' effect. Cases are shuffled with the seed and
numbered anew, so the order does not leak the model or the template.

A ``wording`` sheet shows only the question and the correct answer, once per question, silent and correct cases
mixed (``--correct N`` adds N random correct ones), so that the reader cannot tell which questions went wrong.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

from benchmark_v2.floor_compare import SEED
from benchmark_v2.repeat_summary import group

STORE_NAME = {"sql": "numbers cabinet (SQL)", "graph": "connections cabinet (graph)", "doc": "text cabinet (search)"}
COMBINE = {"intersect": "keep the items found by both steps", "union": "keep the items found by either step",
           "subtract": "keep the items of the first step that are not in the second"}
SHOW = 20
CLASSIFY_FORM = ["case_id", "class", "also_applies", "confidence_1_to_5", "why_quote_the_part_of_the_query"]
WORDING_FORM = ["case_id", "ambiguous_yes_no", "other_reading", "correct_answer_looks_right_yes_no_unsure",
                "what_looks_wrong", "confidence_1_to_5"]


def load_questions(path: str) -> dict[str, dict]:
    out = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            out[rec["question_id"]] = rec
    return out


def load_rows(files: list[str]) -> list[tuple[str, dict]]:
    """(file name, row) for every row. A file made with the semantic checks on is refused."""
    out = []
    for f in files:
        data = json.loads(Path(f).read_text())
        if data.get("meta", {}).get("semantic_checks") or any(r.get("semantic_checks") for r in data["rows"]):
            raise ValueError(f"{f} was made with the semantic checks on: a blind sheet needs a run with the checks off")
        out.extend((Path(f).name, r) for r in data["rows"])
    return out


def steps_of(plan) -> list[dict]:
    return (plan.get("steps") if isinstance(plan, dict) else plan) or []


def render_plan(plan, with_queries: dict[str, str] | None = None) -> list[str]:
    steps = steps_of(plan)
    if not steps:
        return ["(no plan recorded)"]
    lines = []
    for s in steps:
        if s.get("op", "retrieve") == "retrieve":
            where = STORE_NAME.get(s.get("store"), str(s.get("store")))
            ask = s.get("question") or "(no description)"
            lines.append(f"* Step {s['id']}: ask the {where}: {ask}")
            if s.get("bind"):
                lines.append(f"  This step must look only at the ids handed over from step {s['bind']['input']}.")
            if with_queries is not None:
                lines += ["", "  ```", "  " + " ".join(str(with_queries.get(s["id"], "(no query)")).split()), "  ```"]
        else:
            ins = " and ".join(s.get("inputs") or [])
            lines.append(f"* Step {s['id']}: from {ins}: {COMBINE.get(s['op'], s['op'])}")
    return lines


def render_items(items, total: int | None = None) -> str:
    items = list(items)
    n = total if total is not None else len(items)
    head = ", ".join(str(x) for x in items[:SHOW])
    more = f" ... ({n} items in all)" if n > SHOW else f" ({n} items)"
    return (head or "none") + more if n else "none (0 items)"


def answer_block(row: dict, rec: dict) -> list[str]:
    at = rec.get("answer_type") or row.get("answer_type")
    gold = rec["gold"]
    dg = row.get("answer_digest") or {}
    lines = [f"System label: {row.get('guarantee')}   (answer type: {at})"]
    if at == "scalar":
        return lines + [f"System answer: {dg.get('value', row.get('got'))}", f"Correct answer: {gold}"]
    got = list(row.get("got") if row.get("got") is not None else dg.get("head", []))
    size = dg.get("size", len(got))
    lines += [f"System answer: {render_items(got, size)}", f"Correct answer: {render_items(sorted(gold, key=str) if at == 'ids' else gold, len(gold))}"]
    if at == "ids":
        g, w = {str(x) for x in got}, {str(x) for x in gold}
        full = len(got) >= size
        lines.append(f"In the system answer but not in the correct answer: {render_items(sorted(g - w, key=str))}"
                     + ("" if full else f"   (only the first {len(got)} of the system's {size} items are recorded)"))
        lines.append(f"In the correct answer but not in the system answer: {render_items(sorted(w - g, key=str))}"
                     + ("" if full else "   (cannot be listed in full)"))
    return lines


def classify_sheet(cid: str, row: dict, rec: dict) -> str:
    lines = [f"## Case {cid}", "", f"**Question:** {row.get('text') or rec.get('text')}", "",
             "**Plan of the answer key:**", *render_plan(row.get("gold_plan") or rec.get("plan")), "",
             "**Plan the system used, with the final query of each step:**", *render_plan(row.get("plan"), row.get("queries") or {}), "",
             "**Answer:**", "", *["    " + x for x in answer_block(row, rec)], ""]
    return "\n".join(lines)


def wording_sheet(cid: str, rec: dict) -> str:
    at = rec["answer_type"]
    gold = rec["gold"]
    shown = gold if at == "scalar" else render_items(sorted(gold, key=str) if at == "ids" else gold, len(gold))
    return "\n".join([f"## Item {cid}", "", f"**Question:** {rec['text']}", "", f"**Correct answer shown:** {shown}", ""])


def build(files: list[str], questions: dict[str, dict], kind: str, correct: int, seed: int):
    rows = load_rows(files)
    rng = random.Random(seed)
    silent = [(f, r) for f, r in rows if group(r["outcome"]) == "silent" and r["question_id"] in questions]
    if kind == "classify":
        cases = silent
    else:
        seen, cases = set(), []
        for f, r in silent:
            if r["question_id"] not in seen:
                seen.add(r["question_id"])
                cases.append((f, r))
        pool = [(f, r) for f, r in rows if r["outcome"] == "correct" and r["question_id"] in questions
                and r["question_id"] not in seen]
        by_q = {}
        for f, r in pool:
            by_q.setdefault(r["question_id"], (f, r))
        cases += rng.sample(sorted(by_q.values(), key=lambda x: x[1]["question_id"]), min(correct, len(by_q)))
    cases = sorted(cases, key=lambda x: (x[1]["question_id"], x[0]))
    rng.shuffle(cases)
    prefix = "S" if kind == "classify" else "W"
    sheets, key = [], {}
    for i, (f, r) in enumerate(cases, 1):
        cid = f"{prefix}{i:03d}"
        rec = questions[r["question_id"]]
        sheets.append(classify_sheet(cid, r, rec) if kind == "classify" else wording_sheet(cid, rec))
        key[cid] = {"file": f, "question_id": r["question_id"], "outcome": r["outcome"], "template": r.get("template")}
    return sheets, key


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="result files made with the semantic checks off")
    ap.add_argument("--questions", required=True, help="questions file with the gold answers (jsonl)")
    ap.add_argument("--kind", choices=("classify", "wording"), default="classify")
    ap.add_argument("--correct", type=int, default=0, help="wording sheets: how many correctly answered questions to mix in")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args(argv)
    try:
        sheets, key = build(args.files, load_questions(args.questions), args.kind, args.correct, args.seed)
    except (OSError, ValueError, KeyError) as ex:
        print(f"error: {ex}", file=sys.stderr)
        return 2
    p = Path(args.out_prefix)
    title = "Case sheets" if args.kind == "classify" else "Wording check"
    Path(f"{p}.md").write_text(f"# {title}\n\nRead the Reader Kit first. Do not discuss your answers with anyone until you hand them in.\n\n"
                               + "\n---\n\n".join(sheets) + "\n")
    cols = CLASSIFY_FORM if args.kind == "classify" else WORDING_FORM
    with open(f"{p}_form.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for cid in key:
            w.writerow([cid] + [""] * (len(cols) - 1))
    Path(f"{p}.key.json").write_text(json.dumps(key, indent=1))
    print(f"{len(key)} {args.kind} cases written to {p}.md, {p}_form.csv and {p}.key.json (keep the key, do not send it)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
