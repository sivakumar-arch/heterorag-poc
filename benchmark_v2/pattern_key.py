"""
benchmark_v2/pattern_key.py
===========================
The mechanical pattern key of pre-registration v1.6 (section 5). A pattern is a set of OFF-silent cases with the same
key; the key is computed from the result row and the question record, with no reader and no check output.

  plan as in the template (`plan_equal` true):
      (shape, earliest step, direction)
      The earliest step, in plan order, that is an independent retrieve step (no bind) and returned a number of ids
      different from the oracle's recorded size for its store (`intermediates` of the question record); direction
      empty (nothing returned), superset (more ids) or subset (fewer). If no such step deviates, the step is
      `final` and the direction comes from the final answer against the gold (precision and recall).
  plan different from the template:
      (shape, structure signature of the executed plan, direction of the final answer against the gold)

Limit, stated: the question record keeps the SIZE of each independent sub-result, not its identifiers, so the
comparison is by size. A fault that returns the same number of wrong ids falls to `final`. The model is not in the key.
The wording variant, (n - 1) mod 3 for question id suffix n, is printed beside it and is not in the key.

Batch 103 (approved 2026-11-26; a change to the registered definition, recorded for v1.8). Question records written
from generator 2.3 on carry `step_sizes` and `step_hashes` by step id (the oracle's exact set of every retrieve step,
as its size and a hash of its sorted ids), and run step records carry `value_hash` (same hash). For such a record the
earliest-step rule compares by step, not by store: the executed plan is aligned with the template plan from the
final step backwards (inputs of an intersection or union paired by structure; two inputs of the same structure are
left unaligned), and an independent executed step deviates if its size differs from the aligned template step's
size (direction empty, superset or subset by size, as before) or, at equal size, if both hashes exist and differ
(direction `other`: the same number of different ids). Two reasons: four first-contact shapes have two independent
steps on one store, which the store lookup cannot tell apart; and the hash removes the stated limit for equal-size
faults. Records without `step_sizes` (every dev record) take the store lookup above, unchanged. `superset` and
`subset` still mean more and fewer ids: a hash cannot show containment.

    python -m benchmark_v2.pattern_key data/t0_*_cost_1.json --questions data/cp1b_questions.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

SILENT = ("wrong", "partial_violated")


def steps_of(plan) -> list[dict]:
    return (plan.get("steps") if isinstance(plan, dict) else plan) or []


def final_direction(row: dict) -> str:
    dig = row.get("answer_digest") or {}
    if dig.get("type") == "ids" and dig.get("size") == 0:
        return "empty"
    p, r = row.get("precision"), row.get("recall")
    if p is None or r is None:
        return "other"
    if p < 1 and r >= 1:
        return "superset"
    if p >= 1 and r < 1:
        return "subset"
    return "other"


def signature(plan) -> str:
    return "|".join(f"{s.get('op')}:{s.get('store') or '-'}:{s.get('output')}:{'bound' if s.get('bind') else 'free'}"
                    for s in steps_of(plan))


def wording_variant(question_id: str) -> int:
    return (int(question_id.rsplit("-", 1)[1]) - 1) % 3


def _shape(step: dict) -> tuple:
    return (step.get("op"), step.get("store"), step.get("mode"), step.get("output"), step.get("kind", "set"))


def align(executed, template) -> dict[str, str]:
    """executed step id -> template step id, walking both plans from their final steps; binds are ignored (the
    bind planner adds restrictions the template does not have); inputs of an intersection or union are paired by
    structure, and inputs that share a structure are not aligned."""
    ex, tp = steps_of(executed), steps_of(template)
    ex_by, tp_by = {s["id"]: s for s in ex}, {s["id"]: s for s in tp}

    def sink(steps):
        used = {i for s in steps for i in s.get("inputs") or ()}
        last = [s["id"] for s in steps if s["id"] not in used]
        return last[0] if len(last) == 1 else None

    def struct(sid, by, memo):
        if sid not in memo:
            st = by[sid]
            memo[sid] = (_shape(st), tuple(struct(i, by, memo) for i in st.get("inputs") or ()))
        return memo[sid]

    out: dict[str, str] = {}
    me, mt = {}, {}

    def walk(e, t):
        if e not in ex_by or t not in tp_by or _shape(ex_by[e]) != _shape(tp_by[t]):
            return
        out[e] = t
        ei, ti = list(ex_by[e].get("inputs") or ()), list(tp_by[t].get("inputs") or ())
        if len(ei) != len(ti):
            return
        if ex_by[e].get("op") in ("intersect", "union"):
            ks = [struct(i, ex_by, me) for i in ei]
            kt = [struct(i, tp_by, mt) for i in ti]
            if len(set(ks)) != len(ks) or sorted(ks, key=repr) != sorted(kt, key=repr):
                return
            ei = [i for _, i in sorted(zip(ks, ei), key=lambda x: repr(x[0]))]
            ti = [i for _, i in sorted(zip(kt, ti), key=lambda x: repr(x[0]))]
        for a, b in zip(ei, ti):
            walk(a, b)

    se, st = sink(ex), sink(tp)
    if se and st:
        walk(se, st)
    return out


def _by_step(row: dict, qrec: dict) -> tuple | None:
    sizes, hashes = qrec.get("step_sizes") or {}, qrec.get("step_hashes") or {}
    executed = steps_of(row.get("executed_plan"))
    to_tmpl = align(executed, qrec.get("plan") or [])
    by_id = {r["step"]: r for r in row.get("step_records") or []}
    for i, s in enumerate(executed):
        rec, t = by_id.get(s.get("id")), to_tmpl.get(s.get("id"))
        if s.get("op") != "retrieve" or s.get("bind") or rec is None or t not in sizes:
            continue
        size, want = rec.get("value_size"), sizes[t]
        if size is None:
            continue
        if size != want:
            return (row["template"], f"step{i + 1}", "empty" if size == 0 else ("superset" if size > want else "subset"))
        got_h, want_h = rec.get("value_hash"), hashes.get(t)
        if got_h and want_h and got_h != want_h:
            return (row["template"], f"step{i + 1}", "other")
    return None


def pattern_key(row: dict, qrec: dict | None) -> tuple:
    shape = row["template"]
    if not row.get("plan_equal"):
        return (shape, "plan:" + signature(row.get("executed_plan")), final_direction(row))
    if (qrec or {}).get("step_sizes"):
        return _by_step(row, qrec) or (shape, "final", final_direction(row))
    inter = (qrec or {}).get("intermediates") or {}
    by_id = {r["step"]: r for r in row.get("step_records") or []}
    for s in steps_of(row.get("executed_plan")):
        rec = by_id.get(s.get("id"))
        if s.get("op") != "retrieve" or s.get("bind") or rec is None or s.get("store") not in inter:
            continue
        size, want = rec.get("value_size"), inter[s["store"]]
        if size is None or size == want:
            continue
        direction = "empty" if size == 0 else ("superset" if size > want else "subset")
        return (shape, f"step{steps_of(row.get('executed_plan')).index(s) + 1}", direction)
    return (shape, "final", final_direction(row))


def load_questions(path: str) -> dict[str, dict]:
    out = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            out[rec["question_id"]] = rec
    return out


def patterns(files: list[str], questions: dict[str, dict]) -> dict[tuple, list[tuple[str, str, int]]]:
    """key -> [(file name, question id, wording variant)] over the OFF-silent rows."""
    out: dict[tuple, list] = defaultdict(list)
    for f in files:
        data = json.loads(Path(f).read_text())
        for row in data["rows"]:
            if row.get("outcome") in SILENT and not row.get("semantic_checks"):
                key = pattern_key(row, questions.get(row["question_id"]))
                out[key].append((Path(f).name, row["question_id"], wording_variant(row["question_id"])))
    return out


def render(pats: dict) -> str:
    cases = sum(len(v) for v in pats.values())
    pairs = {(k[0], v[2]) for k, vs in pats.items() for v in vs}
    lines = [f"{cases} OFF-silent cases; {len(pats)} patterns under the mechanical key; "
             f"{len(pairs)} distinct (shape, wording variant) pairs for comparison"]
    for k, vs in sorted(pats.items(), key=lambda kv: -len(kv[1])):
        files = sorted({v[0].split('_')[1] if '_' in v[0] else v[0] for v in vs})
        lines.append(f"  {len(vs):3d} cases  {k[0]:4s} {k[1][:60]:60s} {k[2]:9s} files {','.join(files)}  "
                     f"variants {sorted({v[2] for v in vs})}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--questions", required=True)
    args = ap.parse_args(argv)
    print(render(patterns(args.files, load_questions(args.questions))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
