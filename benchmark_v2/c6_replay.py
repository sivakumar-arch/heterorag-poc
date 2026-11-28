"""C6 replay on recorded dev queries, and the registered keep-or-drop rule (Batch 107; build item 4).

For every row of the given result files, the plan that ran (``executed_plan``, else ``plan``) is rebuilt and the C6
empty-bind probe (``heterorag.federation.c6``) is run on the recorded final query of every in-scope step, against the
live stores. Rows are classed as ``dev_pass`` classes them (correct, silent, other). Runs with the checks off made one
attempt per step, so the final query is the only one there is; ON runs probe every attempt of the repair loop at run
time (decision 3 of 2026-11-28), which this replay does not reproduce.

The positive control runs first, through the same probe; if it fails the replay stops (exit 3) and records why.

Rule (registered, accepted 2026-11-15): on the three ladder rungs pooled, C6 is dropped from F if it fires on more than
2 of their correct rows, or if more than 5% of their probes are skipped; otherwise it stays. Files of other models
(Gemma, for example) are reported beside the rule and do not enter it. No tuning of the probe follows from the replay.
Firings on silent rows are in-sample and descriptive.

Read-only on the stores and on the result files; writes one JSON report.

    python -m benchmark_v2.c6_replay data/t0_qwen_cost_1.json data/t0_gptoss_cost_1.json \\
        data/t0_mistral_cost_1.json data/t0_gemma4_cost_1.json --out data/c6_replay_t0.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from benchmark_v2.fa_tally import attribution, repeated_files
from benchmark_v2.replay_checks import outcome_class
from heterorag.federation import Plan
from heterorag.federation import c6 as c6mod

MAX_CORRECT_FIRINGS = 2        # registered: more than 2 correct ladder rows with a firing drops C6
MAX_SKIPPED_SHARE = 0.05       # registered: more than 5% of the ladder probes skipped drops C6
EXIT_CONTROL = 3


def _plan(row: dict) -> Plan | None:
    d = row.get("executed_plan") or row.get("plan")
    if not d:
        return None
    try:
        return Plan.from_dict(d if "steps" in d else {"steps": d}, trust_origin=True)
    except Exception:
        return None


def replay_rows(rows: list[dict], clients: dict) -> dict:
    """Per class: rows, rows with a firing, rows not scored (no plan or no query for an in-scope step); probes,
    firings and skips per store; and the detail of every firing and skip."""
    out = {"rows": Counter(), "fired_rows": Counter(), "unscored": Counter(), "probes": Counter(),
           "fired": Counter(), "skipped": Counter(), "detail": []}
    for r in rows:
        cls = outcome_class(r)
        out["rows"][cls] += 1
        plan, queries = _plan(r), r.get("queries") or {}
        if plan is None:
            out["unscored"][cls] += 1
            continue
        steps = [s for s in plan.steps if c6mod.in_scope(s)]
        if any(s.id not in queries for s in steps):
            out["unscored"][cls] += 1
            continue
        fired = False
        for s in steps:
            client = clients.get(s.store)
            store = s.store.value
            out["probes"][store] += 1
            if client is None:
                out["skipped"][store] += 1
                out["detail"].append({"question_id": r["question_id"], "class": cls, "step": s.id, "store": store,
                                      "status": c6mod.SKIPPED, "reason": "no client for the store"})
                continue
            pr = c6mod.probe(client, s, queries[s.id])
            if pr.status == c6mod.FIRED:
                fired = True
                out["fired"][store] += 1
            elif pr.status == c6mod.SKIPPED:
                out["skipped"][store] += 1
            if pr.status in (c6mod.FIRED, c6mod.SKIPPED):
                out["detail"].append({"question_id": r["question_id"], "class": cls, "step": s.id, "store": store,
                                      "status": pr.status, "reason": pr.reason})
        out["fired_rows"][cls] += fired
    return out


def rule(ladder: list[dict]) -> dict:
    """The keep-or-drop rule over the ladder files pooled."""
    correct = sum(r["rows"]["correct"] for r in ladder)
    fired = sum(r["fired_rows"]["correct"] for r in ladder)
    probes = sum(sum(r["probes"].values()) for r in ladder)
    skipped = sum(sum(r["skipped"].values()) for r in ladder)
    share = skipped / probes if probes else 0.0
    reasons = []
    if fired > MAX_CORRECT_FIRINGS:
        reasons.append(f"fires on {fired} of {correct} correct ladder rows (more than {MAX_CORRECT_FIRINGS})")
    if share > MAX_SKIPPED_SHARE:
        reasons.append(f"{skipped} of {probes} ladder probes skipped ({share:.1%}, more than {MAX_SKIPPED_SHARE:.0%})")
    if not ladder:
        reasons.append("no ladder file was given")
    return {"correct_rows": correct, "correct_rows_fired": fired, "probes": probes, "skipped": skipped,
            "skipped_share": round(share, 4), "verdict": "drop" if reasons else "keep", "reasons": reasons}


def _plain(c: Counter) -> dict:
    return dict(sorted(c.items()))


def main(argv: list[str] | None = None, clients: dict | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("files", nargs="+", help="result files (JSON with meta and rows)")
    ap.add_argument("--out", required=True, help="JSON report")
    args = ap.parse_args(argv)
    repeats = repeated_files(args.files)
    if repeats:
        for m in repeats:
            print(f"refused, a file is given twice: {m}", file=sys.stderr)
        return 2
    data = {}
    for f in args.files:
        try:
            data[f] = json.loads(Path(f).read_text())
        except (OSError, ValueError) as ex:
            print(f"{f}: cannot read ({type(ex).__name__}: {ex})", file=sys.stderr)
            return 2
    if clients is None:
        from heterorag.federation.clients import live_clients
        clients = live_clients()
    control = c6mod.control(clients)
    report: dict = {"control": control, "files": {}, "rule": None}
    print(f"C6 positive control: {'ok' if control['ok'] else 'FAILED'} "
          f"{ {k: v.get('why') or 'ok' for k, v in control['stores'].items()} }")
    if not control["ok"]:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"stopped: the positive control failed; nothing was replayed (recorded in {args.out})", file=sys.stderr)
        return EXIT_CONTROL
    ladder = []
    for f, d in data.items():
        group, model = attribution(d.get("meta") or {})
        res = replay_rows(d.get("rows") or [], clients)
        if group == "ladder":
            ladder.append(res)
        entry = {"group": group, "model": model, "rows": _plain(res["rows"]), "fired_rows": _plain(res["fired_rows"]),
                 "unscored": _plain(res["unscored"]), "probes": _plain(res["probes"]), "fired": _plain(res["fired"]),
                 "skipped": _plain(res["skipped"]), "detail": res["detail"]}
        report["files"][f] = entry
        print(f"{f}: {model or '?'} ({group}{', in the rule' if group == 'ladder' else ', beside the rule'})")
        print(f"    rows {entry['rows']}; rows with a firing {entry['fired_rows']}; not scored {entry['unscored']}")
        print(f"    probes {entry['probes']}; firings {entry['fired']}; skipped {entry['skipped']}")
        for x in res["detail"]:
            print(f"    {x['status']:8s} {x['question_id']:12s} {x['class']:8s} step {x['step']} ({x['store']}): "
                  f"{x['reason'][:160]}")
    report["rule"] = rule(ladder)
    r = report["rule"]
    print(f"rule (ladder pooled): {r['correct_rows_fired']} of {r['correct_rows']} correct rows with a firing; "
          f"{r['skipped']} of {r['probes']} probes skipped ({r['skipped_share']:.1%}); verdict: {r['verdict'].upper()}"
          + (f" ({'; '.join(r['reasons'])})" if r["reasons"] else ""))
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"report written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
