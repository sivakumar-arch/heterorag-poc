"""
benchmark_v2/fa_tally.py
========================
Batch 100 (build item 1): recompute by script the false-alarm tally of the static checks on correctly answered
rows of saved result files (the October sweep and floor files), attributed to hosted and ladder models.

The tally recorded in October (0 of 500 correct rows; 0 of 477 attributed rows, 236 hosted and 241 ladder, 23
unattributed) came from runs of ``replay_checks`` that were not saved as a script. This script is the recorded
method: for every file it scores the correct rows with ``dev_pass.fired`` (the same detectors as the dev pass),
attributes the file by its translator model (the translator writes the queries the checks read), and reports per
file and per group:

  * correct rows, and correct rows that could be scored (a plan and final queries are recorded);
  * the rows on which S (R1 to R5, as replayed in October), R7, and S or R7 fire, with the question ids.

Attribution: ``meta.translator_model`` (or ``planner_model`` when the translator is not recorded), by model name
(Batch 101; Batch 100 grouped every ``ollama:`` model as ladder, which put the smoke model and Gemma into the
ladder total):

  * hosted:       a model beginning with ``claude``;
  * ladder:       exactly the three design rungs, ``ollama:gpt-oss:20b``, ``ollama:mistral-small3.2-nosys`` and
                  ``ollama:qwen3:30b-instruct`` (the registered ladder);
  * gemma:        a model beginning with ``ollama:gemma4`` (Gemma 4 31B; its dev rows are read-only and are
                  reported beside the rules, never inside them);
  * granite:      a model beginning with ``ollama:granite`` (Granite 4.2 30B);
  * other-local:  any other ``ollama:`` model (for example the smoke model llama3.1:8b, which is not a rung);
  * unattributed: no model recorded, or a model of no group above.

The ``attributed`` line adds hosted and ladder only, as the October tally did. The same file given twice (the same
resolved path, or identical content under another name, as when two shell patterns overlap) is refused, because it
would be counted twice. No model call and no store access; nothing is written.

    python -m benchmark_v2.fa_tally data/cp1b_*_1.json data/floor_*.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from benchmark_v2.dev_pass import fired
from benchmark_v2.replay_checks import outcome_class
from heterorag.federation.semantic_checks import Signatures

GROUPS = ("hosted", "ladder", "gemma", "granite", "other-local", "unattributed")
SHOWN = ("S", "R7", "S|R7")
LADDER = frozenset({"ollama:gpt-oss:20b", "ollama:mistral-small3.2-nosys", "ollama:qwen3:30b-instruct"})


def attribution(meta: dict) -> tuple[str, str | None]:
    model = meta.get("translator_model") or meta.get("planner_model")
    if not model:
        return "unattributed", None
    m = str(model)
    if m.startswith("claude"):
        return "hosted", m
    if m in LADDER:
        return "ladder", m
    if m.startswith("ollama:gemma4"):
        return "gemma", m
    if m.startswith("ollama:granite"):
        return "granite", m
    if m.startswith("ollama:"):
        return "other-local", m
    return "unattributed", m


def repeated_files(files: list[str]) -> list[str]:
    """Messages for every file that repeats an earlier one, by resolved path or by identical content."""
    seen_path: dict[Path, str] = {}
    seen_hash: dict[str, str] = {}
    out = []
    for f in files:
        rp = Path(f).resolve()
        if rp in seen_path:
            out.append(f"{f}: the same file as {seen_path[rp]}")
            continue
        seen_path[rp] = f
        try:
            digest = hashlib.sha256(rp.read_bytes()).hexdigest()
        except OSError:
            continue                    # reported as unreadable by main
        if digest in seen_hash:
            out.append(f"{f}: identical content to {seen_hash[digest]}")
        else:
            seen_hash[digest] = f
    return out


def tally_file(data: dict, sigs: Signatures) -> dict:
    rows = data.get("rows", [])
    out = {"correct": 0, "scored": 0, "fires": {d: [] for d in SHOWN}}
    for r in rows:
        if outcome_class(r) != "correct":
            continue
        out["correct"] += 1
        res, _ = fired(r, sigs)
        if res.get("S") is None:
            continue
        out["scored"] += 1
        hits = {"S": bool(res["S"]), "R7": bool(res.get("R7"))}
        hits["S|R7"] = hits["S"] or hits["R7"]
        for d in SHOWN:
            if hits[d]:
                out["fires"][d].append(r.get("question_id"))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    args = ap.parse_args(argv)
    repeats = repeated_files(args.files)
    if repeats:
        for m in repeats:
            print(f"refused, a file is given twice: {m}", file=sys.stderr)
        return 2
    sigs = Signatures.poc()
    totals = {g: Counter() for g in GROUPS}
    for f in args.files:
        try:
            data = json.loads(Path(f).read_text())
        except (OSError, ValueError) as ex:
            print(f"{f}: cannot read ({type(ex).__name__}: {ex})", file=sys.stderr)
            return 2
        group, model = attribution(data.get("meta") or {})
        t = tally_file(data, sigs)
        totals[group]["files"] += 1
        totals[group]["correct"] += t["correct"]
        totals[group]["scored"] += t["scored"]
        for d in SHOWN:
            totals[group][d] += len(t["fires"][d])
        fires = "; ".join(f"{d} {len(t['fires'][d])}" + (f" {t['fires'][d]}" if t["fires"][d] else "") for d in SHOWN)
        print(f"{Path(f).name} [{group}, {model or 'model not recorded'}]: correct {t['correct']}, "
              f"scored {t['scored']}; fires: {fires}")
    print("totals (correct rows; scored; firings of S, R7, S|R7):")
    for g in GROUPS:
        k = totals[g]
        print(f"  {g:12s} files {k['files']}, correct {k['correct']}, scored {k['scored']}; "
              f"S {k['S']}, R7 {k['R7']}, S|R7 {k['S|R7']}")
    att = {x: totals["hosted"][x] + totals["ladder"][x] for x in ("correct", "scored", "S", "R7", "S|R7")}
    print(f"  {'attributed':12s} correct {att['correct']}, scored {att['scored']}; "
          f"S {att['S']}, R7 {att['R7']}, S|R7 {att['S|R7']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
