#!/usr/bin/env bash
# CP1 run: our pipeline (policies cost and none) and the free-form agent (plain and python arms)
# on the validated fresh class D dev questions, REPEATS repeats each, one output file per run.
#
#   bash scripts/run_cp1.sh            # all repeats
#   REPEATS=1 bash scripts/run_cp1.sh  # a smoke test of one repeat
#
# Safe to stop and start again: a run whose output file exists is skipped. The order is repeat by
# repeat (every system once, then every system again), so a partial run is still a balanced one.
# Refuses to start unless HEAD carries the tag cp1-freeze and the working tree is clean.
set -u
cd "$(dirname "$0")/.." || exit 2

Q="${QUESTIONS:-data/questions_v2_dev_classD_cp1.validated.jsonl}"
REPEATS="${REPEATS:-3}"
OUT="${OUT:-data}"

if ! git describe --tags --exact-match HEAD 2>/dev/null | grep -qx "cp1-freeze"; then
  echo "refusing to run: HEAD is not tagged cp1-freeze" >&2; exit 2
fi
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "refusing to run: tracked files have uncommitted changes" >&2; exit 2
fi
[ -f "$Q" ] || { echo "missing question file $Q" >&2; exit 2; }

run() {  # run <output file> <command...>
  local out="$1"; shift
  if [ -s "$out" ]; then echo "skip   $out (exists)"; return 0; fi
  echo "$(date '+%H:%M:%S') start  $out"
  if "$@" > "${out%.json}.log" 2>&1; then
    echo "$(date '+%H:%M:%S') done   $out"
  else
    echo "$(date '+%H:%M:%S') FAILED $out (see ${out%.json}.log)" >&2
    rm -f "$out"
  fi
}

for i in $(seq 1 "$REPEATS"); do
  run "$OUT/cp1_cost_$i.json"   python -m benchmark_v2.run_slice --questions "$Q" --bind-policy cost --cap 10000 --out "$OUT/cp1_cost_$i.json"
  run "$OUT/cp1_none_$i.json"   python -m benchmark_v2.run_slice --questions "$Q" --bind-policy none --cap 10000 --out "$OUT/cp1_none_$i.json"
  run "$OUT/cp1_plain_$i.json"  python -m benchmark_v2.agent_baseline --questions "$Q" --arm plain --cap 10000 --out "$OUT/cp1_plain_$i.json"
  run "$OUT/cp1_python_$i.json" python -m benchmark_v2.agent_baseline --questions "$Q" --arm python --cap 10000 --out "$OUT/cp1_python_$i.json"
done
echo "finished: $(ls "$OUT"/cp1_*_[0-9].json 2>/dev/null | wc -l) result files"
