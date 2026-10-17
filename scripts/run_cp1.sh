#!/usr/bin/env bash
# CP1 run: our pipeline (policies cost and none) and the free-form agent (plain and python arms)
# on the validated fresh class D dev questions, REPEATS repeats each, one output file per run.
#
#   bash scripts/run_cp1.sh            # all repeats
#   REPEATS=1 bash scripts/run_cp1.sh  # a smoke test of one repeat
#
# Built to survive a sleeping laptop or a lost connection:
#   * each run writes its file after every question and resumes where it stopped (--resume);
#   * a question that failed because the model API was unreachable is never counted as a result,
#     it is redone on resume; after three such failures in a row the run stops and exits with 75;
#   * on exit 75 this script waits WAIT seconds and starts the run again, up to TRIES times;
#   * a run whose file is marked complete is skipped, so the whole script can be started again.
# The order is repeat by repeat (every system once, then every system again), so a partial result
# is still balanced. Refuses to start unless HEAD carries the tag cp1-freeze and the tracked
# files are unchanged.
set -u
cd "$(dirname "$0")/.." || exit 2

Q="${QUESTIONS:-data/questions_v2_dev_classD_cp1.validated.jsonl}"
REPEATS="${REPEATS:-3}"
OUT="${OUT:-data}"
WAIT="${WAIT:-180}"
TRIES="${TRIES:-40}"

if ! git describe --tags --exact-match HEAD 2>/dev/null | grep -qx "cp1-freeze"; then
  echo "refusing to run: HEAD is not tagged cp1-freeze" >&2; exit 2
fi
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "refusing to run: tracked files have uncommitted changes" >&2; exit 2
fi
[ -f "$Q" ] || { echo "missing question file $Q" >&2; exit 2; }

complete() { [ -s "$1" ] && grep -q '"complete": true' "$1"; }

run() {  # run <output file> <command...>   (the command is given without --out and --resume)
  local out="$1"; shift
  if complete "$out"; then echo "skip   $out (complete)"; return 0; fi
  local try rc
  for try in $(seq 1 "$TRIES"); do
    echo "$(date '+%H:%M:%S') start  $out (attempt $try)"
    "$@" --out "$out" --resume >> "${out%.json}.log" 2>&1
    rc=$?
    if [ "$rc" -eq 0 ] && complete "$out"; then echo "$(date '+%H:%M:%S') done   $out"; return 0; fi
    if [ "$rc" -eq 75 ]; then
      echo "$(date '+%H:%M:%S') interrupted: model API unreachable; waiting ${WAIT}s"
      sleep "$WAIT"; continue
    fi
    echo "$(date '+%H:%M:%S') FAILED $out (exit $rc, see ${out%.json}.log)" >&2
    return 1
  done
  echo "$(date '+%H:%M:%S') GAVE UP $out after $TRIES attempts; start the script again later" >&2
  return 1
}

for i in $(seq 1 "$REPEATS"); do
  run "$OUT/cp1_cost_$i.json"   python -m benchmark_v2.run_slice --questions "$Q" --bind-policy cost --cap 10000
  run "$OUT/cp1_none_$i.json"   python -m benchmark_v2.run_slice --questions "$Q" --bind-policy none --cap 10000
  run "$OUT/cp1_plain_$i.json"  python -m benchmark_v2.agent_baseline --questions "$Q" --arm plain --cap 10000
  run "$OUT/cp1_python_$i.json" python -m benchmark_v2.agent_baseline --questions "$Q" --arm python --cap 10000
done
echo "finished: $(grep -l '"complete": true' "$OUT"/cp1_*_[0-9].json 2>/dev/null | wc -l) of $((REPEATS * 4)) result files complete"
