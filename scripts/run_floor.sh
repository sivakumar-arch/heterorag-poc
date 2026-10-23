#!/usr/bin/env bash
# Capability-floor run (novelty gate G0, rule L1): the same questions, prompts and harness as the CP1b model sweep,
# with a small local model in place of a hosted one, for our pipeline (policy cost) and the free-form agent
# (python and plain arms). One output file per run, REPEATS repeats, written after every question.
#
#   ollama pull llama3.1:8b                      # once
#   bash scripts/run_floor.sh                    # llama3.1:8b, 3 repeats
#   REPEATS=1 bash scripts/run_floor.sh          # one repeat
#   FLOOR_MODEL=ollama:qwen3:8b TAG=floor_qwen3_8b bash scripts/run_floor.sh
#
# Before it starts it checks that (1) every prompt and tool definition is the one pinned at cp1b-freeze
# (tests/benchmark_v2/test_prompt_pins.py), (2) tracked files are committed, and (3) the model is installed in Ollama.
# It records the commit it ran on in data/${TAG}_commit.txt. Sampling is the model default and the context window is
# HETERORAG_OLLAMA_NUM_CTX (default 32768); a prompt that would not fit is refused, never truncated.
# Like run_cp1b.sh it resumes where it stopped and waits when the model server cannot be reached (exit 75).
set -u
cd "$(dirname "$0")/.." || exit 2

FLOOR_MODEL="${FLOOR_MODEL:-ollama:llama3.1:8b}"
TAG="${TAG:-floor_llama31_8b}"
Q="${QUESTIONS:-data/questions_v2_dev_classD_cp1b.final.jsonl}"
REPEATS="${REPEATS:-3}"
OUT="${OUT:-data}"
WAIT="${WAIT:-60}"
TRIES="${TRIES:-40}"

case "$FLOOR_MODEL" in ollama:*) ;; *) echo "FLOOR_MODEL must start with ollama: (got $FLOOR_MODEL)" >&2; exit 2;; esac
NAME="${FLOOR_MODEL#ollama:}"
python -m pytest tests/benchmark_v2/test_prompt_pins.py -q >/dev/null 2>&1 || { echo "refusing to run: the prompts differ from cp1b-freeze (test_prompt_pins)" >&2; exit 2; }
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "refusing to run: tracked files have uncommitted changes" >&2; exit 2
fi
[ -f "$Q" ] || { echo "missing question file $Q" >&2; exit 2; }
command -v ollama >/dev/null 2>&1 || { echo "ollama is not installed or not on PATH" >&2; exit 2; }
ollama show "$NAME" >/dev/null 2>&1 || { echo "model $NAME is not installed: run  ollama pull $NAME" >&2; exit 2; }
git rev-parse HEAD > "$OUT/${TAG}_commit.txt"

export HETERORAG_PLANNER_MODEL="$FLOOR_MODEL" HETERORAG_TRANSLATOR_MODEL="$FLOOR_MODEL" HETERORAG_AGENT_MODEL="$FLOOR_MODEL"

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
      echo "$(date '+%H:%M:%S') interrupted: model server unreachable; waiting ${WAIT}s"
      sleep "$WAIT"; continue
    fi
    echo "$(date '+%H:%M:%S') FAILED $out (exit $rc, see ${out%.json}.log)" >&2
    return 1
  done
  echo "$(date '+%H:%M:%S') GAVE UP $out after $TRIES attempts; start the script again later" >&2
  return 1
}

for i in $(seq 1 "$REPEATS"); do
  run "$OUT/${TAG}_cost_$i.json"   python -m benchmark_v2.run_slice --questions "$Q" --bind-policy cost --cap 10000
  run "$OUT/${TAG}_python_$i.json" python -m benchmark_v2.agent_baseline --questions "$Q" --arm python --cap 10000
  run "$OUT/${TAG}_plain_$i.json"  python -m benchmark_v2.agent_baseline --questions "$Q" --arm plain --cap 10000
done
echo "finished: $(grep -l '"complete": true' "$OUT"/${TAG}_*_[0-9].json 2>/dev/null | wc -l) of $((REPEATS * 3)) result files complete"
