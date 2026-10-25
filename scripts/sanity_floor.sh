#!/usr/bin/env bash
# Sanity run before the full capability-floor run of one model (novelty gate G0, rule L). A few questions per
# template through the three systems (pipeline policy cost, python agent, plain agent; answers written as text
# count, as in the full run), then benchmark_v2.floor_sanity checks the plumbing: the right model answered,
# nothing crashed, tool calls were read, reasoning was handled, no think markup leaked, how long it takes.
# It checks plumbing and not accuracy. Its files go to /tmp, are not evidence, and are never used to choose or
# drop a model or to change a prompt; a FAIL is fixed in the plumbing and disclosed in the log.
#
#   FLOOR_MODEL=ollama:qwen3:30b bash scripts/sanity_floor.sh          # 2 questions per template (8)
#   FLOOR_MODEL=ollama:qwen3:30b LIMIT=1 bash scripts/sanity_floor.sh  # 1 per template (4)
#
# Same preconditions as run_floor.sh: prompts as pinned at cp1b-freeze, tracked files committed, model installed.
set -u
cd "$(dirname "$0")/.." || exit 2

FLOOR_MODEL="${FLOOR_MODEL:-}"
Q="${QUESTIONS:-data/questions_v2_dev_classD_cp1b.final.jsonl}"
LIMIT="${LIMIT:-2}"

case "$FLOOR_MODEL" in ollama:*) ;; *) echo "set FLOOR_MODEL=ollama:<name>, for example ollama:qwen3:30b" >&2; exit 2;; esac
NAME="${FLOOR_MODEL#ollama:}"
python -m pytest tests/benchmark_v2/test_prompt_pins.py -q >/dev/null 2>&1 || { echo "refusing to run: the prompts differ from cp1b-freeze (test_prompt_pins)" >&2; exit 2; }
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "refusing to run: tracked files have uncommitted changes" >&2; exit 2
fi
[ -f "$Q" ] || { echo "missing question file $Q" >&2; exit 2; }
command -v ollama >/dev/null 2>&1 || { echo "ollama is not installed or not on PATH" >&2; exit 2; }
ollama show "$NAME" >/dev/null 2>&1 || { echo "model $NAME is not installed: run  ollama pull $NAME" >&2; exit 2; }

D="/tmp/sanity_$(echo "$NAME" | tr ':/' '__')_$(date +%m%d_%H%M%S)"
mkdir -p "$D" || exit 2
git rev-parse HEAD > "$D/commit.txt"
{ ollama --version; ollama show "$NAME"; } > "$D/server.txt" 2>&1
export HETERORAG_PLANNER_MODEL="$FLOOR_MODEL" HETERORAG_TRANSLATOR_MODEL="$FLOOR_MODEL" HETERORAG_AGENT_MODEL="$FLOOR_MODEL"
echo "sanity run of $NAME, $LIMIT question(s) per template, files in $D"

stage() {  # stage <name> <command...>
  local name="$1"; shift
  echo "$(date '+%H:%M:%S') start  $name"
  "$@" --out "$D/$name.json" > "$D/$name.log" 2>&1
  local rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "$(date '+%H:%M:%S') FAILED $name (exit $rc); last lines of $D/$name.log:" >&2
    tail -n 8 "$D/$name.log" >&2
    exit 1
  fi
  echo "$(date '+%H:%M:%S') done   $name"
}

stage cost   python -m benchmark_v2.run_slice --questions "$Q" --limit "$LIMIT" --bind-policy cost --cap 10000
stage python python -m benchmark_v2.agent_baseline --questions "$Q" --limit "$LIMIT" --arm python --cap 10000 --accept-text-final
stage plain  python -m benchmark_v2.agent_baseline --questions "$Q" --limit "$LIMIT" --arm plain --cap 10000 --accept-text-final

python -m benchmark_v2.floor_sanity --model "$NAME" --cost "$D/cost.json" --python "$D/python.json" --plain "$D/plain.json"
