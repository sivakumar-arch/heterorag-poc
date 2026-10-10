#!/usr/bin/env python3
"""
evaluation/run_benchmark.py
============================
RUNBOOK Step 8 — Execute the full 120-question benchmark.

Usage:
    python evaluation/run_benchmark.py [--output-dir results] [--mock] [--repeats 3]

Options:
    --output-dir    Directory for raw_results.jsonl (default: results/)
    --mock          Use mock LLM (no API calls) — for smoke-testing infra
    --systems       Comma-separated system names (default: the five standard systems;
                    also available: B4_Fixed_Plan, B3_LLM_FunctionCalling_Legacy)
    --repeats       Repetitions per (question, system), for confidence intervals (default 1)
    --max-attempts  Attempts per run when a run errors or hits an infra error (default 3)
    --abstain-short-circuit / --no-abstain-short-circuit
                    Drop a service immediately when its translation is CANNOT_ANSWER
                    instead of spending the validator's retry on it. ON by default for
                    every system; --no-abstain-short-circuit restores the original
                    one-retry policy. The setting is recorded in run_meta.jsonl.
    --question-ids  Comma-separated question ids to run instead of all 120 (smoke tests),
                    e.g. c1_q05,c2_q01,c3_q01,c4a_q01,c5_q01
    --no-resume     Re-run everything from scratch. By default only runs whose latest
                    outcome is status=ok are skipped, so failures are retried.

Environment:
    ANTHROPIC_API_KEY   Required unless --mock
    PG_HOST, PG_PORT, PG_DBNAME, PG_USER, PG_PASSWORD
    NEO4J_HOST, NEO4J_BOLT_PORT, NEO4J_USER, NEO4J_PASSWORD
    ES_HOST, ES_PORT
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

def main():
    parser = argparse.ArgumentParser(description="HeteroRAG Benchmark Runner")
    parser.add_argument("--output-dir", default="results", type=Path)
    parser.add_argument("--mock",      action="store_true",
                        help="Use mock LLM — no API calls, for infra testing")
    parser.add_argument("--no-resume", action="store_true",
                        help="Re-run from scratch, ignoring existing results")
    parser.add_argument("--systems", default=None,
                        help="Comma-separated system names to run")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--abstain-short-circuit", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--question-ids", default=None)
    args = parser.parse_args()

    from heterorag.evaluation.benchmark_runner import BenchmarkRunner
    runner = BenchmarkRunner.from_env(
        output_dir = args.output_dir,
        mock_llm   = args.mock,
        resume     = not args.no_resume,
        systems    = [x.strip() for x in args.systems.split(",")] if args.systems else None,
        repeats    = args.repeats,
        max_attempts = args.max_attempts,
        abstain_short_circuit = args.abstain_short_circuit,
        question_ids = [x.strip() for x in args.question_ids.split(",")] if args.question_ids else None,
    )
    raw_path = runner.run()
    print(f"\nBenchmark complete. Raw results: {raw_path}")
    print(f"Run:  python evaluation/compute_metrics.py --results-dir {args.output_dir}")


if __name__ == "__main__":
    main()
