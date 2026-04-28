#!/usr/bin/env python3
"""
evaluation/run_benchmark.py
============================
RUNBOOK Step 8 — Execute the full 120-question benchmark.

Usage:
    python evaluation/run_benchmark.py [--output-dir results] [--mock] [--resume]

Options:
    --output-dir    Directory for raw_results.jsonl (default: results/)
    --mock          Use mock LLM (no API calls) — for smoke-testing infra
    --resume        Skip already-completed (question, system) pairs (default: True)
    --no-resume     Re-run everything from scratch

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
    args = parser.parse_args()

    from heterorag.evaluation.benchmark_runner import BenchmarkRunner
    runner = BenchmarkRunner.from_env(
        output_dir = args.output_dir,
        mock_llm   = args.mock,
        resume     = not args.no_resume,
    )
    raw_path = runner.run()
    print(f"\nBenchmark complete. Raw results: {raw_path}")
    print(f"Run:  python evaluation/compute_metrics.py --results-dir {args.output_dir}")


if __name__ == "__main__":
    main()
