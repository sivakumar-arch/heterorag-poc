#!/usr/bin/env python3
"""
evaluation/run_spotcheck.py
============================
Runs the 15-question manual spot-check through HeteroRAG and baselines B1-B3.
Reuses BenchmarkRunner infrastructure exactly — no code duplication.

Usage:
    export PYTHONPATH=.
    export ANTHROPIC_API_KEY="sk-ant-..."

    # Smoke test (no API calls)
    python evaluation/run_spotcheck.py --mock

    # Real run (~60 API calls, ~$0.60, ~10 minutes)
    python evaluation/run_spotcheck.py

Output:
    results/spotcheck/spotcheck_answers.csv
"""

import argparse
import csv
import logging
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── 15 spot-check questions ───────────────────────────────────────────────────
SPOTCHECK_QUESTIONS = [
    # CLASS 1 — SQL ONLY
    ("Q1-A", 1, "How many users in the database have a reputation score above 10,000?"),
    ("Q1-B", 1, "Who is the highest-reputation user in the database and what is their reputation score?"),
    ("Q1-C", 1, "How many accepted answers has the user whuber (id=919) provided?"),
    # CLASS 2 — GRAPH ONLY
    ("Q2-A", 2, "Which tag has the most co-occurrence relationships with other tags in the knowledge graph?"),
    ("Q2-B", 2, "Which tags most frequently co-occur with the regression tag and how many times does each co-occur?"),
    ("Q2-C", 2, "How many distinct tags co-occur with the machine-learning tag across all questions?"),
    # CLASS 3 — DOCUMENT ONLY
    ("Q3-A", 3, "What does the highest-scored question in the database (post 2691, score 1343) ask about PCA and eigenvalues?"),
    ("Q3-B", 3, "According to posts about k-means clustering, what methods are recommended for choosing the number of clusters?"),
    ("Q3-C", 3, "What does post 133656 say about the assumptions k-means makes about data distribution?"),
    # CLASS 4b — SQL + DOCUMENT
    ("Q4b-A", "4b", "What is the view count of the highest-scored question in the database, and what does that question ask about PCA and eigenvalues?"),
    ("Q4b-B", "4b", "How many answers has the highest-reputation user whuber posted, and what statistical topics appear in their top answers?"),
    ("Q4b-C", "4b", "How many questions are tagged machine-learning in the database, and what does post 133656 say about k-means assumptions?"),
    # CLASS 5 — ALL THREE
    ("Q5-A", 5, "Who is the highest-reputation user, how many answers have they posted, which users most often accepted their answers, and what statistical topics do their top answers cover?"),
    ("Q5-B", 5, "What is the score and view count of the top PCA question, which tags most co-occur with the PCA tag, and what does that question ask about eigenvalues?"),
    ("Q5-C", 5, "How many accepted answers has whuber provided, which users most frequently accepted their answers, and what do their top-scoring answers explain?"),
]

SYSTEMS_TO_RUN = ["HeteroRAG_Full", "B1_SQL_Only", "B2_Document_Only", "B3_LLM_FunctionCalling"]


def main():
    parser = argparse.ArgumentParser(description="HeteroRAG 15-question spot-check")
    parser.add_argument("--mock", action="store_true",
                        help="Use mock LLM — no API calls, for smoke testing")
    parser.add_argument("--output-dir", default="results/spotcheck", type=Path)
    args = parser.parse_args()

    # ── Reuse BenchmarkRunner.from_env() to get all systems ──────────────────
    from heterorag.evaluation.benchmark_runner import BenchmarkRunner
    from heterorag.evaluation.baselines import (
        B1_SQLOnlyRouter,
        B2_DocumentOnlyRAG,
        B3_LLMFunctionCallingRouter,
    )

    log.info("Initialising systems via BenchmarkRunner.from_env()...")
    runner = BenchmarkRunner.from_env(
        output_dir=args.output_dir,
        mock_llm=args.mock,
        resume=False,
    )

    # Extract the systems dict from the runner
    systems = runner._systems  # dict: system_name → BaselineSystem

    # Filter to only the systems we want for spot-check
    selected = {k: v for k, v in systems.items() if k in SYSTEMS_TO_RUN}
    log.info(f"Systems: {list(selected.keys())}")

    # ── Output setup ─────────────────────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "spotcheck_answers.csv"

    fields = ["question_id", "class", "system", "answer", "duration_ms", "timestamp"]
    rows   = []

    total = len(SPOTCHECK_QUESTIONS) * len(selected)
    done  = 0

    log.info(f"Running {len(SPOTCHECK_QUESTIONS)} questions × {len(selected)} systems = {total} calls")
    if args.mock:
        log.info("MOCK MODE — no real API calls")

    # ── Run each question through each system ─────────────────────────────────
    for qid, qclass, qtext in SPOTCHECK_QUESTIONS:
        log.info(f"\n{'─'*60}")
        log.info(f"{qid} (Class {qclass}): {qtext[:80]}...")

        for system_name, system in selected.items():
            log.info(f"  [{system_name}]...")
            t0 = time.perf_counter()
            try:
                result = system.run(query_id=qid, natural_query=qtext)
                answer = getattr(result, "answer", str(result)).strip()
            except Exception as e:
                log.error(f"  ERROR: {e}")
                answer = f"[ERROR: {e}]"
            duration_ms = round((time.perf_counter() - t0) * 1000, 1)

            done += 1
            log.info(f"  → {duration_ms:.0f}ms | {answer[:100]}...")

            rows.append({
                "question_id": qid,
                "class":       qclass,
                "system":      system_name,
                "answer":      answer,
                "duration_ms": duration_ms,
                "timestamp":   datetime.utcnow().isoformat(),
            })

        # Save after every question — progress not lost on interruption
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_ALL)
            writer.writeheader()
            writer.writerows(rows)
        log.info(f"  Saved ({done}/{total})")

    log.info(f"\n{'='*60}")
    log.info(f"✅ Spot-check complete — {len(rows)} results saved to {output_path}")
    log.info(f"Next: open {output_path} alongside manual_ground_truth.csv and fill in answers + correct columns")


if __name__ == "__main__":
    main()
