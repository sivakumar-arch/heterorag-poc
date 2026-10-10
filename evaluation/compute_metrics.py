#!/usr/bin/env python3
"""
evaluation/compute_metrics.py
================================
Compute all paper metrics from raw_results.jsonl.

Usage:
    python evaluation/compute_metrics.py [--results-dir results]

Produces:
    results/metrics_summary.csv
    results/metrics_summary.json
    results/sc_per_class.csv
    results/af_per_class.csv
    results/latency_percentiles.csv
    results/qtsr_icr.csv
"""

import argparse
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results", type=Path)
    parser.add_argument("--fixture-dir",
                        default="ground-truth/document/fixtures", type=Path)
    args = parser.parse_args()

    raw_path = args.results_dir / "raw_results.jsonl"
    if not raw_path.exists():
        print(f"ERROR: {raw_path} not found. Run run_benchmark.py first.")
        return

    # Optional GT connections
    pg_conn = neo4j_driver = None
    try:
        import psycopg2
        pg_conn = psycopg2.connect(
            host=os.getenv("PG_HOST","localhost"),
            port=int(os.getenv("PG_PORT","5432")),
            dbname=os.getenv("PG_DBNAME","heterorag"),
            user=os.getenv("PG_USER","heterorag"),
            password=os.getenv("PG_PASSWORD","heterorag_secret"),
        )
    except Exception as e:
        print(f"Warning: PostgreSQL not available for GT computation: {e}")
    try:
        from neo4j import GraphDatabase
        neo4j_driver = GraphDatabase.driver(
            f"bolt://{os.getenv('NEO4J_HOST','localhost')}:{os.getenv('NEO4J_BOLT_PORT','7687')}",
            auth=(os.getenv("NEO4J_USER","neo4j"), os.getenv("NEO4J_PASSWORD","heterorag_secret")),
        )
    except Exception as e:
        print(f"Warning: Neo4j not available for GT computation: {e}")

    from heterorag.evaluation.metrics import MetricsComputer
    computer = MetricsComputer(
        raw_results_path = raw_path,
        output_dir       = args.results_dir,
        pg_conn          = pg_conn,
        neo4j_driver     = neo4j_driver,
        fixture_dir      = args.fixture_dir,
    )
    summary = computer.compute()

    print("\n=== Aggregate Results ===")
    for sys_name in summary["source_coverage"]:
        sc_agg = summary["source_coverage"].get(sys_name, {}).get("aggregate", "N/A")
        af_agg = summary["answer_faithfulness"].get(sys_name, {}).get("aggregate", "N/A")
        rl_agg = summary["retrieval_latency"].get(sys_name, {}).get("aggregate", {})
        print(f"  {sys_name:30s}  SC={sc_agg}  AF={af_agg}  "
              f"RL_mean={rl_agg.get('mean','N/A')}ms  "
              f"RL_p95={rl_agg.get('p95','N/A')}ms")

    dq = dict(summary.get("data_quality", {}))
    af_cov = dq.pop("af_ground_truth", {})
    bad = {k: v for k, v in dq.items() if v.get("error") or v.get("infra_error") or v.get("n_missing")}
    if bad:
        print("\n=== DATA QUALITY WARNING: runs excluded from the metrics above ===")
        for sys_name, v in bad.items():
            print(f"  {sys_name:30s}  error={v.get('error',0)}  infra_error={v.get('infra_error',0)}  "
                  f"never-succeeded={v.get('n_missing',0)}")
        print("  Re-run run_benchmark.py (same --output-dir) to retry them before reporting numbers.")

    # AF is only defined where the question has usable ground truth.
    if af_cov:
        first = next(iter(af_cov.values()))
        scored = {c: v["scored"] for c, v in first.items()}
        absent = {c: v["no_ground_truth"] for c, v in first.items()}
        if any(absent.values()):
            print("\n=== AF GROUND-TRUTH COVERAGE (per system; classes with no ground truth are excluded, not scored 0) ===")
            for cls in sorted(first):
                total = scored[cls] + absent[cls]
                print(f"  class {cls:4s}  scored {scored[cls]:3d} of {total:3d} runs  "
                      f"(no ground truth: {absent[cls]})")

if __name__ == "__main__":
    main()
