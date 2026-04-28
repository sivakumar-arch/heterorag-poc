"""
heterorag/evaluation/metrics.py
=================================
Step 13 — Metrics computation.

Computes all metrics defined in Foundation Doc v1.6 §7.6 from raw_results.jsonl
and ground truth, and writes paper-ready tables to results/.

Primary metrics:
  SC   — Source Coverage per query class + aggregate
  AF   — Answer Faithfulness per class (F1 / NDCG@10 / Recall@10)
  RL   — Retrieval Latency: mean, p50, p95, p99 per class + aggregate

Secondary metrics:
  QTSR — Query Translation Success Rate per service
  ICR  — Integration Conflict Rate (cross-service queries only)

Output files:
  results/metrics_summary.csv      — all metrics, all systems, per class
  results/latency_percentiles.csv  — RL distributions for the latency figure
  results/sc_per_class.csv         — SC heatmap data
  results/af_per_class.csv         — AF heatmap data
  results/qtsr_icr.csv             — QTSR and ICR
  results/metrics_summary.json     — machine-readable full results

Usage:
    computer = MetricsComputer(
        raw_results_path=Path("results/raw_results.jsonl"),
        output_dir=Path("results"),
        pg_conn=pg_conn,
        neo4j_driver=neo4j_driver,
        fixture_dir=Path("ground-truth/document/fixtures"),
    )
    computer.compute()
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Query class labels in canonical order for tables
CLASS_ORDER = ["1", "2", "3", "4a", "4b", "4c", "5"]
SYSTEMS_ORDER = [
    "HeteroRAG_Full",
    "B1_SQL_Only",
    "B2_Document_Only",
    "B3_LLM_FunctionCalling",
    "B4_Fixed_Plan",
]
ALL_SERVICES = ["sql", "graph", "document"]
CROSS_SERVICE_CLASSES = {"4a", "4b", "4c", "5"}


# =============================================================================
# Metric implementations
# =============================================================================

def compute_sc(required: list[str], queried: list[str]) -> float:
    """
    Source Coverage for one query.
    SC(q) = |S*(q) ∩ S(q)| / |S*(q)|
    """
    if not required:
        return 1.0
    req_set    = set(s.lower() for s in required)
    queried_set = set()
    for s in queried:
        if "sql" in s or "activity" in s or "user" in s:
            queried_set.add("sql")
        elif "graph" in s or "knowledge" in s or "neo4j" in s:
            queried_set.add("graph")
        elif "document" in s or "content" in s or "elastic" in s:
            queried_set.add("document")
    return len(req_set & queried_set) / len(req_set)


def set_f1(retrieved: list, ground_truth: list) -> float:
    """
    Set-level F1 between retrieved and ground truth result sets.
    Used for SQL and Graph traversal queries.
    Items are compared as strings (canonicalised).
    """
    if not ground_truth:
        return 1.0 if not retrieved else 0.0
    if not retrieved:
        return 0.0
    r_set  = set(str(x) for x in retrieved)
    gt_set = set(str(x) for x in ground_truth)
    tp        = len(r_set & gt_set)
    precision = tp / len(r_set)   if r_set  else 0.0
    recall    = tp / len(gt_set)  if gt_set else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def ndcg_at_k(retrieved: list, ground_truth: list, k: int = 10) -> float:
    """
    NDCG@k for ranked retrieval (Graph algorithm queries Q11–Q20).
    ground_truth is the ideal ranking (index 0 = most relevant).
    Uses graded relevance: item at GT position i has relevance 1/(i+1).
    Binary NDCG (all-in-set = 1.0) is incorrect for ranked evaluation.
    """
    if not ground_truth:
        return 1.0
    # Graded relevance: GT rank 1 → rel=1.0, rank 2 → rel=0.5, etc.
    gt_rel = {str(x): 1.0 / (i + 1) for i, x in enumerate(ground_truth[:k])}
    # IDCG: ideal order = GT order
    idcg = sum(
        gt_rel[str(x)] / math.log2(i + 2)
        for i, x in enumerate(ground_truth[:k])
    )
    if idcg == 0:
        return 0.0
    dcg = sum(
        gt_rel.get(str(x), 0.0) / math.log2(i + 2)
        for i, x in enumerate(retrieved[:k])
    )
    return min(dcg / idcg, 1.0)


def recall_at_k(retrieved: list, ground_truth: list, k: int = 10) -> float:
    """
    Recall@k for document retrieval.
    Fraction of GT passage IDs appearing in top-k retrieved.
    """
    if not ground_truth:
        return 1.0
    gt_set    = set(str(x) for x in ground_truth)
    top_k_set = set(str(x) for x in retrieved[:k])
    return len(gt_set & top_k_set) / len(gt_set)


def percentile(values: list[float], p: float) -> float:
    """p-th percentile (0–100) of values."""
    if not values:
        return 0.0
    s = sorted(values)
    idx = (len(s) - 1) * p / 100
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


# =============================================================================
# Ground truth extractors
# =============================================================================

def _gt_sql_rows(view_name: str, pg_conn) -> list[str]:
    """Return row IDs from a Flyway GT view as canonicalised strings."""
    if pg_conn is None or view_name is None:
        return []
    try:
        with pg_conn.cursor() as cur:
            cur.execute(f"SELECT * FROM {view_name} LIMIT 200")
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
            # Represent each row as a sorted key:value string for F1 comparison
            return [
                "|".join(f"{c}={v}" for c, v in zip(cols, row) if v is not None)
                for row in rows
            ]
    except Exception as e:
        log.warning("GT SQL load failed for view '%s': %s", view_name, e)
        return []


def _gt_graph_rows(cypher: str, neo4j_driver) -> list[str]:
    """Execute a Cypher GT query and return canonicalised result strings."""
    if neo4j_driver is None or cypher is None:
        return []
    try:
        with neo4j_driver.session() as s:
            result  = s.run(cypher)
            records = result.fetch(200)
            return ["|".join(f"{k}={v}" for k, v in sorted(dict(r).items())) for r in records]
    except Exception as e:
        log.warning("GT Cypher failed: %s", e)
        return []


def _gt_doc_ids(fixture_file: str, fixture_dir: Path) -> list[str]:
    """Load expected document IDs from a fixture file."""
    if fixture_file is None or fixture_dir is None:
        return []
    path = fixture_dir / fixture_file
    if not path.exists():
        return []
    with path.open() as f:
        data = json.load(f)
    return [str(x) for x in data.get("expected_ids", [])]


def _extract_result_ids(answer: str, service: str) -> list[str]:
    """
    Heuristic: extract numeric IDs from an answer string for F1/Recall computation.
    In the full system this would compare against raw retrieved rows; here we
    parse the answer text as a proxy since we don't persist raw rows.
    A production implementation would persist RawServiceResult alongside the answer.
    """
    import re
    return re.findall(r"\b\d{4,}\b", answer)   # 4+ digit numbers = likely SO IDs


# =============================================================================
# MetricsComputer
# =============================================================================

class MetricsComputer:
    """
    Step 13: computes all paper metrics from raw_results.jsonl.

    Args:
        raw_results_path: Path to JSONL file written by BenchmarkRunner.
        output_dir:       Where to write metric CSV/JSON tables.
        pg_conn:          Live PostgreSQL connection for GT SQL views.
        neo4j_driver:     Live Neo4j driver for GT Cypher queries.
        fixture_dir:      Directory of pre-generated document GT fixtures.
    """

    def __init__(
        self,
        raw_results_path: Path,
        output_dir:       Path,
        pg_conn           = None,
        neo4j_driver      = None,
        fixture_dir:      Path | None = None,
    ):
        self._raw_path   = raw_results_path
        self._out        = output_dir
        self._pg         = pg_conn
        self._neo4j      = neo4j_driver
        self._fixture_dir = fixture_dir
        self._out.mkdir(parents=True, exist_ok=True)

    def compute(self) -> dict[str, Any]:
        """Run all metrics. Returns a nested dict mirroring the JSON output."""
        records = self._load_records()
        log.info("MetricsComputer: %d raw records loaded", len(records))

        sc     = self._compute_sc(records)
        af     = self._compute_af(records)
        rl     = self._compute_rl(records)
        qtsr   = self._compute_qtsr(records)
        icr    = self._compute_icr(records)

        summary = {
            "source_coverage":  sc,
            "answer_faithfulness": af,
            "retrieval_latency": rl,
            "qtsr":             qtsr,
            "icr":              icr,
        }

        self._write_json(summary)
        self._write_sc_csv(sc)
        self._write_af_csv(af)
        self._write_rl_csv(rl)
        self._write_qtsr_icr_csv(qtsr, icr)
        self._write_summary_csv(sc, af, rl)

        log.info("MetricsComputer: all outputs written to %s", self._out)
        return summary

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def _load_records(self) -> list[dict]:
        records = []
        if not self._raw_path.exists():
            log.error("raw_results.jsonl not found at %s", self._raw_path)
            return records
        with self._raw_path.open() as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return records

    # ------------------------------------------------------------------
    # SC — Source Coverage
    # ------------------------------------------------------------------

    def _compute_sc(self, records: list[dict]) -> dict:
        # sc[system][class] = list of per-question SC values
        sc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

        for r in records:
            s = compute_sc(
                r.get("required_services", []),
                r.get("queried_service_ids", []),
            )
            sc[r["system_name"]][r["query_class_label"]].append(s)

        result = {}
        for sys_name in SYSTEMS_ORDER:
            if sys_name not in sc:
                continue
            result[sys_name] = {}
            all_vals = []
            for cls in CLASS_ORDER:
                vals = sc[sys_name].get(cls, [])
                mean = statistics.mean(vals) if vals else 0.0
                result[sys_name][cls] = round(mean, 4)
                all_vals.extend(vals)
            result[sys_name]["aggregate"] = round(
                statistics.mean(all_vals) if all_vals else 0.0, 4
            )
        return result

    # ------------------------------------------------------------------
    # AF — Answer Faithfulness
    # ------------------------------------------------------------------

    def _compute_af(self, records: list[dict]) -> dict:
        af: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

        for r in records:
            score = self._af_for_record(r)
            af[r["system_name"]][r["query_class_label"]].append(score)

        result = {}
        for sys_name in SYSTEMS_ORDER:
            if sys_name not in af:
                continue
            result[sys_name] = {}
            all_vals = []
            for cls in CLASS_ORDER:
                vals = af[sys_name].get(cls, [])
                mean = statistics.mean(vals) if vals else 0.0
                result[sys_name][cls] = round(mean, 4)
                all_vals.extend(vals)
            result[sys_name]["aggregate"] = round(
                statistics.mean(all_vals) if all_vals else 0.0, 4
            )
        return result

    def _af_for_record(self, r: dict) -> float:
        """Compute AF for one (question, system) record."""
        metric    = r.get("af_metric", "f1")
        answer    = r.get("answer", "")
        cls_label = r.get("query_class_label", "1")

        # Extract result IDs from answer text (proxy — see docstring on _extract_result_ids)
        retrieved_ids = _extract_result_ids(answer, "")

        if metric == "f1":
            # Load ground truth from the appropriate source
            gt_view    = r.get("gt_sql_view")
            gt_cypher  = r.get("gt_cypher")
            gt_fixture = r.get("gt_doc_fixture")

            gt_ids: list[str] = []
            if gt_view:
                gt_ids.extend(_gt_sql_rows(gt_view, self._pg))
            if gt_cypher:
                gt_ids.extend(_gt_graph_rows(gt_cypher, self._neo4j))
            if not gt_ids:
                return 0.0
            return set_f1(retrieved_ids, gt_ids)

        elif metric == "ndcg10":
            gt_cypher = r.get("gt_cypher")
            if not gt_cypher:
                return 0.0
            gt_ranked = _gt_graph_rows(gt_cypher, self._neo4j)
            return ndcg_at_k(retrieved_ids, gt_ranked, k=10)

        elif metric == "recall10":
            gt_ids = _gt_doc_ids(r.get("gt_doc_fixture", ""), self._fixture_dir)
            return recall_at_k(retrieved_ids, gt_ids, k=10)

        return 0.0

    # ------------------------------------------------------------------
    # RL — Retrieval Latency
    # ------------------------------------------------------------------

    def _compute_rl(self, records: list[dict]) -> dict:
        # rl[system][class] = list of retrieval_ms values
        rl: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

        for r in records:
            ms = r.get("retrieval_ms", 0.0)
            rl[r["system_name"]][r["query_class_label"]].append(ms)

        result = {}
        for sys_name in SYSTEMS_ORDER:
            if sys_name not in rl:
                continue
            result[sys_name] = {}
            all_vals = []
            for cls in CLASS_ORDER:
                vals = rl[sys_name].get(cls, [])
                result[sys_name][cls] = self._rl_stats(vals)
                all_vals.extend(vals)
            result[sys_name]["aggregate"] = self._rl_stats(all_vals)
        return result

    @staticmethod
    def _rl_stats(vals: list[float]) -> dict:
        if not vals:
            return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "n": 0}
        return {
            "mean": round(statistics.mean(vals), 1),
            "p50":  round(percentile(vals, 50), 1),
            "p95":  round(percentile(vals, 95), 1),
            "p99":  round(percentile(vals, 99), 1),
            "n":    len(vals),
        }

    # ------------------------------------------------------------------
    # QTSR — Query Translation Success Rate
    # ------------------------------------------------------------------

    def _compute_qtsr(self, records: list[dict]) -> dict:
        """
        QTSR per service = fraction of runs where that service was queried
        (i.e., not dropped by QueryValidator).
        Requires per-service tracking; approximated from queried_service_ids.
        """
        # Attempts and successes per (system, service)
        attempts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        successes: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

        _svc_map = {
            "user-activity-service":   "sql",
            "knowledge-graph-service": "graph",
            "content-service":         "document",
        }

        for r in records:
            sys_name = r["system_name"]
            required = r.get("required_services", [])
            queried  = r.get("queried_service_ids", [])

            queried_normalized = set()
            for s in queried:
                for svc_id, short in _svc_map.items():
                    if svc_id in s or short in s:
                        queried_normalized.add(short)

            for svc in required:
                attempts[sys_name][svc] += 1
                if svc in queried_normalized:
                    successes[sys_name][svc] += 1

        result = {}
        for sys_name in SYSTEMS_ORDER:
            result[sys_name] = {}
            for svc in ALL_SERVICES:
                n = attempts[sys_name].get(svc, 0)
                s = successes[sys_name].get(svc, 0)
                result[sys_name][svc] = round(s / n, 4) if n > 0 else None
        return result

    # ------------------------------------------------------------------
    # ICR — Integration Conflict Rate
    # ------------------------------------------------------------------

    def _compute_icr(self, records: list[dict]) -> dict:
        """
        ICR = queries where conflict_count > 0 / total cross-service queries.
        Reported per system.
        """
        result = {}
        for sys_name in SYSTEMS_ORDER:
            sys_records = [r for r in records
                           if r["system_name"] == sys_name
                           and r["query_class_label"] in CROSS_SERVICE_CLASSES]
            total    = len(sys_records)
            conflict = sum(1 for r in sys_records if r.get("conflict_count", 0) > 0)
            result[sys_name] = round(conflict / total, 4) if total > 0 else 0.0
        return result

    # ------------------------------------------------------------------
    # Output writers
    # ------------------------------------------------------------------

    def _write_json(self, summary: dict) -> None:
        path = self._out / "metrics_summary.json"
        with path.open("w") as f:
            json.dump(summary, f, indent=2)
        log.info("Written: %s", path)

    def _write_sc_csv(self, sc: dict) -> None:
        path = self._out / "sc_per_class.csv"
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["system"] + CLASS_ORDER + ["aggregate"])
            for sys_name in SYSTEMS_ORDER:
                if sys_name not in sc:
                    continue
                row = [sys_name] + [sc[sys_name].get(c, "") for c in CLASS_ORDER] \
                      + [sc[sys_name].get("aggregate", "")]
                w.writerow(row)
        log.info("Written: %s", path)

    def _write_af_csv(self, af: dict) -> None:
        path = self._out / "af_per_class.csv"
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["system"] + CLASS_ORDER + ["aggregate"])
            for sys_name in SYSTEMS_ORDER:
                if sys_name not in af:
                    continue
                row = [sys_name] + [af[sys_name].get(c, "") for c in CLASS_ORDER] \
                      + [af[sys_name].get("aggregate", "")]
                w.writerow(row)
        log.info("Written: %s", path)

    def _write_rl_csv(self, rl: dict) -> None:
        path = self._out / "latency_percentiles.csv"
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["system", "class", "mean_ms", "p50_ms", "p95_ms", "p99_ms", "n"])
            for sys_name in SYSTEMS_ORDER:
                if sys_name not in rl:
                    continue
                for cls in CLASS_ORDER + ["aggregate"]:
                    s = rl[sys_name].get(cls, {})
                    w.writerow([
                        sys_name, cls,
                        s.get("mean",""), s.get("p50",""),
                        s.get("p95",""), s.get("p99",""),
                        s.get("n",""),
                    ])
        log.info("Written: %s", path)

    def _write_qtsr_icr_csv(self, qtsr: dict, icr: dict) -> None:
        path = self._out / "qtsr_icr.csv"
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["system", "qtsr_sql", "qtsr_graph", "qtsr_document", "icr"])
            for sys_name in SYSTEMS_ORDER:
                q = qtsr.get(sys_name, {})
                w.writerow([
                    sys_name,
                    q.get("sql", ""),
                    q.get("graph", ""),
                    q.get("document", ""),
                    icr.get(sys_name, ""),
                ])
        log.info("Written: %s", path)

    def _write_summary_csv(self, sc: dict, af: dict, rl: dict) -> None:
        """One-row-per-system aggregate summary — the main paper results table."""
        path = self._out / "metrics_summary.csv"
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "system",
                "SC_aggregate",
                "AF_aggregate",
                "RL_mean_ms", "RL_p50_ms", "RL_p95_ms", "RL_p99_ms",
            ])
            for sys_name in SYSTEMS_ORDER:
                rl_agg = rl.get(sys_name, {}).get("aggregate", {})
                w.writerow([
                    sys_name,
                    sc.get(sys_name, {}).get("aggregate", ""),
                    af.get(sys_name, {}).get("aggregate", ""),
                    rl_agg.get("mean", ""),
                    rl_agg.get("p50", ""),
                    rl_agg.get("p95", ""),
                    rl_agg.get("p99", ""),
                ])
        log.info("Written: %s", path)
