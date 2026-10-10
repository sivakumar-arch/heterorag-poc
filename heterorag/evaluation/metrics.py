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
from typing import Any, Optional

log = logging.getLogger(__name__)

# Query class labels in canonical order for tables
CLASS_ORDER = ["1", "2", "3", "4a", "4b", "4c", "5"]
# Canonical table order. Systems found in the data but not listed here are
# appended alphabetically (see _ordered_systems), so new systems never silently
# disappear from the tables.
SYSTEMS_ORDER = [
    "HeteroRAG_Full",
    "B1_SQL_Only",
    "B2_Document_Only",
    "Select_Parallel",
    "Select_Sequential",
    "B3_LLM_FunctionCalling",           # legacy (pre-rerun) B3
    "B3_LLM_FunctionCalling_Legacy",
    "B4_Fixed_Plan",
]
ALL_SERVICES = ["sql", "graph", "document"]
CROSS_SERVICE_CLASSES = {"4a", "4b", "4c", "5"}


# =============================================================================
# Metric implementations
# =============================================================================

def _short_name(service_id: str) -> str | None:
    """Map a service_id to 'sql' | 'graph' | 'document' (None if unrecognised)."""
    if "sql" in service_id or "activity" in service_id or "user" in service_id:
        return "sql"
    if "graph" in service_id or "knowledge" in service_id or "neo4j" in service_id:
        return "graph"
    if "document" in service_id or "content" in service_id or "elastic" in service_id:
        return "document"
    return None


def _short_set(service_ids: list[str]) -> set[str]:
    return {n for n in (_short_name(s) for s in service_ids) if n}


def compute_sc(required: list[str], queried: list[str]) -> float:
    """
    Source Coverage for one query (RECALL of required services).
    SC(q) = |S*(q) ∩ S(q)| / |S*(q)|

    SC alone rewards querying everything: a system that always queries all
    services has SC = 1 by construction. Report it together with
    compute_service_prf(), which also penalises unnecessary services.
    """
    if not required:
        return 1.0
    req_set    = set(s.lower() for s in required)
    return len(req_set & _short_set(queried)) / len(req_set)


def compute_service_prf(required: list[str], selected: list[str]) -> tuple[float, float, float]:
    """
    Precision / recall / F1 of a set of services against the required set S*.
    P = |S*∩S|/|S|, R = |S*∩S|/|S*|. Empty S gives P = 0 unless S* is empty too.
    """
    req = set(s.lower() for s in required)
    sel = _short_set(selected)
    if not req and not sel:
        return 1.0, 1.0, 1.0
    tp = len(req & sel)
    p  = tp / len(sel) if sel else 0.0
    r  = tp / len(req) if req else 1.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def cluster_bootstrap_ci(
    values_by_cluster: dict[str, list[float]],
    n_boot: int = 2000,
    seed:   int = 0,
    alpha:  float = 0.05,
) -> tuple[float, float, float]:
    """
    Mean and (1-alpha) percentile-bootstrap CI, resampling QUESTIONS (clusters),
    not individual runs: repeats of the same question are not independent.
    Returns (mean, lo, hi). With < 2 clusters the CI collapses to the mean.
    """
    import random
    cluster_means = [statistics.mean(v) for v in values_by_cluster.values() if v]
    if not cluster_means:
        return 0.0, 0.0, 0.0
    mean = statistics.mean(cluster_means)
    if len(cluster_means) < 2:
        return mean, mean, mean
    rng   = random.Random(seed)
    n     = len(cluster_means)
    boots = sorted(
        statistics.mean(cluster_means[rng.randrange(n)] for _ in range(n))
        for _ in range(n_boot)
    )
    return mean, boots[int(n_boot * alpha / 2)], boots[min(n_boot - 1, int(n_boot * (1 - alpha / 2)))]


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
        # A failed statement leaves the transaction aborted; without a rollback every
        # later lookup on this connection fails too, even for views that exist.
        try:
            pg_conn.rollback()
        except Exception:
            pass
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


def _ordered_systems(records: list[dict]) -> list[str]:
    present = {r["system_name"] for r in records}
    known   = [x for x in SYSTEMS_ORDER if x in present]
    extra   = sorted(present - set(SYSTEMS_ORDER))
    return known + extra if present else list(SYSTEMS_ORDER)


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
        self._systems: list[str] = list(SYSTEMS_ORDER)
        self.data_quality: dict[str, Any] = {}
        self._out.mkdir(parents=True, exist_ok=True)

    def compute(self) -> dict[str, Any]:
        """Run all metrics. Returns a nested dict mirroring the JSON output."""
        records = self._load_records()
        log.info("MetricsComputer: %d usable records (status=ok, latest per run)", len(records))
        self._systems = _ordered_systems(records)

        sc     = self._compute_sc(records)
        af     = self._compute_af(records)
        rl     = self._compute_rl(records)
        qtsr   = self._compute_qtsr(records)
        icr    = self._compute_icr(records)
        prf    = self._compute_service_prf(records)
        drops  = self._compute_drop_breakdown(records)
        cost   = self._compute_cost(records)

        summary = {
            "source_coverage":  sc,
            "answer_faithfulness": af,
            "retrieval_latency": rl,
            "qtsr":             qtsr,
            "icr":              icr,
            "service_selection_prf": prf,
            "drop_breakdown":   drops,
            "cost":             cost,
            "data_quality":     self.data_quality,
        }

        self._write_json(summary)
        self._write_sc_csv(sc)
        self._write_af_csv(af)
        self._write_rl_csv(rl)
        self._write_qtsr_icr_csv(qtsr, icr)
        self._write_summary_csv(sc, af, rl)
        self._write_prf_csv(prf)
        self._write_drop_csv(drops)
        self._write_cost_csv(cost)

        log.info("MetricsComputer: all outputs written to %s", self._out)
        return summary

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def _load_records(self) -> list[dict]:
        """
        Load raw records and keep, per (question, system, repeat), the LATEST
        record whose status is "ok". Records with status "error"/"infra_error"
        (LLM rate limits, connection failures, exceptions) are excluded from every
        metric and counted in data_quality, instead of being scored as if the
        system had answered. A run that never succeeded is simply missing; the
        per-system `n_missing` in data_quality shows how many.
        """
        from heterorag.evaluation.benchmark_runner import record_key, record_status

        records: list[dict] = []
        if not self._raw_path.exists():
            log.error("raw_results.jsonl not found at %s", self._raw_path)
            return records

        latest_ok: dict[tuple, dict] = {}
        quality: dict[str, dict[str, int]] = defaultdict(
            lambda: {"ok": 0, "error": 0, "infra_error": 0, "superseded": 0, "legacy_no_trace": 0})
        seen_keys: dict[str, set] = defaultdict(set)
        with self._raw_path.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    key = record_key(r)
                except (json.JSONDecodeError, KeyError):
                    continue
                st = record_status(r)
                sysname = r["system_name"]
                seen_keys[sysname].add(key)
                if st != "ok":
                    quality[sysname][st] += 1
                    continue
                if key in latest_ok:
                    quality[sysname]["superseded"] += 1
                latest_ok[key] = r
        for key, r in latest_ok.items():
            quality[r["system_name"]]["ok"] += 1
            if "trace" not in r:
                quality[r["system_name"]]["legacy_no_trace"] += 1
            records.append(r)
        for sysname, keys in seen_keys.items():
            ok_keys = {k for k in latest_ok if k[1] == sysname}
            quality[sysname]["n_missing"] = len(keys - ok_keys)
        self.data_quality = {k: dict(v) for k, v in quality.items()}
        return records

    # ------------------------------------------------------------------
    # SC — Source Coverage
    # ------------------------------------------------------------------

    def _compute_sc(self, records: list[dict]) -> dict:
        # sc[system][class] = list of per-question SC values
        sc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

        by_q: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for r in records:
            s = compute_sc(
                r.get("required_services", []),
                r.get("queried_service_ids", []),
            )
            sc[r["system_name"]][r["query_class_label"]].append(s)
            by_q[r["system_name"]][r["question_id"]].append(s)

        result = {}
        for sys_name in self._systems:
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
            _m, lo, hi = cluster_bootstrap_ci(by_q[sys_name])
            result[sys_name]["aggregate_ci95"] = [round(lo, 4), round(hi, 4)]
        return result

    # ------------------------------------------------------------------
    # AF — Answer Faithfulness
    # ------------------------------------------------------------------

    def _compute_af(self, records: list[dict]) -> dict:
        """AF per system and class.

        A run whose question has no usable ground truth (a declared view that does not
        exist or returns no rows, an empty fixture) is excluded from the mean and counted
        under data_quality["af_ground_truth"]. Scoring it 0.0 would make "no ground truth"
        indistinguishable from "wrong answer".
        """
        af: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        coverage: dict[str, dict[str, dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: {"scored": 0, "no_ground_truth": 0}))

        for r in records:
            score = self._af_for_record(r)
            sys_name, cls = r["system_name"], r["query_class_label"]
            if score is None:
                coverage[sys_name][cls]["no_ground_truth"] += 1
                continue
            coverage[sys_name][cls]["scored"] += 1
            af[sys_name][cls].append(score)

        result = {}
        for sys_name in self._systems:
            if sys_name not in coverage:
                continue
            result[sys_name] = {}
            all_vals = []
            for cls in CLASS_ORDER:
                vals = af[sys_name].get(cls, [])
                result[sys_name][cls] = round(statistics.mean(vals), 4) if vals else None
                all_vals.extend(vals)
            result[sys_name]["aggregate"] = (
                round(statistics.mean(all_vals), 4) if all_vals else None)

        self.data_quality["af_ground_truth"] = {
            sys_name: {cls: dict(v) for cls, v in by_cls.items()}
            for sys_name, by_cls in coverage.items()
        }
        missing = sum(v["no_ground_truth"] for by_cls in coverage.values() for v in by_cls.values())
        total = sum(v["scored"] + v["no_ground_truth"] for by_cls in coverage.values() for v in by_cls.values())
        if missing:
            log.warning("AF: %d of %d runs have no usable ground truth and are excluded "
                        "(see data_quality.af_ground_truth for the per-class breakdown)",
                        missing, total)
        return result

    def _af_for_record(self, r: dict) -> Optional[float]:
        """Compute AF for one (question, system) record, or None when the question has
        no usable ground truth (see _compute_af)."""
        metric = r.get("af_metric", "f1")
        answer = r.get("answer", "")

        # Extract result IDs from answer text (proxy — see docstring on _extract_result_ids)
        retrieved_ids = _extract_result_ids(answer, "")

        if metric == "f1":
            gt_view    = r.get("gt_sql_view")
            gt_cypher  = r.get("gt_cypher")
            gt_ids: list[str] = []
            declared = 0
            for source, rows in (
                (gt_view,   lambda: _gt_sql_rows(gt_view, self._pg)),
                (gt_cypher, lambda: _gt_graph_rows(gt_cypher, self._neo4j)),
            ):
                if not source:
                    continue
                declared += 1
                got = rows()
                if not got:
                    return None        # a declared component is empty: ground truth incomplete
                gt_ids.extend(got)
            if not declared or not gt_ids:
                return None
            return set_f1(retrieved_ids, gt_ids)

        elif metric == "ndcg10":
            gt_cypher = r.get("gt_cypher")
            gt_ranked = _gt_graph_rows(gt_cypher, self._neo4j) if gt_cypher else []
            if not gt_ranked:
                return None
            return ndcg_at_k(retrieved_ids, gt_ranked, k=10)

        elif metric == "recall10":
            gt_ids = _gt_doc_ids(r.get("gt_doc_fixture", ""), self._fixture_dir)
            if not gt_ids:
                return None
            return recall_at_k(retrieved_ids, gt_ids, k=10)

        return None

    # ------------------------------------------------------------------
    # RL — Retrieval Latency
    # ------------------------------------------------------------------

    def _compute_rl(self, records: list[dict]) -> dict:
        # rl[system][class] = list of retrieval_ms values
        rl: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

        by_q: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for r in records:
            ms = r.get("retrieval_ms", 0.0)
            rl[r["system_name"]][r["query_class_label"]].append(ms)
            by_q[r["system_name"]][r["question_id"]].append(ms)

        result = {}
        for sys_name in self._systems:
            if sys_name not in rl:
                continue
            result[sys_name] = {}
            all_vals = []
            for cls in CLASS_ORDER:
                vals = rl[sys_name].get(cls, [])
                result[sys_name][cls] = self._rl_stats(vals)
                all_vals.extend(vals)
            result[sys_name]["aggregate"] = self._rl_stats(all_vals)
            _m, lo, hi = cluster_bootstrap_ci(by_q[sys_name])
            result[sys_name]["aggregate"]["mean_ci95"] = [round(lo, 1), round(hi, 1)]
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
        "QTSR" per service = fraction of runs REQUIRING that service in which the
        service was actually queried and returned without error (service in
        queried_service_ids).

        Caution: this is a service-reached rate, not a translation success rate.
        It is 0 for any run in which the service was never selected, was
        abstained on by the translator, was dropped by the validator, or
        failed at execution, and it is structurally low for single-source
        baselines. The decomposition into those causes is in the
        `drop_breakdown` output (requires schema-2 records with a trace).
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
        for sys_name in self._systems:
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
        for sys_name in self._systems:
            sys_records = [r for r in records
                           if r["system_name"] == sys_name
                           and r["query_class_label"] in CROSS_SERVICE_CLASSES]
            total    = len(sys_records)
            conflict = sum(1 for r in sys_records if r.get("conflict_count", 0) > 0)
            result[sys_name] = round(conflict / total, 4) if total > 0 else 0.0
        return result

    # ------------------------------------------------------------------
    # Service-selection precision / recall / F1
    # ------------------------------------------------------------------

    def _compute_service_prf(self, records: list[dict]) -> dict:
        """
        queried_*  : P/R/F1 of the services whose query was executed successfully
                     (queried_service_ids) vs the required set.
        selected_* : P/R/F1 of the services the system chose to translate
                     (trace.shortlisted_service_ids; schema-2 records only).
        """
        out: dict = {}
        for sys_name in self._systems:
            rs = [r for r in records if r["system_name"] == sys_name]
            if not rs:
                continue
            def _avg(vals): return round(statistics.mean(vals), 4) if vals else None
            q = [compute_service_prf(r.get("required_services", []), r.get("queried_service_ids", []))
                 for r in rs]
            sel = [compute_service_prf(r.get("required_services", []),
                                       r["trace"]["shortlisted_service_ids"])
                   for r in rs if r.get("trace", {}).get("shortlisted_service_ids") is not None]
            out[sys_name] = {
                "queried_precision": _avg([x[0] for x in q]),
                "queried_recall":    _avg([x[1] for x in q]),
                "queried_f1":        _avg([x[2] for x in q]),
                "selected_precision": _avg([x[0] for x in sel]),
                "selected_recall":    _avg([x[1] for x in sel]),
                "selected_f1":        _avg([x[2] for x in sel]),
                "n": len(rs), "n_with_trace": len(sel),
            }
        return out

    # ------------------------------------------------------------------
    # Where do required services get lost?
    # ------------------------------------------------------------------

    def _compute_drop_breakdown(self, records: list[dict]) -> dict:
        """
        For every (system, required service) pair, classify the outcome:
          not_selected     service never reached translation (selection filtered it out)
          abstain          translator wrote CANNOT_ANSWER (and validator retry did not recover it)
          invalid_query    structurally invalid after the one retry
          translation_error  the LLM call failed (should be ~0 after retry/backoff)
          retrieval_error  query sent but failed at the service (e.g. unknown column)
          empty            query ran, returned 0 rows
          ok               query ran and returned rows
        Counts are per (run, required service). Schema-2 records only.
        """
        out: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        for r in records:
            tr = r.get("trace")
            if not tr or "shortlisted_service_ids" not in tr:
                continue
            short_sel  = {_short_name(x): x for x in tr["shortlisted_service_ids"]}
            dropped    = {_short_name(k): v for k, v in tr.get("dropped", {}).items()}
            retrieval  = {_short_name(k): v for k, v in tr.get("retrieval", {}).items()}
            for svc in r.get("required_services", []):
                if svc not in short_sel:
                    cause = "not_selected"
                elif svc in dropped:
                    cause = dropped[svc]
                elif svc in retrieval:
                    ret = retrieval[svc]
                    cause = ("retrieval_error" if not ret.get("succeeded")
                             else "empty" if ret.get("rows", 0) == 0 else "ok")
                else:
                    cause = "not_selected"
                out[r["system_name"]][svc][cause] += 1
                out[r["system_name"]][svc]["n"] += 1
        return {a: {b: dict(c) for b, c in d.items()} for a, d in out.items()}

    # ------------------------------------------------------------------
    # Cost: end-to-end latency, LLM calls, tokens
    # ------------------------------------------------------------------

    def _compute_cost(self, records: list[dict]) -> dict:
        """End-to-end (LLM time included) latency and LLM usage per run."""
        out: dict = {}
        for sys_name in self._systems:
            rs = [r for r in records if r["system_name"] == sys_name and "e2e_ms" in r]
            if not rs:
                continue
            e2e   = [r["e2e_ms"] for r in rs]
            calls = [r.get("llm", {}).get("calls", 0) for r in rs]
            toks  = [r.get("llm", {}).get("prompt_tokens", 0) + r.get("llm", {}).get("reply_tokens", 0)
                     for r in rs]
            retr  = [r.get("llm", {}).get("retries", 0) for r in rs]
            out[sys_name] = {
                "n": len(rs),
                "e2e_ms_mean": round(statistics.mean(e2e), 1),
                "e2e_ms_p50":  round(percentile(e2e, 50), 1),
                "e2e_ms_p95":  round(percentile(e2e, 95), 1),
                "llm_calls_mean": round(statistics.mean(calls), 2),
                "tokens_mean":    round(statistics.mean(toks), 1),
                "llm_retries_total": int(sum(retr)),
            }
        return out

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
            for sys_name in self._systems:
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
            for sys_name in self._systems:
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
            for sys_name in self._systems:
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
            for sys_name in self._systems:
                q = qtsr.get(sys_name, {})
                w.writerow([
                    sys_name,
                    q.get("sql", ""),
                    q.get("graph", ""),
                    q.get("document", ""),
                    icr.get(sys_name, ""),
                ])
        log.info("Written: %s", path)

    def _write_prf_csv(self, prf: dict) -> None:
        path = self._out / "service_selection_prf.csv"
        cols = ["queried_precision", "queried_recall", "queried_f1",
                "selected_precision", "selected_recall", "selected_f1", "n", "n_with_trace"]
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["system"] + cols)
            for sys_name, v in prf.items():
                w.writerow([sys_name] + [v.get(c, "") for c in cols])
        log.info("Written: %s", path)

    def _write_drop_csv(self, drops: dict) -> None:
        path = self._out / "drop_breakdown.csv"
        causes = ["ok", "empty", "retrieval_error", "invalid_query", "abstain",
                  "translation_error", "not_selected"]
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["system", "required_service", "n"] + causes)
            for sys_name, by_svc in drops.items():
                for svc, c in by_svc.items():
                    w.writerow([sys_name, svc, c.get("n", 0)] + [c.get(k, 0) for k in causes])
        log.info("Written: %s", path)

    def _write_cost_csv(self, cost: dict) -> None:
        path = self._out / "cost_per_query.csv"
        cols = ["n", "e2e_ms_mean", "e2e_ms_p50", "e2e_ms_p95",
                "llm_calls_mean", "tokens_mean", "llm_retries_total"]
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["system"] + cols)
            for sys_name, v in cost.items():
                w.writerow([sys_name] + [v.get(c, "") for c in cols])
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
            for sys_name in self._systems:
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
