"""
heterorag/layer4/trace.py
==========================
Per-run diagnostic trace.

The original raw records kept only `queried_service_ids`, which cannot tell
apart (a) a service the planner never selected, (b) a service whose LLM
translation was an abstention, (c) an invalid query, (d) an LLM/API failure,
and (e) a query that was sent but failed or returned nothing at the service.
Those five cases have very different scientific meaning, so every run now
records them explicitly.

The trace is plain JSON-serialisable data attached to GenerationResult.trace.
"""

from __future__ import annotations

from typing import Any

from heterorag.layer1.models import I1_ServiceDiscoveryOutput
from heterorag.layer2.models import I2_QueryTranslationOutput
from heterorag.layer3.models import RetrievalBatch

# Retrieval-side error types that indicate the *infrastructure* misbehaved
# (and so the run should be excluded from metrics and retried), as opposed to
# the generated query being wrong (syntax error, unknown column, ...), which is
# a legitimate, reportable outcome of the system under test.
INFRA_RETRIEVAL_ERRORS = {
    "Timeout", "ConfigError", "UnexpectedError",
    "OperationalError", "InterfaceError",                      # psycopg2 connection
    "ServiceUnavailable", "SessionExpired", "AuthError",       # neo4j
    "ConnectionError", "ConnectionTimeout", "TransportError",  # elasticsearch
    "ConnectionRefusedError", "TimeoutError", "ReadTimeout", "ConnectTimeout",
    "ModuleNotFoundError", "ImportError",                      # driver not installed
}

_MAX_TEXT = 400


def _clip(s: str | None, n: int = _MAX_TEXT) -> str | None:
    if s is None:
        return None
    return s if len(s) <= n else s[:n] + "…"


def build_trace(
    i1:        I1_ServiceDiscoveryOutput,
    i2:        I2_QueryTranslationOutput,
    batch:     RetrievalBatch,
    timings_ms: dict[str, float],
) -> dict[str, Any]:
    retrieval: dict[str, dict[str, Any]] = {}
    infra_reasons: list[str] = []

    for r in batch.results:
        entry: dict[str, Any] = {
            "succeeded": bool(r.succeeded),
            "rows":      len(r.rows),
            "ms":        round(r.retrieval_ms, 2),
            "query":     _clip(r.native_query),
        }
        if r.error is not None:
            entry["error_type"]    = r.error.error_type
            entry["error_message"] = _clip(r.error.error_message)
            if r.error.error_type in INFRA_RETRIEVAL_ERRORS:
                infra_reasons.append(f"retrieval:{r.service_id}:{r.error.error_type}")
        retrieval[r.service_id] = entry

    for sid, msg in i2.translation_errors.items():
        infra_reasons.append(f"translation:{sid}:{_clip(msg, 80)}")

    validation = [
        {
            "service_id": v.service_id,
            "attempt":    v.attempt,
            "status":     v.status.value,
            "error":      _clip(v.error_message),
        }
        for v in i2.validation_log
    ]

    return {
        "shortlisted_service_ids": [d.service_id for d in i1.descriptors],
        "translated_service_ids":  i2.service_ids(),
        "dropped":                 dict(i2.drop_reasons),
        "dropped_translations":    {k: _clip(v) for k, v in i2.dropped_translations.items()},
        "validation":              validation,
        "validation_retries":      sum(1 for v in i2.validation_log if v.attempt > 1),
        "retrieval":               retrieval,
        "ms":                      {k: round(v, 2) for k, v in timings_ms.items()},
        "infra_error":             bool(infra_reasons),
        "infra_reasons":           infra_reasons,
    }
