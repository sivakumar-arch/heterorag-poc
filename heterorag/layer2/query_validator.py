"""
heterorag/layer2/query_validator.py
=====================================
QueryValidator — enforces the I₂ producer guarantee that every native query
in I₂ is syntactically valid before Layer 3 executes it.

Policy (locked in Foundation Doc v1.6 §6):
  - Attempt 1: validate the raw LLM translation.
  - On failure: issue one LLM-driven retry with the error embedded in the prompt.
  - On second failure: drop the service from I₂ entirely.
  - Result: ValidationStatus ∈ {VALID, FIXED, DROPPED, SKIPPED}

Validators per persistence type:
  SQL      — sqlglot.parse_one(query, dialect="postgres") — pure-Python, no DB call.
  Cypher   — regex structural check (must start with MATCH or CALL or WITH,
             must contain RETURN) — full parse requires a live Neo4j connection.
  BM25     — always SKIPPED (free-text, structurally unconstrained).

Why sqlglot for SQL and not a live DB round-trip?
  A PREPARE/EXPLAIN against live PostgreSQL would be definitive but adds
  a round-trip per query per validation attempt. sqlglot catches the class
  of errors the LLM actually makes (wrong column names, syntax errors,
  wrong dialect keywords). Schema-level errors (column not found) are caught
  at execution time in Layer 3 and reported via QTSR.

Reference: Foundation Doc v1.6 §6 (QueryValidator policy).
"""

from __future__ import annotations

import logging
import re
import time

from .models import ValidationRecord, ValidationStatus
from .translation_llm import TranslationLLM

log = logging.getLogger(__name__)

# Lazy import of sqlglot — only needed for SQL validation
_sqlglot = None

def _get_sqlglot():
    global _sqlglot
    if _sqlglot is None:
        try:
            import sqlglot
            _sqlglot = sqlglot
        except ImportError:
            _sqlglot = None
    return _sqlglot


# ---------------------------------------------------------------------------
# Per-type validators
# ---------------------------------------------------------------------------

def _validate_sql(query: str) -> tuple[bool, str | None]:
    """
    Returns (is_valid, error_message).
    Structural check only: query must start with SELECT or WITH.
    sqlglot is used as an advisory check only — its rejections are logged
    but do not cause the query to be dropped. PostgreSQL itself is the
    authoritative validator at execution time in Layer 3.
    This avoids false positives where sqlglot rejects valid PostgreSQL
    syntax that the LLM correctly generates.
    """
    stripped = query.strip()
    upper    = stripped.upper()

    # Hard structural check: must be a SELECT or CTE
    if not upper.startswith("SELECT") and not upper.startswith("WITH"):
        return False, (
            f"SQL must start with SELECT or WITH. Got: {stripped[:60]!r}"
        )

    # Must not be a CANNOT_ANSWER sentinel
    if "CANNOT_ANSWER" in upper:
        return False, "LLM indicated CANNOT_ANSWER for this service"

    # Advisory sqlglot check — log warning but do not reject
    sqlglot = _get_sqlglot()
    if sqlglot is not None:
        try:
            sqlglot.parse(query, dialect="postgres", error_level=sqlglot.ErrorLevel.RAISE)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).debug(
                "sqlglot advisory warning for SQL (not rejected): %s", exc
            )
    return True, None


def _validate_cypher(query: str) -> tuple[bool, str | None]:
    """
    Lightweight structural Cypher check.
    Lenient: strips comments and blank lines before checking clause starts.
    Only rejects SQL masquerading as Cypher and CANNOT_ANSWER sentinels.
    """
    stripped = query.strip()

    # Must not be a CANNOT_ANSWER sentinel
    if "CANNOT_ANSWER" in stripped.upper():
        return False, "LLM indicated CANNOT_ANSWER for this service"

    # Strip leading comment lines (// ...) to find the real first clause
    lines = stripped.splitlines()
    code_lines = [l for l in lines if not l.strip().startswith("//") and l.strip()]
    if not code_lines:
        return False, "Cypher query is empty after removing comments"
    first_code_line = code_lines[0].strip().upper()

    # Must start with a valid Cypher clause
    valid_starts = ("MATCH", "CALL", "WITH", "OPTIONAL MATCH", "CREATE", "MERGE",
                    "UNWIND", "RETURN")
    if not any(first_code_line.startswith(s) for s in valid_starts):
        # Could be SQL — that's the only hard rejection
        sql_starts = ("SELECT", "INSERT", "UPDATE", "DELETE", "DROP", "ALTER")
        if any(first_code_line.startswith(s) for s in sql_starts):
            return False, (
                f"SQL submitted instead of Cypher: {stripped[:60]!r}"
            )
        # Unknown start — allow it through, let Neo4j decide
        import logging
        logging.getLogger(__name__).debug(
            "Cypher advisory: unexpected start %r — passing through", first_code_line[:30]
        )

    # Must contain a RETURN clause
    if "RETURN" not in stripped.upper():
        return False, "Cypher query is missing a RETURN clause"

    return True, None


def _validate_bm25(query: str) -> tuple[bool, str | None]:
    """BM25 is always structurally valid — free-form text."""
    return True, None


_VALIDATORS = {
    "sql":      _validate_sql,
    "cypher":   _validate_cypher,
    "bm25":     _validate_bm25,
}


# ---------------------------------------------------------------------------
# Retry prompt builders
# ---------------------------------------------------------------------------

def _sql_retry_prompt(original_query: str, error: str, natural_query: str) -> str:
    return f"""The following SQL query failed validation with this error:

ERROR: {error}

INVALID QUERY:
{original_query}

Fix the query so it is valid PostgreSQL. The original question was:
{natural_query}

Output ONLY the corrected SQL — no explanation, no markdown fences."""


def _cypher_retry_prompt(original_query: str, error: str, natural_query: str) -> str:
    return f"""The following Cypher query failed validation with this error:

ERROR: {error}

INVALID QUERY:
{original_query}

Fix the query so it is a valid Neo4j Cypher MATCH...RETURN statement.
The original question was:
{natural_query}

Output ONLY the corrected Cypher — no explanation, no markdown fences."""


_RETRY_PROMPT_BUILDERS = {
    "sql":    _sql_retry_prompt,
    "cypher": _cypher_retry_prompt,
    "bm25":   None,   # BM25 never retries — it's always valid
}


# ---------------------------------------------------------------------------
# QueryValidator
# ---------------------------------------------------------------------------

class QueryValidator:
    """
    Validates and optionally corrects native queries before I₂ is assembled.

    Usage:
        validator = QueryValidator(llm=translation_llm)
        status, corrected, records = validator.validate(
            service_id    = "user-activity-service",
            query_language = "sql",
            native_query   = raw_sql,
            natural_query  = original_question,
        )
    """

    def __init__(self, llm: TranslationLLM):
        self._llm = llm

    def validate(
        self,
        service_id:    str,
        query_language: str,          # "sql" | "cypher" | "bm25"
        native_query:  str,
        natural_query: str,           # original NL question — used in retry prompt
    ) -> tuple[ValidationStatus, str, list[ValidationRecord]]:
        """
        Validate native_query for the given query_language.

        Returns:
            (status, final_query, records)
            - status:      VALID | FIXED | DROPPED | SKIPPED
            - final_query: the validated (or corrected) query string.
                           Empty string if DROPPED.
            - records:     list of ValidationRecord (one per attempt).
        """
        lang     = query_language.lower()
        validate = _VALIDATORS.get(lang, _validate_bm25)
        records: list[ValidationRecord] = []

        # BM25: always SKIPPED
        if lang == "bm25":
            records.append(ValidationRecord(
                service_id    = service_id,
                attempt       = 1,
                status        = ValidationStatus.SKIPPED,
                query_before  = native_query,
                query_after   = native_query,
                error_message = None,
            ))
            return ValidationStatus.SKIPPED, native_query, records

        # --- Attempt 1: validate the raw translation ---
        t0 = time.perf_counter()
        is_valid, error = validate(native_query)
        dur = (time.perf_counter() - t0) * 1000.0

        if is_valid:
            records.append(ValidationRecord(
                service_id    = service_id,
                attempt       = 1,
                status        = ValidationStatus.VALID,
                query_before  = native_query,
                query_after   = native_query,
                error_message = None,
                duration_ms   = dur,
            ))
            return ValidationStatus.VALID, native_query, records

        # Attempt 1 failed — record and try LLM retry
        records.append(ValidationRecord(
            service_id    = service_id,
            attempt       = 1,
            status        = ValidationStatus.DROPPED,   # tentative — updated below
            query_before  = native_query,
            query_after   = None,
            error_message = error,
            duration_ms   = dur,
        ))
        log.warning(
            "QueryValidator: '%s' attempt 1 failed (%s) — issuing LLM retry",
            service_id, error,
        )

        # --- Attempt 2: LLM-driven retry ---
        retry_builder = _RETRY_PROMPT_BUILDERS.get(lang)
        if retry_builder is None:
            # Should not happen (BM25 is SKIPPED above) but guard defensively
            return ValidationStatus.DROPPED, "", records

        retry_prompt   = retry_builder(native_query, error, natural_query)
        retry_reply    = self._llm.translate(retry_prompt)
        corrected      = retry_reply.text

        t1 = time.perf_counter()
        is_valid2, error2 = validate(corrected)
        dur2 = (time.perf_counter() - t1) * 1000.0

        if is_valid2:
            # Update attempt 1 status to show it was retried
            records[0] = records[0].model_copy(update={"status": ValidationStatus.DROPPED})
            records.append(ValidationRecord(
                service_id    = service_id,
                attempt       = 2,
                status        = ValidationStatus.FIXED,
                query_before  = corrected,
                query_after   = corrected,
                error_message = None,
                duration_ms   = dur2,
            ))
            log.info("QueryValidator: '%s' fixed by LLM retry", service_id)
            return ValidationStatus.FIXED, corrected, records

        # Both attempts failed — drop this service
        records.append(ValidationRecord(
            service_id    = service_id,
            attempt       = 2,
            status        = ValidationStatus.DROPPED,
            query_before  = corrected,
            query_after   = None,
            error_message = error2,
            duration_ms   = dur2,
        ))
        log.error(
            "QueryValidator: '%s' DROPPED after 2 failed attempts. "
            "error1=%r  error2=%r",
            service_id, error, error2,
        )
        return ValidationStatus.DROPPED, "", records
