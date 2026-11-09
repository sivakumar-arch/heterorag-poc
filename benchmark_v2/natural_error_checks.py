"""
benchmark_v2/natural_error_checks.py
====================================
Candidate static checks for translated queries, written after reading the silent errors that weak local
models produced on the CP1b questions (capability-floor runs, 2026-11-07). The code now lives in
`heterorag/federation/semantic_checks.py` (switched on by HETERORAG_SEMANTIC_CHECKS=1); this module re-exports it so that
`replay_checks` can apply the functions to queries that were already
recorded, so that false rejections and catches can be counted without a model call.

Three checks, each aimed at one class seen in the rows:

  R1 relationship signature (this is the registered mechanism M2): a relationship pattern must follow a
     (start label, type, end label) triple of the graph descriptor. Seen: `(t:Tag)-[:TAGGED_WITH]->(:Question)`
     and `(q:Question)-[:ASKED]->(u)`, both reversed, which return the empty set.
  R2 bind entity: the column (SQL) or property (Cypher) that is compared with a bound placeholder must belong
     to the entity of the bind. Seen: `users.id = ANY(%(s1)s)` where s1 holds question identifiers.
  R3 neutralised condition: `OR TRUE` / `OR 1=1` makes a condition vacuous. Seen: `reputation >= 3648 AND
     (users.id = ANY(%(s1)s) OR TRUE)`, which keeps the placeholder the validator demands and ignores it.

R2 and R3 were identified after the rows were read (post hoc); R1 was named before (M2 of the F2 record).
"""

from __future__ import annotations

from heterorag.federation.semantic_checks import (  # noqa: F401  (re-exported for replay_checks and the tests)
    CHECKS, Signatures, candidate_errors, cypher_bind_entity_errors, join_sanity_errors, negated_bind_errors, neutralised_condition_errors,
    relationship_signature_errors, sql_bind_entity_errors)
