# Rerun-hardening patch: what changed and why

Purpose: make the benchmark harness safe to rerun for the TKDE paper. Nothing in
this patch changes retrieval quality on purpose; it changes what is *measured*,
*recorded* and *retried*. Every number from the earlier runs should be treated as
superseded by a rerun made with this code.

## Problems in the previous harness (all verified in the code)

| # | Problem | Effect on reported numbers |
|---|---------|----------------------------|
| 1 | `AnthropicProvider.complete` had no retry. A 429 raised inside the planner thread pool became a placeholder query, failed validation, and was recorded as a **dropped service**. | Rate-limit errors were indistinguishable from translation failures (lowers SC/QTSR, inflates "SQL drop rate"). For BM25 the placeholder passed validation and was executed as the search string; a 429 inside the validator's retry call aborted the whole question. |
| 2 | Runner `except` wrote an `[ERROR ...]` record and `_load_completed()` counted it as done. | Resuming never retried failures; error rows were scored as real answers (SC = 0, CANNOT_ANSWER). |
| 3 | Validator treats the `CANNOT_ANSWER` sentinel (which the SQL/Cypher prompts tell the LLM to emit when a service cannot answer) as an invalid query, spends a retry on it, then drops the service. | "Drop" mixes legitimate abstention with real failures. Needs a breakdown before QTSR can be interpreted. |
| 4 | Old B3 differed from HeteroRAG in many ways besides "sequential": one JSON routing+translation prompt with no per-service schema, no validation, no `SemanticIntegrator`, 20-row cap, falls back to the raw NL question as the native query. | B3-vs-HeteroRAG latency/coverage gaps could not be attributed to the sequential schedule. |
| 5 | `queried_service_ids` was the only per-service signal. | Could not tell not-selected / abstained / invalid / LLM error / query error / empty result apart. |
| 6 | SC is recall-only; B4/HeteroRAG_Full query everything, so SC = 1 by construction. | SC cannot distinguish selection quality. |
| 7 | One run per cell, no CIs; retrieval-only latency. | No uncertainty, no end-to-end cost. |

## What the patch does

- `llm_provider.py`: `RetryingProvider` (exponential backoff + jitter, honours
  `Retry-After`; retries 429/5xx/529/connection errors) wraps every provider, and
  a process-wide `LLM_USAGE` meter counts calls, retries, tokens and time.
  Env: `HETERORAG_LLM_MAX_RETRIES` (8), `HETERORAG_LLM_BACKOFF_BASE_S` (2),
  `HETERORAG_LLM_BACKOFF_MAX_S` (60).
- `layer2`: an LLM *call failure* is a `translation_error` (no validator retry,
  run flagged as infra error). Dropped services carry a reason:
  `abstain | invalid_query | translation_error`. Optional
  `abstain_short_circuit` (default **off**, so system behaviour is unchanged).
- `layer3/retrieval_executor.py`: `execution_mode="sequential"` (sum of service
  times, same code path otherwise); an `as_completed` timeout no longer aborts the
  whole question.
- `layer4/trace.py` + `pipeline.py`: each run records shortlisted / translated /
  dropped(+reason) / validation attempts / per-service retrieval outcome (rows,
  ms, error) / stage timings / `infra_error`. B1 and B2 record the same.
- `evaluation/systems.py`: systems differ along two named axes only.

  | System | selection | schedule |
  |---|---|---|
  | `HeteroRAG_Full` (== old `B4_Fixed_Plan`) | all services | parallel |
  | `Select_Parallel` | LLM picks services | parallel |
  | `Select_Sequential` (**replaces old B3**) | LLM picks services | sequential |
  | `B1_SQL_Only`, `B2_Document_Only` | fixed single service | n/a |

  Old B3 is kept as `B3_LLM_FunctionCalling_Legacy`, old B4 as `B4_Fixed_Plan`,
  both opt-in via `--systems`, only for reproducing earlier numbers.
- `benchmark_runner.py`: records are schema 2 (`status`, `repeat`, `attempt`,
  `trace`, `llm`, `e2e_ms`). Only `status == "ok"` counts as completed, so
  resuming retries failures. Per-run retry (`--max-attempts`, 15/60/180 s
  backoff), `--repeats`, a circuit breaker after 5 consecutive failed runs (dead
  database or key), and `run_meta.jsonl` (git commit, dirty flag, config, Postgres
  row counts, Neo4j node/relationship counts, Elasticsearch doc count).
- `metrics.py`: failed runs are excluded and counted (`data_quality` in the JSON,
  warning in `compute_metrics.py`); latest ok record wins; question-clustered
  bootstrap 95% CIs for SC and RL mean; precision/recall/F1 of service selection
  (`service_selection_prf.csv`); `drop_breakdown.csv` (why each required service
  was or was not reached); `cost_per_query.csv` (end-to-end latency, LLM calls,
  tokens, retries). The old "QTSR" is documented as a *service-reached rate*.

## How to rerun

```bash
python evaluation/run_benchmark.py --output-dir results/run1 --repeats 3
# re-run the SAME command after any interruption: only failed/missing runs execute
python evaluation/compute_metrics.py --results-dir results/run1
```
Read `data_quality` and `drop_breakdown.csv` **before** quoting any SC/AF number.

## Not done here (deliberately)

- B3 as true multi-turn tool use. `LLMProvider` is single-turn text; a real
  tool-calling baseline needs a provider extension. `Select_Sequential` is the
  controlled comparison; add a tool-use agent as a separate, clearly named baseline.
- Stage 1 (embedding) selection. With three services it cannot do anything; it
  only matters at mesh scale.
- AF is still a regex over 4+ digit numbers in the answer text, and the 120
  questions are Stack Overflow specific. Benchmark v2 with programmatic ground
  truth is a separate work item.
- `abstain_short_circuit` is a switch, not a decision. Decide after reading
  `drop_breakdown.csv` from the first clean run.
