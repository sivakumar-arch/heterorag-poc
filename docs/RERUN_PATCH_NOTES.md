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

## Data-load fixes found while bringing up a fresh stack (batch 2)

Found when loading the `stats` community on a new machine and validating every count
against the source XML. Each item below produced a silent wrong result, not an error.

| Problem | Effect | Fix |
|---|---|---|
| `--dry-run` recorded files as complete in `_load_progress` | The next real run skipped PostgreSQL entirely, leaving empty tables | `mark_complete` is skipped on dry runs |
| Current dumps store tags as `\|python\|pandas\|`; code assumed `<python><pandas>` | Neo4j and Elasticsearch received one garbage tag per question; SQL views filtering `'%<python>%'` matched nothing; the SQL translation prompt told the LLM to use the same pattern | Shared `heterorag/tags.py::parse_tags` (both encodings); descriptor now documents `tags LIKE '%\|tag\|%'`; V2 migration replaces the four tag views |
| `TAGGED_WITH` edges were built before `Question` nodes existed | Zero edges, no error; the log reported parsed pairs as "loaded" | Edges are built after posts; the log reports edges actually created and the load fails if none exist |
| Elasticsearch labelled every non-question post as `answer` | About 1% extra "answers" that were tag wikis and excerpts | Only post types 1 and 2 are indexed as questions and answers |
| Neo4j healthcheck used `curl`, which the image lacks | Container permanently `unhealthy` although the database was fine | Healthcheck uses `cypher-shell` |

### Applying to a stack that was loaded with the old code

1. `docker compose --profile migrate up flyway` applies `V2` (the existing V1 record is untouched).
2. Elasticsearch must be rebuilt, because its documents carry the wrong tags and extra "answers":
   delete the index, then `python scripts/load_stackoverflow_dump.py --data-dir data/stats --community-name stats --only elasticsearch`.
3. Neo4j needs no reload if `scripts/fix_neo4j_edges.py` has already been run (it parses both encodings).

### Known benchmark mismatches (not data bugs)

`gt_c1_q17` references a Stack Overflow question id that does not exist in other communities,
and `gt_c1_q18` (reputation = 0) is empty on any Stack Exchange site because accounts start at
reputation 1. These need re-parameterising in benchmark v2.

## Evaluation fixes found by the mock run (batch 3)

| Problem | Effect | Fix |
|---|---|---|
| A failed ground-truth view lookup left the PostgreSQL transaction aborted | Every later lookup failed too, including views that exist (`gt_c4b_q01_sql`, `gt_c5_q01_sql`) | `rollback()` after a failed lookup |
| 42 of the 45 multi-service questions (`c4a_q02`..`q15`, `c4b_q02`..`q15`, `c5_q02`..`q15`) name ground-truth views that are not defined in any migration | AF was scored 0.0 for them, so "no ground truth" looked like "wrong answer" | AF is `None` when a declared ground-truth component is missing or empty; such runs are excluded from the mean and reported under `data_quality.af_ground_truth` and in the `compute_metrics.py` output |

### Open issue, not fixed here: AF matching

`set_f1` and `ndcg_at_k` compare the answer's extracted numbers (any run of 4 or more digits)
against ground-truth rows rendered as strings such as `id=919|display_name=whuber|reputation=322774`.
A bare number can never equal such a string, so for SQL and graph questions the overlap is empty
and AF is near zero whatever the system answered. AF therefore needs a redesign (compare entity
keys, not rendered rows, and score against the retrieved rows rather than a regex over the
answer) before it can be reported. This belongs to benchmark v2.

### Smoke testing with a real LLM

`run_benchmark.py --question-ids c1_q05,c2_q01,c3_q01,c4a_q01,c5_q01` runs only those questions
(one per class, and the two multi-service ones that have SQL ground truth). An unknown id is an
error. Use a separate `--output-dir` so smoke results never mix with the full run.

### Model selection (batch 5)

* `claude-sonnet-4-20250514`, the model of the original evaluation and the former code default,
  was retired by Anthropic on 2026-06-15. The default `AnthropicProvider.DEFAULT_MODEL` is now
  `claude-sonnet-5-5`. A like-for-like rerun with the original model is no longer possible.
* `HETERORAG_LLM_MODEL` used to be honoured only when `HETERORAG_LLM_PROVIDER` was also set;
  set alone, it was silently ignored. It now works on its own.
* `run_meta.jsonl` records the model that was actually selected (`llm_model`), not the text
  "(provider default)".
* For anything reported, pin the model explicitly: `export HETERORAG_LLM_MODEL=claude-sonnet-5-5`.

### Models that do not accept `temperature` (batch 6)

With `claude-sonnet-5-5` and a current `anthropic` SDK, `messages.create(temperature=...)` raised
`TypeError: unexpected keyword argument 'temperature'`; other newer models answer HTTP 400. The
provider now sends `temperature` first and, if the SDK or API refuses it, retries once without it
and stops sending it for the rest of the run. Any other error is re-raised unchanged.
Consequence for the study: decoding is then at the model default and not under our control, so
report these results over repeated runs (`--repeats 3`) with confidence intervals.
