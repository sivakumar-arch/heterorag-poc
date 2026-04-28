# HeteroRAG POC — RUNBOOK

**Version:** 1.0  
**Dataset:** Stack Overflow data dump (Stack Exchange CC BY-SA licence)  
**Services:** PostgreSQL 16 · Neo4j 5.18 · Elasticsearch 8.13

This document provides the complete eight-step reproducible setup for the HeteroRAG POC simulation environment. Any researcher following these steps on a machine with Docker and Python 3.10+ should be able to reproduce all benchmark results.

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Docker | ≥ 24.0 | Docker Engine or Docker Desktop |
| Docker Compose | ≥ 2.20 | Bundled with Docker Desktop |
| Python | ≥ 3.10 | For loader and evaluation scripts |
| pip packages | — | psycopg2-binary, elasticsearch, neo4j |
| RAM | ≥ 8 GB | 16 GB recommended for full SO dataset |
| Disk | ≥ 30 GB | Full Stack Overflow dump is ~22 GB uncompressed |

### Download the Stack Exchange data dump

```bash
# Full Stack Overflow dump (~22 GB compressed)
# Source: https://archive.org/details/stackexchange
# File: stackoverflow.com.7z (or the per-file archives)

mkdir -p ./data
# After downloading and extracting, the data directory should contain:
#   Users.xml  Posts.xml  Comments.xml  Votes.xml
#   Tags.xml   Badges.xml PostLinks.xml
```

> **Smaller dataset option:** For faster iteration during development, use a smaller Stack Exchange community dump (e.g. `stats.stackexchange.com` — ~500 MB uncompressed). The schema is identical. Set `DATA_DIR=./data` pointing to that community's extracted files. All benchmark questions and ground truth views will work on any Stack Exchange dataset.

---

## Step 1 — Start all three services

```bash
docker compose up -d
```

This starts:
- **PostgreSQL** on port 5432 (`heterorag` database, user `heterorag`)
- **Neo4j** on port 7474 (browser) and 7687 (Bolt)
- **Elasticsearch** on port 9200

Wait for all health checks to pass before proceeding:

```bash
docker compose ps
# All three services should show status: healthy
# This typically takes 30–60 seconds on first start.
```

To verify manually:
```bash
# PostgreSQL
pg_isready -h localhost -U heterorag -d heterorag

# Neo4j
curl -s http://localhost:7474 | grep -q neo4j && echo "Neo4j ready"

# Elasticsearch
curl -s http://localhost:9200/_cluster/health | python3 -m json.tool
```

---

## Step 2 — Load the Stack Overflow data dump

```bash
./scripts/load_stackoverflow_dump.sh
```

This script:
1. Checks that all seven XML files are present in `./data`
2. Streams each XML file with `iterparse` (memory-efficient — never loads full file)
3. Loads into **PostgreSQL**: Users, Posts (metadata), Votes, Badges, Tags, Comments (metadata), PostLinks
4. Loads into **Neo4j**: User/Question/Answer/Tag nodes; ASKED, ANSWERED, ANSWERS, ACCEPTED, TAGGED_WITH, LINKED_TO, DUPLICATE_OF edges; derives CO_OCCURS_WITH from tag co-occurrence
5. Loads into **Elasticsearch**: Question bodies, Answer bodies, Comment text, User AboutMe, Tag wiki excerpts
6. Tracks progress in PostgreSQL `_load_progress` table — **resumable on failure**

**Environment variables (optional overrides):**

```bash
DATA_DIR=./data \
BATCH_SIZE=2000 \
ONLY=postgres \        # postgres | neo4j | elasticsearch | (omit for all)
./scripts/load_stackoverflow_dump.sh
```

**Expected duration:**

| Dataset | Approximate load time |
|---|---|
| stats.stackexchange.com (~1M posts) | 5–15 minutes |
| Full Stack Overflow (~60M posts) | 3–6 hours |

**Dry-run validation (no writes):**

```bash
DRY_RUN=1 ./scripts/load_stackoverflow_dump.sh
```

---

## Step 3 — Apply SQL schema migrations and ground truth views (Flyway)

```bash
docker compose --profile migrate up flyway
```

This applies Flyway versioned migrations from `./sql/migrations/`:

| File | Content |
|---|---|
| `V1__ground_truth_views.sql` | 20 SQL-only ground truth views (`gt_c1_q01` … `gt_c1_q20`) plus SQL components of cross-service questions |

Flyway tracks applied migrations in the `flyway_schema_history` table and is idempotent — re-running applies only new migrations.

Verify:
```bash
psql "host=localhost port=5432 dbname=heterorag user=heterorag password=heterorag_secret" \
  -c "\dv gt_*"
# Should list all ground truth views
```

---

## Step 4 — Apply Graph schema migrations (Liquibase)

```bash
liquibase \
  --url=jdbc:neo4j:bolt://localhost:7687 \
  --username=neo4j \
  --password=heterorag_secret \
  --changeLogFile=graph/migrations/001_ground_truth_fixtures.cypher \
  update
```

This creates `GroundTruth` fixture nodes in Neo4j for Class 2 (Graph-only) and cross-service questions. These nodes store the expected Cypher templates and metadata used by the evaluation harness.

> **Note:** Liquibase Neo4j support requires the `liquibase-neo4j` extension. Install via:
> ```bash
> liquibase --version   # verify Liquibase is installed
> # Download liquibase-neo4j plugin from https://github.com/liquibase/liquibase-neo4j/releases
> # Place the JAR in your Liquibase lib/ directory
> ```

Verify:
```bash
# In Neo4j Browser (http://localhost:7474)
MATCH (gt:GroundTruth) RETURN gt.id, gt.description LIMIT 20
```

---

## Step 5 — Create Elasticsearch index

```bash
python ground-truth/document/setup_index.py
```

Creates the `heterorag_content` index with custom English-language BM25 mappings. Idempotent — no-op if the index already exists.

Verify:
```bash
curl -s http://localhost:9200/heterorag_content | python3 -m json.tool
```

---

## Step 6 — Load document corpus into Elasticsearch

The document corpus was loaded in Step 2. This step verifies the document count and optionally regenerates if needed.

```bash
# Verify document count
curl -s "http://localhost:9200/heterorag_content/_count" | python3 -m json.tool

# Reload documents only (if Step 2 was run with --only postgres or neo4j)
ONLY=elasticsearch ./scripts/load_stackoverflow_dump.sh
```

---

## Step 7 — Generate document ground truth fixtures

# Optional — fixtures already included in repo for the three Stack Exchange communities.
# Only required if you switch to a different dataset or want to regenerate.
python ground-truth/document/run_gt_queries.py --verify    # confirm fixtures are present
python ground-truth/document/run_gt_queries.py --generate-fixtures  # regenerate if needed

Runs the BM25 retrieval queries for Class 3 (Document-only) benchmark questions against the live Elasticsearch index. Stores expected document ID + passage offset sets as fixture files in `ground-truth/document/fixtures/`.

**Run this exactly once** against the fixed dataset. The fixture files become the ground truth for all subsequent evaluations. Committing them to git ensures reproducibility.

Verify:
```bash
ls ground-truth/document/fixtures/
# Should contain: gt_c3_q01.json … gt_c3_q20.json
```

---

## Step 8 — Execute the full benchmark

```bash
python evaluation/run_benchmark.py
```

Runs all 120 benchmark questions against HeteroRAG and all four baselines:
- B1: SQL-only router
- B2: Document-only RAG
- B3: LLM function-calling router
- B4: HeteroRAG fixed-plan ablation

Computes all metrics:
- Source Coverage (SC) per query class
- Answer Faithfulness (AF) per query class and metric variant (F1 / NDCG@10 / Recall@10)
- Retrieval Latency (RL): mean, p50, p95, p99
- Query Translation Success Rate (QTSR) per service
- Integration Conflict Rate (ICR)

Outputs:
- `results/benchmark_results.json` — full per-question results
- `results/benchmark_summary.csv` — aggregate table for paper tables
- `results/latency_distribution.csv` — latency percentile data

---

## Resetting the environment

```bash
# Remove all containers and volumes (destructive — data will be lost)
docker compose down -v

# Remove generated fixtures and results
rm -rf ground-truth/document/fixtures/ results/

# Restart from Step 1
docker compose up -d
```

---

## Environment variables reference

| Variable | Default | Description |
|---|---|---|
| `PG_DSN` | `host=localhost port=5432 dbname=heterorag user=heterorag password=heterorag_secret` | PostgreSQL connection string |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j Bolt URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | `heterorag_secret` | Neo4j password |
| `ES_URL` | `http://localhost:9200` | Elasticsearch URL |
| `DATA_DIR` | `./data` | Stack Exchange XML dump directory |
| `BATCH_SIZE` | `1000` | Write batch size for loader |
| `ONLY` | *(all)* | Restrict loader to one service |
| `DRY_RUN` | *(false)* | Parse without writing |

---

## Troubleshooting

**Neo4j health check fails on first start**  
Neo4j 5.x takes 45–90 seconds to initialise on first boot. Wait for `docker compose ps` to show `healthy` before proceeding.

**`pg_isready` reports "no pg_hba.conf entry"**  
The PostgreSQL container uses trust authentication internally. Connect from the host using the credentials in the DSN above.

**Elasticsearch returns `yellow` cluster health**  
Expected for a single-node cluster — `yellow` means all shards are allocated but no replicas. This is correct for this setup (`number_of_replicas: 0`).

**Load script fails partway through**  
The loader is resumable. Re-run `./scripts/load_stackoverflow_dump.sh` — already-completed files are tracked in `_load_progress` and will be skipped.

**Out of memory during CO_OCCURS_WITH derivation**  
The co-occurrence derivation query scans the full tag graph. On the full Stack Overflow dataset, increase Neo4j heap: set `NEO4J_server_memory_heap_max__size=4g` in `docker-compose.yml`.

---

*This RUNBOOK is the reproducibility contract for the HeteroRAG POC. Any researcher should be able to execute Steps 1–8 and obtain identical benchmark results against the same Stack Exchange dataset.*
