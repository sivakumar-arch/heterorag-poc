# HeteroRAG — Proof-of-Concept Simulation

**Paper:** HeteroRAG: A Federated Heterogeneous Retrieval Framework for Natural Language Querying across Microservice Architectures  
**Target venue:** IEEE Transactions on Knowledge and Data Engineering / Springer JIIS  
**Licence:** MIT (code) · CC BY 4.0 (benchmark questions and ground truth)  
**Data:** Stack Exchange CC BY-SA 4.0

---

## Overview

This repository contains the complete proof-of-concept simulation for Paper 1 of the HeteroRAG research series. It implements a four-layer federated heterogeneous RAG framework across three structurally distinct microservices and evaluates it against four baselines on a 120-question benchmark spanning seven query complexity classes.

```
┌──────────────────────────────────────────────┐
│  Layer 4: Generation (GenerationLLM)         │
├──────────────────────────────────────────────┤
│  Layer 3: Semantic Integration               │
│  Parallel Retrieval → Normalise → Dedupe → Rank │
├──────────────────────────────────────────────┤
│  Layer 2: Query Translation                  │
│  RelevanceFilter → TranslationLLM → Validator → Planner │
├──────────────────────────────────────────────┤
│  Layer 1: Service Discovery (ServiceRegistry) │
└──────────────────────────────────────────────┘
         ↑  ↑  ↑
   SQL  Graph  Document
```

---

## Three-Service Partition

| Service | Technology | Data |
|---|---|---|
| User & Activity Service | PostgreSQL 16 | Users, Posts, Votes, Tags, PostLinks |
| Knowledge Graph Service | Neo4j 5.18 | User/Question/Answer/Tag nodes; ASKED, ANSWERED, TAGGED_WITH, CO_OCCURS_WITH, LINKED_TO, DUPLICATE_OF edges |
| Content Service | Elasticsearch 8.12 | Question bodies, Answer bodies, User AboutMe, Tag wikis |

---

## Repository Structure

```
heterorag-poc/
├── heterorag/                    # Python package — four-layer implementation
│   ├── layer1/                   # Service Discovery
│   │   ├── models.py             # ServiceDescriptor, I1_ServiceDiscoveryOutput
│   │   ├── registry.py           # ServiceRegistry with heartbeat and cold start
│   │   └── poc_descriptors.py    # Three POC service descriptors
│   ├── layer2/                   # Query Translation
│   │   ├── translation_llm.py    # Anthropic API wrapper for NL→native translation
│   │   ├── query_validator.py    # Syntactic validation + LLM retry policy
│   │   ├── query_planner.py      # Weight attachment + I₂ construction
│   │   └── prompts.py            # Schema-aware translation prompt templates
│   ├── layer3/                   # Semantic Integration
│   │   ├── retrieval_executor.py # Parallel execution with thread pool
│   │   ├── result_normaliser.py  # SQL/Graph/Document → NormalisedContent
│   │   ├── entity_resolver.py    # Cross-source entity matching + deduplication
│   │   └── ranker.py             # weight × cosine_sim scoring → top-K
│   ├── layer4/                   # Generation
│   │   ├── generation.py         # GenerationLLM + GenerationResult
│   │   └── pipeline.py           # HeteroRAGPipeline (end-to-end)
│   └── evaluation/               # Baselines, benchmark runner, metrics
│       ├── baselines.py          # B1 SQL-Only, B2 Doc-Only, B3 LLM-FC, B4 Fixed-Plan
│       ├── benchmark_runner.py   # 120-question × 5-system orchestrator
│       └── metrics.py            # SC, AF, RL, QTSR, ICR computation
├── evaluation/                   # CLI entrypoints
│   ├── run_benchmark.py          # RUNBOOK step 8
│   └── compute_metrics.py        # Post-benchmark metric computation
├── tests/                        # Unit + integration tests
│   ├── layer1/                   # Layer 1 tests
│   ├── layer2/                   # Layer 2 tests
│   ├── layer3/                   # Layer 3 tests
│   └── test_steps_10_13.py       # Layer 4, baselines, benchmark, metrics tests
├── scripts/
│   ├── load_stackoverflow_dump.py   # Stack Exchange XML data loader
│   ├── load_stackoverflow_dump.sh   # Shell wrapper for single community
│   └── load_multi_community.sh      # Load multiple communities in sequence
├── sql/
│   ├── init/00_bootstrap.sql        # PostgreSQL schema bootstrap
│   └── migrations/V1__ground_truth_views.sql  # Flyway ground truth views
├── graph/
│   └── migrations/001_ground_truth_fixtures.cypher  # Liquibase GT fixtures
├── ground-truth/
│   └── document/
│       ├── setup_index.py           # Elasticsearch index creation
│       ├── run_gt_queries.py        # RUNBOOK step 7 — generate doc GT fixtures
│       └── seed_documents.py        # Verify document corpus
├── docker-compose.yml               # PostgreSQL + Neo4j + Elasticsearch
├── pyproject.toml                   # Python package configuration
└── RUNBOOK.md                       # Eight-step reproducible setup
```

---

## Prerequisites

- Docker Desktop (or Docker Engine + Compose v2)
- Python 3.11+
- An Anthropic API key ([console.anthropic.com](https://console.anthropic.com))
- ~10 GB free disk space for the three Stack Exchange communities

---

## Quick Start — Eight Steps

### Step 1 — Start services

```bash
docker compose up -d
docker compose ps   # wait until all three show (healthy)
```

### Step 2 — Download and load data

Download three Stack Exchange communities from the [Stack Exchange data dump](https://archive.org/details/stackexchange) (CC BY-SA):

- `stats.stackexchange.com` (~500 MB compressed)
- `dba.stackexchange.com` (~200 MB compressed)
- `datascience.stackexchange.com` (~150 MB compressed)

Extract each into `data/<community>/`:

```
data/
├── stats/        ← Users.xml, Posts.xml, Tags.xml, Votes.xml, PostLinks.xml
├── dba/          ← Users.xml, Posts.xml, Tags.xml
└── datascience/  ← Users.xml, Posts.xml, Tags.xml
```

Load all three communities:

```bash
chmod +x scripts/load_multi_community.sh
./scripts/load_multi_community.sh
```

### Step 3 — Apply SQL migrations (Flyway)

```bash
docker compose --profile migrate up flyway
```

### Step 4 — Apply Graph migrations (Liquibase)

```bash
liquibase --url=jdbc:neo4j:bolt://localhost:7687 \
  --username=neo4j --password=heterorag_secret \
  --changeLogFile=graph/migrations/001_ground_truth_fixtures.cypher update
```

### Step 5 — Set up Elasticsearch index

```bash
python ground-truth/document/setup_index.py
```

### Step 6 — Verify document corpus

```bash
python ground-truth/document/seed_documents.py
```

### Step 7 — Generate document ground truth fixtures

```bash
python ground-truth/document/run_gt_queries.py --generate-fixtures
python ground-truth/document/run_gt_queries.py --verify
```

### Step 8 — Run the benchmark

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Smoke test first (no API calls, verifies infrastructure)
python evaluation/run_benchmark.py --mock --output-dir results/smoke/
python evaluation/compute_metrics.py --results-dir results/smoke/

# Full benchmark (~600 API calls, ~$5–6 USD, ~60–90 minutes)
python evaluation/run_benchmark.py --output-dir results/small/
python evaluation/compute_metrics.py --results-dir results/small/
```

The benchmark runner resumes automatically if interrupted — re-run the same command to continue from where it stopped.

---

## Benchmark Results (Preliminary — Three Stack Exchange Communities)

| System | SC (aggregate) | RL mean (ms) | RL p95 (ms) |
|---|---|---|---|
| **HeteroRAG Full** | **0.856** | 232 | 492 |
| B1 SQL-Only | 0.329 | 156 | 322 |
| B2 Doc-Only RAG | 0.333 | 23 | 46 |
| B3 LLM Function-Calling | 0.571 | 88 | 301 |
| B4 Fixed-Plan Ablation | 0.856 | 222 | 470 |

Source Coverage per query class:

| Class | HeteroRAG | B1 | B2 | B3 | B4 |
|---|---|---|---|---|---|
| 1 — SQL only | **1.000** | 1.000 | 0.000 | 0.100 | 1.000 |
| 2 — Graph only | **0.950** | 0.000 | 0.000 | 0.600 | 0.950 |
| 3 — Doc only | **1.000** | 0.000 | 1.000 | 1.000 | 1.000 |
| 4a — SQL+Graph | **0.700** | 0.467 | 0.000 | 0.367 | 0.700 |
| 4b — SQL+Doc | 0.600 | 0.500 | 0.500 | 0.500 | 0.600 |
| 4c — Graph+Doc | **0.967** | 0.000 | 0.500 | 0.900 | 0.967 |
| 5 — All three | **0.644** | 0.333 | 0.333 | 0.533 | 0.644 |

*Results are preliminary — three Stack Exchange communities, 665,016 indexed documents, 120-question benchmark. Full Stack Overflow evaluation pending.*

---

## Running Tests

```bash
# Unit tests only (no Docker, no API key)
pytest tests/ -m "not integration" -v

# Integration tests (requires Docker services running)
pytest tests/ -m integration -v

# All tests
pytest tests/ -v
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Required for real benchmark runs |
| `PG_HOST` | `localhost` | PostgreSQL host |
| `PG_PORT` | `5432` | PostgreSQL port |
| `PG_DBNAME` | `heterorag` | PostgreSQL database name |
| `PG_USER` | `heterorag` | PostgreSQL user |
| `PG_PASSWORD` | `heterorag_secret` | PostgreSQL password |
| `NEO4J_HOST` | `localhost` | Neo4j host |
| `NEO4J_BOLT_PORT` | `7687` | Neo4j Bolt port |
| `NEO4J_USER` | `neo4j` | Neo4j user |
| `NEO4J_PASSWORD` | `heterorag_secret` | Neo4j password |
| `ES_URL` | `http://localhost:9200` | Elasticsearch URL |
| `ES_INDEX` | `heterorag_content` | Elasticsearch index name |

---

## Citation

If you use this code or benchmark in your research, please cite:

```bibtex
@article{heterorag2025,
  title   = {HeteroRAG: A Federated Heterogeneous Retrieval Framework
             for Natural Language Querying across Microservice Architectures},
  author  = {[Author names]},
  journal = {IEEE Transactions on Knowledge and Data Engineering},
  year    = {2025},
  note    = {Under review}
}
```

---

## Licence

- **Code:** MIT Licence — see `LICENSE`
- **Benchmark questions and ground truth:** CC BY 4.0
- **Stack Exchange data:** CC BY-SA 4.0 — [stackexchange.com/legal](https://stackexchange.com/legal)

---

## Related

- **Paper 2** (companion): Evolutionary Retrieval Planning in HeteroRAG — Genetic Algorithm, Fitness Function, Convergence Analysis
- **Foundation document:** `HeteroRAG_Research_Foundation_v1_6.md` (available in the project knowledge base)
