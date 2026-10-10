# Data-ownership refactor (Benchmark v2 spec D2, D10, D11)

## Rule

Each fact lives in exactly one store. Identifiers (post id, user id, tag name) are the only shared keys.

| Store | Holds | Does not hold |
|---|---|---|
| PostgreSQL | attributes: user profile and counters, post metrics and dates, votes, badges, tag usage counts, comment metadata | relationships (author, parent, accepted answer, post tags, post links), any text |
| Neo4j | relationships: ASKED, ANSWERED, ANSWERS, ACCEPTED, TAGGED_WITH, CO_OCCURS_WITH, LINKED_TO, DUPLICATE_OF; nodes carry identifiers only (Tag: name) | every attribute, every text, titles |
| Elasticsearch | text: question titles and bodies, answer bodies, comments, about-me, tag wikis | scores, dates, authorship, parents, tags |

Event records (votes, badges, comments) keep the post and user identifiers they refer to (D10). Entity type (question, answer, comment, and so on) is part of an entity's identity in every store.

## What changed

- `sql/init/00_bootstrap.sql`: `posts` loses owner, last editor, parent, accepted answer, tags and title; `users` loses about_me; `tags` loses excerpt and wiki post references; `post_links` is removed.
- `scripts/load_stackoverflow_dump.py`: Postgres, Neo4j and Elasticsearch loaders write only what the store owns. Stale-store guards refuse to load into a store created before this change.
- `heterorag/es_index.py` (new): the single definition of the Elasticsearch index. `ground-truth/document/setup_index.py` and the loader both use it. Previously each had its own mapping and analyzer, so the analysis chain depended on which ran first.
- `heterorag/layer1/poc_descriptors.py`: descriptors (schema version 2.0.0) describe the ownership boundary in the text the translation LLM and the discovery filter see.
- `heterorag/layer3/result_normaliser.py`: graph records yield identifier entities from any `*_id` key (not `*_type_id`), so aliases such as `question_id` are recognised.
- Removed: Flyway views (`sql/migrations/V1`, `V2`), the Flyway service in `docker-compose.yml`, the Liquibase ground-truth fixtures (`graph/migrations/001_ground_truth_fixtures.cypher`).
- `benchmark_v2/fidelity.py` (new): checks the live stores against the oracle (counts, value totals, ownership, sampled records).

## Consequences to keep in mind

- The legacy 25-question smoke benchmark (`c1_*` to `c5_*`) scored against the removed views and fixtures. It still runs for smoke purposes but its SQL ground truth no longer exists; it is superseded by Benchmark v2.
- The virtual union is the set of facts loaded into the stores. Fields not loaded anywhere (last editor, favourite count, owner display name, community-owned date, post history) are outside it and must not appear in questions.
- Questions that used to be answerable from one store (for example "views of questions tagged X") now need the graph and SQL together. That is intended.
- The dump's `answer_count` is a number stored in SQL; the answers themselves are in the graph. A question that needs the true number of answers must say which source defines it; the oracle can compute both.
