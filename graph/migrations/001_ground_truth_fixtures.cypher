--liquibase formatted cypher

--changeset heterorag:1 labels:schema comment:constraints-already-applied-via-loader
-- Constraints and indexes are applied by the loader's ensure_neo4j_schema() call.
-- This changeset is a no-op placeholder to allow Liquibase to track schema state.
RETURN 'Schema managed by loader' AS status;

--changeset heterorag:2 labels:ground-truth comment:class2-graph-only-fixture-nodes
-- Ground truth fixture nodes for Class 2 (Graph-only) queries.
-- Each GT node stores the expected result as a JSON property for comparison
-- by the evaluation harness without re-executing Cypher at evaluation time.

-- GT C2 Q01: Find all questions connected to a given question via LINKED_TO within 2 hops
MERGE (gt:GroundTruth {id: 'c2_q01'})
SET   gt.description     = 'Questions linked within 2 hops of question 11227809',
      gt.query_class     = 2,
      gt.services        = ['graph'],
      gt.cypher_template = 'MATCH (q:Question {id: $question_id})-[:LINKED_TO*1..2]->(linked:Question) RETURN DISTINCT linked.id AS id ORDER BY id';

MERGE (gt:GroundTruth {id: 'c2_q02'})
SET   gt.description     = 'Tags that co-occur with python tag',
      gt.query_class     = 2,
      gt.services        = ['graph'],
      gt.cypher_template = 'MATCH (t:Tag {name: "python"})-[r:CO_OCCURS_WITH]-(other:Tag) RETURN other.name AS tag, r.weight AS weight ORDER BY weight DESC LIMIT 20';

MERGE (gt:GroundTruth {id: 'c2_q03'})
SET   gt.description     = 'Users who answered questions tagged with javascript',
      gt.query_class     = 2,
      gt.services        = ['graph'],
      gt.cypher_template = 'MATCH (u:User)-[:ANSWERED]->(a:Answer)-[:ANSWERS]->(q:Question)-[:TAGGED_WITH]->(t:Tag {name: "javascript"}) RETURN DISTINCT u.id AS user_id ORDER BY user_id LIMIT 50';

MERGE (gt:GroundTruth {id: 'c2_q04'})
SET   gt.description     = 'Questions that are duplicates of a given question',
      gt.query_class     = 2,
      gt.services        = ['graph'],
      gt.cypher_template = 'MATCH (q:Question)-[:DUPLICATE_OF]->(original:Question {id: $question_id}) RETURN q.id AS duplicate_id ORDER BY duplicate_id';

MERGE (gt:GroundTruth {id: 'c2_q05'})
SET   gt.description     = 'Tag cluster: tags co-occurring with both python and pandas',
      gt.query_class     = 2,
      gt.services        = ['graph'],
      gt.cypher_template = 'MATCH (t:Tag)-[:CO_OCCURS_WITH]->(py:Tag {name:"python"}), (t)-[:CO_OCCURS_WITH]->(pd:Tag {name:"pandas"}) RETURN t.name AS tag ORDER BY tag LIMIT 20';

MERGE (gt:GroundTruth {id: 'c2_q06'})
SET   gt.description     = 'Users who both asked and answered questions in the same tag',
      gt.query_class     = 2,
      gt.services        = ['graph'],
      gt.cypher_template = 'MATCH (u:User)-[:ASKED]->(q:Question)-[:TAGGED_WITH]->(t:Tag {name:"python"}), (u)-[:ANSWERED]->(a:Answer)-[:ANSWERS]->(q2:Question)-[:TAGGED_WITH]->(t) RETURN DISTINCT u.id AS user_id ORDER BY user_id LIMIT 30';

MERGE (gt:GroundTruth {id: 'c2_q07'})
SET   gt.description     = 'Questions with accepted answers where answerer is not the asker',
      gt.query_class     = 2,
      gt.services        = ['graph'],
      gt.cypher_template = 'MATCH (asker:User)-[:ASKED]->(q:Question)-[:ACCEPTED]->(a:Answer)<-[:ANSWERED]-(answerer:User) WHERE asker.id <> answerer.id RETURN q.id AS question_id, asker.id AS asker_id, answerer.id AS answerer_id ORDER BY question_id LIMIT 50';

MERGE (gt:GroundTruth {id: 'c2_q08'})
SET   gt.description     = 'Tag neighbourhood depth 1 of sql tag',
      gt.query_class     = 2,
      gt.services        = ['graph'],
      gt.cypher_template = 'MATCH (sql:Tag {name:"sql"})-[r:CO_OCCURS_WITH]-(neighbour:Tag) RETURN neighbour.name AS tag, r.weight AS weight ORDER BY weight DESC LIMIT 15';

MERGE (gt:GroundTruth {id: 'c2_q09'})
SET   gt.description     = 'Questions linked to a duplicate chain',
      gt.query_class     = 2,
      gt.services        = ['graph'],
      gt.cypher_template = 'MATCH (q:Question)-[:DUPLICATE_OF*1..3]->(root:Question) WHERE NOT EXISTS { MATCH (root)-[:DUPLICATE_OF]->() } RETURN root.id AS root_id, COUNT(q) AS duplicate_count ORDER BY duplicate_count DESC LIMIT 20';

MERGE (gt:GroundTruth {id: 'c2_q10'})
SET   gt.description     = 'Network diameter: shortest path between two users via answered questions',
      gt.query_class     = 2,
      gt.services        = ['graph'],
      gt.cypher_template = 'MATCH path = shortestPath((u1:User {id: $user1_id})-[*]-(u2:User {id: $user2_id})) RETURN length(path) AS path_length';

-- Algorithm-based GT (Q11-Q20): stored as fixture snapshots after first run
-- The evaluation harness runs these queries once at setup time, stores results
-- as GroundTruth nodes, and uses NDCG@10 for comparison at evaluation time.

MERGE (gt:GroundTruth {id: 'c2_q11'})
SET   gt.description     = 'PageRank top 10 users by influence',
      gt.query_class     = 2,
      gt.services        = ['graph'],
      gt.metric          = 'NDCG@10',
      gt.cypher_template = 'CALL gds.pageRank.stream("userGraph") YIELD nodeId, score RETURN gds.util.asNode(nodeId).id AS user_id, score ORDER BY score DESC LIMIT 10';

MERGE (gt:GroundTruth {id: 'c2_q12'})
SET   gt.description     = 'Community detection: tag communities via Louvain',
      gt.query_class     = 2,
      gt.services        = ['graph'],
      gt.metric          = 'NDCG@10',
      gt.cypher_template = 'CALL gds.louvain.stream("tagGraph") YIELD nodeId, communityId RETURN gds.util.asNode(nodeId).name AS tag, communityId ORDER BY communityId, tag LIMIT 50';

--changeset heterorag:3 labels:ground-truth comment:cross-service-fixture-nodes
-- Fixture nodes for cross-service questions (Classes 4a, 4b, 4c, 5)
-- The SQL component is handled by Flyway views; these nodes record the
-- expected Graph component and the combined service set.

MERGE (gt:GroundTruth {id: 'c4a_q01'})
SET   gt.description     = 'Reputation of users who asked Python questions + tag topology of python',
      gt.query_class     = '4a',
      gt.services        = ['sql', 'graph'],
      gt.sql_view        = 'gt_c4a_q01_sql',
      gt.graph_template  = 'MATCH (t:Tag {name:"python"})-[r:CO_OCCURS_WITH]-(other:Tag) RETURN other.name AS tag, r.weight AS weight ORDER BY weight DESC LIMIT 10';

MERGE (gt:GroundTruth {id: 'c5_q01'})
SET   gt.description     = 'High-reputation 2023 active users + their tag communities + content of their top answers',
      gt.query_class     = 5,
      gt.services        = ['sql', 'graph', 'document'],
      gt.sql_view        = 'gt_c5_q01_sql',
      gt.graph_template  = 'MATCH (u:User)-[:ANSWERED]->(a:Answer)-[:ANSWERS]->(q:Question)-[:TAGGED_WITH]->(t:Tag) WHERE u.id IN $user_ids RETURN t.name AS tag, COUNT(*) AS freq ORDER BY freq DESC LIMIT 10',
      gt.es_query        = '{"query":{"bool":{"must":[{"terms":{"user_id":$user_ids}},{"term":{"doc_type":"answer"}},{"range":{"score":{"gte":10}}}]}},"size":10}';
