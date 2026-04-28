-- =============================================================================
-- HeteroRAG POC — Flyway Migration V1
-- File: sql/migrations/V1__ground_truth_views.sql
-- Ground truth views for Class 1 (SQL-only) and SQL components of Classes 4a, 4b, 5
-- Naming convention: gt_<class>_q<nn>
-- Each view is the authoritative answer for its benchmark question.
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Class 1 — SQL-only queries (Q01–Q20)
-- Ground truth: exact result sets derivable from SQL tables alone
-- ---------------------------------------------------------------------------

-- Q01: Top 10 users by reputation
CREATE OR REPLACE VIEW gt_c1_q01 AS
SELECT id, display_name, reputation, location
FROM   users
ORDER  BY reputation DESC
LIMIT  10;

-- Q02: Users with more than 1000 reputation created before 2015
CREATE OR REPLACE VIEW gt_c1_q02 AS
SELECT id, display_name, reputation, creation_date::DATE AS joined
FROM   users
WHERE  reputation > 1000
  AND  creation_date < '2015-01-01'
ORDER  BY reputation DESC;

-- Q03: Posts with highest score that are questions (post_type_id = 1)
CREATE OR REPLACE VIEW gt_c1_q03 AS
SELECT id, title, score, view_count, answer_count, tags
FROM   posts
WHERE  post_type_id = 1
ORDER  BY score DESC
LIMIT  20;

-- Q04: Count of questions per tag (top 20 tags by post volume)
CREATE OR REPLACE VIEW gt_c1_q04 AS
SELECT tag_name, count
FROM   tags
ORDER  BY count DESC
LIMIT  20;

-- Q05: Questions tagged with 'python' ordered by score
CREATE OR REPLACE VIEW gt_c1_q05 AS
SELECT id, title, score, view_count, creation_date::DATE AS created
FROM   posts
WHERE  post_type_id = 1
  AND  tags ILIKE '%<python>%'
ORDER  BY score DESC
LIMIT  20;

-- Q06: Users who have cast more than 500 up-votes
CREATE OR REPLACE VIEW gt_c1_q06 AS
SELECT id, display_name, up_votes, down_votes, reputation
FROM   users
WHERE  up_votes > 500
ORDER  BY up_votes DESC;

-- Q07: Questions with an accepted answer and score > 50
CREATE OR REPLACE VIEW gt_c1_q07 AS
SELECT id, title, score, accepted_answer_id, answer_count
FROM   posts
WHERE  post_type_id = 1
  AND  accepted_answer_id IS NOT NULL
  AND  score > 50
ORDER  BY score DESC;

-- Q08: Badge count per user (top 20 most badged users)
CREATE OR REPLACE VIEW gt_c1_q08 AS
SELECT u.id, u.display_name, COUNT(b.id) AS badge_count
FROM   users u
JOIN   badges b ON b.user_id = u.id
GROUP  BY u.id, u.display_name
ORDER  BY badge_count DESC
LIMIT  20;

-- Q09: Gold badge holders
CREATE OR REPLACE VIEW gt_c1_q09 AS
SELECT DISTINCT u.id, u.display_name, u.reputation
FROM   users u
JOIN   badges b ON b.user_id = u.id
WHERE  b.class = 1  -- Gold
ORDER  BY u.reputation DESC;

-- Q10: Questions with more than 10 answers
CREATE OR REPLACE VIEW gt_c1_q10 AS
SELECT id, title, score, answer_count, view_count
FROM   posts
WHERE  post_type_id = 1
  AND  answer_count > 10
ORDER  BY answer_count DESC;

-- Q11: Monthly question volume in 2022
CREATE OR REPLACE VIEW gt_c1_q11 AS
SELECT DATE_TRUNC('month', creation_date)::DATE AS month,
       COUNT(*) AS question_count
FROM   posts
WHERE  post_type_id = 1
  AND  creation_date >= '2022-01-01'
  AND  creation_date <  '2023-01-01'
GROUP  BY 1
ORDER  BY 1;

-- Q12: Users with both up-votes > 200 and down-votes > 50
CREATE OR REPLACE VIEW gt_c1_q12 AS
SELECT id, display_name, up_votes, down_votes, reputation
FROM   users
WHERE  up_votes > 200
  AND  down_votes > 50
ORDER  BY reputation DESC;

-- Q13: Posts with the most comments
CREATE OR REPLACE VIEW gt_c1_q13 AS
SELECT p.id, p.title, p.post_type_id, p.comment_count
FROM   posts p
ORDER  BY comment_count DESC
LIMIT  20;

-- Q14: View count distribution — questions with > 10000 views
CREATE OR REPLACE VIEW gt_c1_q14 AS
SELECT id, title, view_count, score, tags
FROM   posts
WHERE  post_type_id = 1
  AND  view_count > 10000
ORDER  BY view_count DESC;

-- Q15: Users from a specific location (London)
CREATE OR REPLACE VIEW gt_c1_q15 AS
SELECT id, display_name, reputation, location
FROM   users
WHERE  location ILIKE '%london%'
ORDER  BY reputation DESC;

-- Q16: Tag-based badge holders (tag-based = TRUE)
CREATE OR REPLACE VIEW gt_c1_q16 AS
SELECT b.name AS badge_name, COUNT(DISTINCT b.user_id) AS holder_count
FROM   badges b
WHERE  b.tag_based = TRUE
GROUP  BY b.name
ORDER  BY holder_count DESC
LIMIT  20;

-- Q17: Score distribution of answers for a specific question
--       (parameterised via question_id = 11227809 — well-known SO question)
CREATE OR REPLACE VIEW gt_c1_q17 AS
SELECT p.id AS answer_id, p.score, p.creation_date::DATE AS created
FROM   posts p
WHERE  p.post_type_id = 2
  AND  p.parent_id = 11227809
ORDER  BY p.score DESC;

-- Q18: Users with zero reputation (brand-new accounts)
CREATE OR REPLACE VIEW gt_c1_q18 AS
SELECT id, display_name, creation_date::DATE AS joined
FROM   users
WHERE  reputation = 0
ORDER  BY creation_date DESC
LIMIT  50;

-- Q19: Posts closed as duplicates (closed_date IS NOT NULL and duplicate link exists)
CREATE OR REPLACE VIEW gt_c1_q19 AS
SELECT DISTINCT p.id, p.title, p.score
FROM   posts p
JOIN   post_links pl ON pl.post_id = p.id AND pl.link_type_id = 3
WHERE  p.closed_date IS NOT NULL
ORDER  BY p.score DESC;

-- Q20: Average score per tag (top 15 highest-quality tags)
CREATE OR REPLACE VIEW gt_c1_q20 AS
SELECT t.tag_name,
       AVG(p.score)::NUMERIC(10,2) AS avg_score,
       COUNT(p.id) AS question_count
FROM   tags t
JOIN   posts p ON p.tags ILIKE '%<' || t.tag_name || '>%'
              AND p.post_type_id = 1
GROUP  BY t.tag_name
HAVING COUNT(p.id) >= 50          -- meaningful sample size
ORDER  BY avg_score DESC
LIMIT  15;

-- =============================================================================
-- SQL components of cross-service questions (Classes 4a, 4b, 5)
-- These return the SQL portion of the answer; combined GT is at the
-- evaluation layer by joining with Graph and Document components.
-- Naming: gt_c<class>_q<nn>_sql
-- =============================================================================

-- Class 4a Q01 SQL component: reputation of users who asked Python questions
CREATE OR REPLACE VIEW gt_c4a_q01_sql AS
SELECT DISTINCT p.owner_user_id AS user_id,
       u.reputation,
       u.display_name
FROM   posts p
JOIN   users u ON u.id = p.owner_user_id
WHERE  p.post_type_id = 1
  AND  p.tags ILIKE '%<python>%'
ORDER  BY u.reputation DESC
LIMIT  20;

-- Class 4b Q01 SQL component: score and view_count for questions about list comprehension
CREATE OR REPLACE VIEW gt_c4b_q01_sql AS
SELECT id, score, view_count, answer_count
FROM   posts
WHERE  post_type_id = 1
  AND  tags ILIKE '%<python>%'
ORDER  BY score DESC
LIMIT  20;

-- Class 5 Q01 SQL component: high-reputation users active in 2023
CREATE OR REPLACE VIEW gt_c5_q01_sql AS
SELECT id AS user_id, display_name, reputation
FROM   users
WHERE  reputation > 5000
  AND  EXISTS (
       SELECT 1 FROM posts
       WHERE  owner_user_id = users.id
         AND  creation_date >= '2023-01-01'
  )
ORDER  BY reputation DESC
LIMIT  20;

COMMENT ON VIEW gt_c1_q01 IS 'GT Class 1 Q01: Top 10 users by reputation';
COMMENT ON VIEW gt_c1_q20 IS 'GT Class 1 Q20: Average score per tag (top 15)';
