-- V2: make the tag-based ground-truth views work on current Stack Exchange dumps.
--
-- Dumps from late 2023 onward store tags as "|python|pandas|" (not "<python><pandas>").
-- The V1 views matched '%<python>%' and therefore returned no rows on such data.
-- CREATE OR REPLACE keeps the column lists unchanged. Matching is exact per tag
-- (delimiters included), so "python" does not match "python-3.x" or "ipython".
-- Secondary sort keys are added so LIMIT results are deterministic on score ties.

-- Q05: Questions tagged with 'python' ordered by score
CREATE OR REPLACE VIEW gt_c1_q05 AS
SELECT id, title, score, view_count, creation_date::DATE AS created
FROM   posts
WHERE  post_type_id = 1
  AND  tags LIKE '%|python|%'
ORDER  BY score DESC, id
LIMIT  20;

-- Q20: Average score per tag (top 15 highest-quality tags with at least 50 questions)
CREATE OR REPLACE VIEW gt_c1_q20 AS
SELECT t.tag_name,
       AVG(x.score)::NUMERIC(10,2) AS avg_score,
       COUNT(x.id) AS question_count
FROM (
        SELECT p.id, p.score,
               unnest(string_to_array(trim(both '|' from p.tags), '|')) AS tag_name
        FROM   posts p
        WHERE  p.post_type_id = 1
          AND  p.tags IS NOT NULL
          AND  p.tags <> ''
     ) x
JOIN   tags t ON t.tag_name = x.tag_name
GROUP  BY t.tag_name
HAVING COUNT(x.id) >= 50
ORDER  BY avg_score DESC, t.tag_name
LIMIT  15;

-- Class 4a Q01 SQL component: reputation of users who asked Python questions
CREATE OR REPLACE VIEW gt_c4a_q01_sql AS
SELECT DISTINCT p.owner_user_id AS user_id,
       u.reputation,
       u.display_name
FROM   posts p
JOIN   users u ON u.id = p.owner_user_id
WHERE  p.post_type_id = 1
  AND  p.tags LIKE '%|python|%'
ORDER  BY u.reputation DESC, user_id
LIMIT  20;

-- Class 4b Q01 SQL component: score and view_count for Python questions
CREATE OR REPLACE VIEW gt_c4b_q01_sql AS
SELECT id, score, view_count, answer_count
FROM   posts
WHERE  post_type_id = 1
  AND  tags LIKE '%|python|%'
ORDER  BY score DESC, id
LIMIT  20;
