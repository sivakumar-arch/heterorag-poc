"""
heterorag/federation/step_validator.py
======================================
Static checks on a translated native query, before it is executed
(Interface Specification v2.1, section 5).

The checks are conservative: a query that passes may still fail at the store (an unqualified
column that does not exist, for example), and that is reported as a failed step with the
store's error. A query that fails here is returned to the translator once with the messages.

Checks, by store:

* all stores: read-only; bind placeholders are exactly the declared bind; no fetch limit
  unless the step is ranked, where the limit must equal k;
* SQL: one SELECT statement; tables and qualified columns belong to the store; the first
  selected item is the entity identifier; no stray ``%`` (the client uses a parameter style
  that gives ``%`` a meaning);
* Cypher: no write clause and no procedure call; labels, relationship types and properties
  belong to the store; the first returned item is the entity identifier;
* Elasticsearch: a JSON object with exactly the key ``query``; no scripts; fields belong to
  the store; a ``doc_type`` constraint that matches the step's output entity.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .contract import Catalogue, DOC_TYPE_OF_ENTITY
from .guarantee import Kind
from .plan import Mode, Step, Store

_SQL_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|copy|call|do|execute|vacuum|"
    r"analyze|set|reset|listen|notify|lock|merge|into|pg_sleep|pg_read_file|lo_import|dblink|"
    r"pg_terminate_backend|pg_cancel_backend)\b", re.I)
_CYPHER_FORBIDDEN = re.compile(
    r"\b(create|merge|delete|detach|set|remove|drop|foreach|load|call|apoc|dbms|yield|skip)\b", re.I)
_ES_FORBIDDEN_KEYS = {"script", "script_score", "function_score", "runtime_mappings",
                      "scripted_metric", "script_fields", "pit", "search_after", "from", "size", "sort"}
# Column names that carry an entity identifier in the SQL store (users.id, badges.user_id, comments.post_id).
_SQL_ID_NAMES = {"user": ("id", "user_id"), "question": ("id", "post_id"), "answer": ("id", "post_id"),
                 "tag": ("tag_name",)}
_ES_LEAF = {"term", "terms", "match", "match_phrase", "match_phrase_prefix", "range", "prefix",
            "wildcard", "regexp", "fuzzy"}


# ----------------------------------------------------------------------------- masking

def _mask(text: str, line_comment: str) -> str:
    """Remove comments and replace the content of string literals by ``x``.
    Double-quoted SQL identifiers keep their name (they are not literals)."""
    out: list[str] = []
    i, n = 0, len(text)
    sql = line_comment == "--"
    while i < n:
        c = text[i]
        if text.startswith(line_comment, i):
            j = text.find("\n", i)
            i = n if j < 0 else j
            continue
        if text.startswith("/*", i):
            j = text.find("*/", i + 2)
            i = n if j < 0 else j + 2
            out.append(" ")
            continue
        if c == "'" or (c == '"' and not sql):
            q = c
            i += 1
            while i < n:
                if text[i] == "\\" and not sql:
                    i += 2
                    continue
                if text[i] == q:
                    if sql and i + 1 < n and text[i + 1] == q:      # doubled quote
                        i += 2
                        continue
                    break
                i += 1
            i += 1
            out.append("'x'")
            continue
        if c == '"' and sql:
            j = text.find('"', i + 1)
            j = n if j < 0 else j
            out.append(text[i + 1:j])
            i = j + 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def _top_level_first_item(s: str) -> str:
    """The first comma-separated item of a select/return list, up to a top-level keyword."""
    depth, buf = 0, []
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            break
        buf.append(ch)
    return "".join(buf).strip()


def _limit_errors(masked: str, step: Step) -> list[str]:
    limits = re.findall(r"\blimit\s+(\S+)", masked, re.I)
    if step.kind is Kind.RANKED:
        errs = []
        if not re.search(r"\border\s+by\b", masked, re.I):
            errs.append("a ranked step needs ORDER BY")
        if len(limits) != 1 or limits[0] != str(step.k):
            errs.append(f"a ranked step needs exactly one LIMIT {step.k}")
        return errs
    if limits:
        return ["remove LIMIT: the executor reads results to the end; fetch limits are not written in queries"]
    return []


# ----------------------------------------------------------------------------- SQL

def check_sql(step: Step, query: str, cat: Catalogue) -> list[str]:
    errs: list[str] = []
    if not query.strip():
        return ["the query is empty"]
    m = _mask(query, "--").strip()
    body = m.rstrip().rstrip(";").strip()
    if ";" in body:
        errs.append("only one SQL statement is allowed")
    first = re.match(r"\(*\s*(\w+)", body)
    if not first or first.group(1).lower() not in ("select", "with"):
        errs.append("the query must be a SELECT statement")
    bad = {w.lower() for w in _SQL_FORBIDDEN.findall(body)}
    if bad:
        errs.append(f"not allowed in a read-only query: {sorted(bad)}")

    # bind placeholders; any other percent sign is rejected
    params = set(re.findall(r"%\((\w+)\)s", body))
    declared = {step.bind.name} if step.bind else set()
    if params != declared:
        errs.append(f"bind placeholders must be exactly {sorted(declared) or 'none'}, found {sorted(params) or 'none'}")
    # checked on the raw text: the driver reads percent signs inside literals and comments too
    if re.search(r"%(?!\(\w+\)s)", query):
        errs.append("do not use the percent sign except in bind placeholders; "
                    "use strpos(lower(col), 'text') > 0 or starts_with() for substring tests")

    errs += _limit_errors(body, step)

    # tables and qualified columns
    scrub = re.sub(r"\b(extract|substring|trim|overlay|position)\s*\((?:[^()]|\([^()]*\))*\)", "X", body, flags=re.I)
    ctes = set(re.findall(r"\b(\w+)\s+as\s*\(", scrub, re.I))
    alias: dict[str, str] = {}
    for kw, name, al in re.findall(r"\b(from|join)\s+(\w+)(?:\s+(?:as\s+)?(?!on\b|where\b|join\b|inner\b|left\b|right\b|group\b|order\b|using\b|cross\b|full\b)(\w+))?",
                                   scrub, re.I):
        if name in ctes:
            continue
        if name not in cat.sql.tables:
            errs.append(f"unknown table {name!r}; this store has {sorted(cat.sql.tables)}")
            continue
        alias[name] = name
        if al:
            alias[al] = name
    for a, col in re.findall(r"\b(\w+)\.(\w+)\b", scrub):
        t = alias.get(a)
        if t is not None and col not in cat.sql.tables[t] and col != "*":
            errs.append(f"table {t!r} has no column {col!r}")

    # identifier first
    sel = re.match(r"\s*(?:with\b.*?\)\s*)?select\s+(?:distinct\s+)?(.*?)\s+from\b", scrub, re.I | re.S)
    if sel and step.kind is not Kind.SCALAR:
        item = _top_level_first_item(sel.group(1))
        aliased = re.search(r"\s+as\s+(\w+)$", item, flags=re.I)
        item = re.sub(r"\s+as\s+\w+$", "", item, flags=re.I).strip()
        ok_names = _SQL_ID_NAMES.get(step.output, ("id",))
        names = "|".join(ok_names)
        if not (re.search(rf"(^|\.)({names})$", item) or (aliased and aliased.group(1) in ok_names)):
            errs.append(f"the first selected item must be the {step.output} identifier "
                        f"(one of {', '.join(ok_names)}), found {item!r}")
    return errs


# ----------------------------------------------------------------------------- Cypher

def check_cypher(step: Step, query: str, cat: Catalogue) -> list[str]:
    errs: list[str] = []
    if not query.strip():
        return ["the query is empty"]
    m = _mask(query, "//").strip().rstrip(";")
    if ";" in m:
        errs.append("only one Cypher statement is allowed")
    bad = {w.lower() for w in _CYPHER_FORBIDDEN.findall(m)}
    if bad:
        errs.append(f"not allowed in a read-only query: {sorted(bad)}")
    if not re.search(r"\breturn\b", m, re.I):
        errs.append("the query must contain RETURN")

    params = set(re.findall(r"\$(\w+)", m))
    declared = {step.bind.name} if step.bind else set()
    if params != declared:
        errs.append(f"parameters must be exactly {sorted(declared) or 'none'}, found {sorted(params) or 'none'}")

    errs += _limit_errors(m, step)

    for grp in re.findall(r"\(\s*\w*\s*((?::\s*\w+\s*)+)", m):
        for lab in re.findall(r":\s*(\w+)", grp):
            if lab not in cat.graph.labels:
                errs.append(f"unknown node label {lab!r}; this store has {sorted(cat.graph.labels)}")
    for grp in re.findall(r"\[\s*\w*\s*:\s*([\w|:]+)", m):
        for rel in re.split(r"[|:]+", grp):
            if rel and rel not in cat.graph.relationships:
                errs.append(f"unknown relationship type {rel!r}; this store has {sorted(cat.graph.relationships)}")
    for prop in re.findall(r"\b[A-Za-z_]\w*\.([A-Za-z_]\w*)", m):
        if prop not in cat.graph.properties:
            errs.append(f"unknown property {prop!r}; this store has {sorted(cat.graph.properties)}")
    for prop in re.findall(r"\{\s*(\w+)\s*:", m):
        if prop not in cat.graph.properties:
            errs.append(f"unknown property {prop!r} in a pattern")

    # every variable used in an expression must be bound in a pattern or by AS / WITH
    defined = set(re.findall(r"\(\s*([A-Za-z_]\w*)\s*(?=[:){])", m))                 # node variables
    defined |= set(re.findall(r"\[\s*([A-Za-z_]\w*)\s*(?=[:\]{*|])", m))             # relationship variables
    defined |= set(re.findall(r"\bas\s+([A-Za-z_]\w*)", m, re.I))                      # aliases
    defined |= set(re.findall(r"\b([A-Za-z_]\w*)\s*=\s*(?=\()", m))                    # path variables
    for var in sorted(set(re.findall(r"\b([A-Za-z_]\w*)\.[A-Za-z_]\w*", m)) - defined):
        errs.append(f"variable {var!r} is used but not defined; bind it in MATCH, for example (q:Question), "
                    f"and use q.id (a label is not a variable)")

    ret = re.search(r"\breturn\s+(?:distinct\s+)?(.*?)(?:\border\s+by\b|\blimit\b|$)", m, re.I | re.S)
    if ret and step.kind is not Kind.SCALAR:
        item = _top_level_first_item(ret.group(1))
        item = re.sub(r"\s+as\s+\w+$", "", item, flags=re.I).strip()
        want = "name" if step.output == "tag" else "id"
        if not re.search(rf"\.{want}$", item):
            errs.append(f"the first returned item must be the {step.output} identifier (.{want}), found {item!r}")
    return errs


# ----------------------------------------------------------------------------- Elasticsearch

def _walk(node: Any, fn) -> None:
    if isinstance(node, dict):
        fn(node)
        for v in node.values():
            _walk(v, fn)
    elif isinstance(node, list):
        for v in node:
            _walk(v, fn)


def check_doc(step: Step, query: str, cat: Catalogue) -> list[str]:
    errs: list[str] = []
    try:
        body = json.loads(query)
    except ValueError as ex:
        return [f"the query is not valid JSON: {ex}"]
    if not isinstance(body, dict) or set(body) != {"query"}:
        return ["the query must be a JSON object with exactly one key, 'query'"]

    fields_seen: set[str] = set()
    sentinels: set[str] = set()
    doc_types: set[str] = set()
    forbidden: set[str] = set()

    def visit(d: dict) -> None:
        for k, v in d.items():
            if k in _ES_FORBIDDEN_KEYS:
                forbidden.add(k)
            if k in _ES_LEAF and isinstance(v, dict):
                for f, val in v.items():
                    if f in ("boost", "_name"):
                        continue
                    fields_seen.add(f)
                    if f == "doc_type" and k in ("term", "terms"):
                        vals = val.get("value") if isinstance(val, dict) else val
                        doc_types.update(vals if isinstance(vals, list) else [vals])
            if k == "exists" and isinstance(v, dict) and "field" in v:
                fields_seen.add(v["field"])
            if k in ("multi_match", "query_string", "simple_query_string") and isinstance(v, dict):
                for f in v.get("fields", []):
                    fields_seen.add(str(f).split("^")[0])
                if k != "multi_match" and step.mode is Mode.FILTER:
                    forbidden.add(k)
        for v in d.values():
            if isinstance(v, str) and v.startswith("$bind:"):
                sentinels.add(v[len("$bind:"):])

    _walk(body["query"], visit)
    # sentinels may sit inside lists
    def lists(node):
        if isinstance(node, list):
            for x in node:
                if isinstance(x, str) and x.startswith("$bind:"):
                    sentinels.add(x[len("$bind:"):])
        if isinstance(node, dict):
            for x in node.values():
                lists(x)
        elif isinstance(node, list):
            for x in node:
                lists(x)
    lists(body["query"])

    if forbidden:
        errs.append(f"not allowed here: {sorted(forbidden)} (no scripts, paging or sorting keys; "
                    "no query_string in filter mode)")
    unknown = fields_seen - cat.doc.fields
    if unknown:
        errs.append(f"unknown fields {sorted(unknown)}; this store has {sorted(cat.doc.fields)}")
    declared = {step.bind.name} if step.bind else set()
    if sentinels != declared:
        errs.append(f"bind sentinels must be exactly {sorted(declared) or 'none'}, found {sorted(sentinels) or 'none'}")
    expected = DOC_TYPE_OF_ENTITY.get(step.output)
    if expected is None:
        errs.append(f"the document store has no identifier for entity type {step.output!r}")
    elif expected not in doc_types:
        errs.append(f"the query must restrict doc_type to {expected!r} with a term or terms clause")
    elif doc_types - {expected}:
        errs.append(f"doc_type may only be {expected!r} for output {step.output!r}, found {sorted(doc_types)}")
    return errs


# ----------------------------------------------------------------------------- dispatcher

class StepValidator:
    def __init__(self, catalogue: Catalogue):
        self.cat = catalogue

    def check(self, step: Step, query: str) -> list[str]:
        if step.store is Store.SQL:
            return check_sql(step, query, self.cat)
        if step.store is Store.GRAPH:
            return check_cypher(step, query, self.cat)
        if step.store is Store.DOC:
            return check_doc(step, query, self.cat)
        return [f"step {step.id} has no store"]
