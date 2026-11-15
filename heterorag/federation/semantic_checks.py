"""
heterorag/federation/semantic_checks.py
=======================================
Static checks on a translated query that go beyond the catalogue: necessary conditions of translation
correctness (Formal Core draft, section 5) that the catalogue check cannot see. All are deterministic, use no
model and read only the query text, the step and the graph descriptor's relationship triples.

  R1 relationship signature (the registered mechanism M2): a relationship pattern must follow a
     (start label, type, end label) triple of the graph descriptor.
  R2 bind entity: the column (SQL) or property (Cypher) compared with a bound placeholder must belong to the
     entity of the bind.
  R3 neutralised condition: `OR TRUE` / `OR 1=1` makes a condition vacuous.
  R5 join sanity (SQL): the tables of a query must be connected by column-to-column join conditions (no
     cartesian product), and an equality between identifier columns of different entities is a type error.
  R4 positive bind: the bound identifiers may only restrict a result (membership), never be negated
     (NOT IN, <> ALL, NOT EXISTS, must_not): the guarantee propagation treats a bound step as monotone in its
     bound set (Formal Core P3), and set difference is a plan operator, executed in code.

R1 was named before the data (M2 of the F2 record); R2 and R3 were found by reading dev rows (post hoc).
They run only when the validator is built with ``semantic_checks=True`` (environment variable
``HETERORAG_SEMANTIC_CHECKS=1``); the default leaves the validator exactly as at cp1b-freeze. See
``claude/HeteroRAG_Option1_PreReg_v0.md``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .contract import LABEL_OF_ENTITY
from .step_validator import _mask

_NODE = r"\(\s*(?P<v{n}>\w*)\s*(?P<l{n}>(?::\s*\w+\s*)*)[^()]*\)"
_SEGMENT = re.compile(
    r"(?=(?:" + _NODE.format(n=1) + r")\s*(?P<left><)?-\s*\[\s*\w*\s*:\s*(?P<types>[\w|:]+)[^\]]*\]\s*-(?P<right>>)?\s*"
    + _NODE.format(n=2) + r")")


@dataclass(frozen=True)
class Signatures:
    triples: frozenset            # (from_label, edge_type, to_label)

    @classmethod
    def from_descriptor(cls, graph_descriptor) -> "Signatures":
        return cls(frozenset((e.from_label, e.edge_type, e.to_label) for e in graph_descriptor.schema_spec.edge_types))

    @classmethod
    def from_catalogue(cls, catalogue) -> "Signatures":
        return cls(frozenset(catalogue.graph.triples))

    @classmethod
    def poc(cls) -> "Signatures":
        from heterorag.layer1.poc_descriptors import KNOWLEDGE_GRAPH_DESCRIPTOR
        return cls.from_descriptor(KNOWLEDGE_GRAPH_DESCRIPTOR)


def _first_label(group: str) -> str | None:
    labs = re.findall(r":\s*(\w+)", group or "")
    return labs[0] if labs else None


def _var_labels(m: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for var, grp in re.findall(r"\(\s*(\w+)\s*((?::\s*\w+\s*)+)", m):
        out.setdefault(var, _first_label(grp))
    return out


# ------------------------------------------------------------------------------------------------ R1

def relationship_signature_errors(query: str, sigs: Signatures) -> list[str]:
    m = _mask(query, "//").strip().rstrip(";")
    labels = _var_labels(m)
    errs: list[str] = []
    for seg in _SEGMENT.finditer(m):
        a = _first_label(seg.group("l1")) or labels.get(seg.group("v1"))
        b = _first_label(seg.group("l2")) or labels.get(seg.group("v2"))
        if a is None and b is None:
            continue
        left, right = bool(seg.group("left")), bool(seg.group("right"))
        if left and right:
            continue
        types = [t for t in re.split(r"[|:]+", seg.group("types")) if t]
        known = [t for t in types if any(tr[1] == t for tr in sigs.triples)]
        if not known:
            continue                                   # unknown types are the validator's concern
        if left:
            orient = [(b, a)]                          # (start, end)
        elif right:
            orient = [(a, b)]
        else:
            orient = [(a, b), (b, a)]
        ok = any((s is None or s == f) and (e is None or e == t)
                 for typ in known for (f, ty, t) in sigs.triples if ty == typ for (s, e) in orient)
        if not ok:
            have = "; ".join(f"{f}-[:{ty}]->{t}" for (f, ty, t) in sorted(sigs.triples) if ty in known)
            read = (f"{b}<-[:{'|'.join(known)}]-{a}" if left else f"{a}-[:{'|'.join(known)}]->{b}")
            errs.append(f"relationship direction or end labels do not match the store: it has {have}; "
                        f"the query reads {read}")
    return list(dict.fromkeys(errs))


# ------------------------------------------------------------------------------------------------ R2

_SQL_ID_ENTITY_OF_TABLE = {"users": "user", "posts": "post", "tags": "tag"}


def _sql_entity_of(table: str | None, col: str) -> str | None:
    if col == "id":
        return _SQL_ID_ENTITY_OF_TABLE.get(table or "")
    if col == "user_id":
        return "user"
    if col == "post_id":
        return "post"
    if col == "tag_name":
        return "tag"
    return None


_BIND_ENTITY = {"user": "user", "question": "post", "answer": "post", "tag": "tag"}
_SQL_KEYWORDS_AFTER = r"on\b|where\b|join\b|inner\b|left\b|right\b|group\b|order\b|using\b|cross\b|full\b"


def sql_bind_entity_errors(query: str, bind_name: str, entity: str) -> list[str]:
    body = _mask(query, "--").strip().rstrip(";")
    want = _BIND_ENTITY.get(entity)
    if want is None:
        return []
    alias: dict[str, str] = {}
    tables: list[str] = []
    for _kw, name, al in re.findall(rf"\b(from|join)\s+(\w+)(?:\s+(?:as\s+)?(?!{_SQL_KEYWORDS_AFTER})(\w+))?", body, re.I):
        alias[name] = name
        if al:
            alias[al] = name
        tables.append(name)
    ph = re.escape(f"%({bind_name})s")
    errs: list[str] = []
    for ref in re.findall(rf"([\w.]+)\s*=\s*ANY\s*\(\s*{ph}\s*\)|([\w.]+)\s+IN\s*\(?\s*{ph}", body, re.I):
        col = ref[0] or ref[1]
        q, _, c = col.rpartition(".")
        table = alias.get(q) if q else (tables[0] if len(set(tables)) == 1 else None)
        got = _sql_entity_of(table, c)
        if got is not None and got != want:
            errs.append(f"the bound identifiers are {entity} identifiers, but they are compared with {col!r}, "
                        f"which is a {got} identifier")
    return list(dict.fromkeys(errs))


def cypher_bind_entity_errors(query: str, bind_name: str, entity: str) -> list[str]:
    m = _mask(query, "//").strip().rstrip(";")
    want = LABEL_OF_ENTITY.get(entity)
    if want is None:
        return []
    labels = _var_labels(m)
    errs: list[str] = []
    for var, prop in re.findall(rf"\b(\w+)\.(\w+)\s+IN\s+\${re.escape(bind_name)}\b", m, re.I):
        lab = labels.get(var)
        if lab is not None and lab != want:
            errs.append(f"the bound identifiers are {entity} identifiers, but they are compared with "
                        f"{var}.{prop}, and {var} is a {lab} node")
    return list(dict.fromkeys(errs))


# ------------------------------------------------------------------------------------------------ R3

_TAUTOLOGY = re.compile(r"\bor\s+\(?\s*(?:true|1\s*=\s*1)\b|\b(?:true|1\s*=\s*1)\s*\)?\s+or\b", re.I)


def neutralised_condition_errors(query: str, comment: str = "--") -> list[str]:
    body = _mask(query, comment)
    if _TAUTOLOGY.search(body):
        return ["a condition is joined with OR TRUE (or OR 1 = 1), which makes it vacuous; remove it"]
    return []


# ------------------------------------------------------------------------------------------------ R4

_NEGATED_BIND_MESSAGE = ("the bound identifiers are used in a negated position; a bind may only restrict the result "
                         "to the bound set (membership). To remove items use a subtract step of the plan, not a "
                         "negation in the query")


def negated_bind_errors(step, query: str) -> list[str]:
    """The bind placeholder occurs in a negated position (SQL: NOT IN, <> ALL, NOT (x = ANY(..)), NOT EXISTS;
    Cypher: NOT x IN $b; Elasticsearch: inside must_not). No bind, no error."""
    if not getattr(step, "bind", None):
        return []
    name = step.bind.name
    store = getattr(getattr(step, "store", None), "value", str(getattr(step, "store", "")))
    if store == "sql":
        body = _mask(query, "--")
        ph = re.escape(f"%({name})s")
        patterns = (rf"\bnot\s+in\s*\(?\s*{ph}", rf"(?:<>|!=)\s*all\s*\(\s*{ph}",
                    rf"\bnot\s*\(?\s*[\w.]+\s*=\s*any\s*\(\s*{ph}", rf"\bnot\s+exists\b[^;]*{ph}")
        return [_NEGATED_BIND_MESSAGE] if any(re.search(p, body, re.I | re.S) for p in patterns) else []
    if store == "graph":
        body = _mask(query, "//")
        v = re.escape(name)
        patterns = (rf"\bnot\s*\(?\s*\w+(?:\.\w+)?\s+in\s+\${v}\b", rf"\bnot\s+in\s+\${v}\b")
        return [_NEGATED_BIND_MESSAGE] if any(re.search(p, body, re.I) for p in patterns) else []
    if store == "doc":
        try:
            tree = json.loads(query)
        except (TypeError, ValueError):
            return []
        sentinel = f"$bind:{name}"

        def walk(node, negated):
            if isinstance(node, dict):
                return any(walk(v, negated or k == "must_not") for k, v in node.items())
            if isinstance(node, list):
                return any(walk(v, negated) for v in node)
            return negated and node == sentinel

        return [_NEGATED_BIND_MESSAGE] if walk(tree, False) else []
    return []


# ------------------------------------------------------------------------------------------------ R5

_JOIN_SANITY_SKIP = re.compile(r"\bunion\b|\bwith\b|\busing\b|\bnatural\b|\bcross\s+join\b|\(\s*select\b", re.I)
_QUALIFIED_EQ = re.compile(r"\b(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)\b(?!\s*\()")
_BARE_EQ = re.compile(r"(?<![\w.%$'])([A-Za-z_]\w*)\s*=\s*([A-Za-z_]\w*)(?![\w.(])")
_NOT_A_COLUMN = {"true", "false", "null", "any", "all"}


def join_sanity_errors(query: str) -> list[str]:
    """SQL only. (a) With two or more tables, the join conditions col = col must connect them all; a table that only
    appears in conditions on constants or on the bind is a cartesian product. Skipped, to avoid false alarms, when the
    query has a sub-select, UNION, WITH, USING, NATURAL or an unqualified col = col condition. (b) An equality between
    two identifier columns of different entities (posts.id = users.id) compares identifiers of different types."""
    body = _mask(query, "--").strip().rstrip(";")
    # FROM inside a function (EXTRACT(YEAR FROM d), SUBSTRING(x FROM 2), TRIM(BOTH FROM x)) is not a table
    body = re.sub(r"\b(?:extract|substring|trim|overlay|position)\s*\([^()]*\)", " 0 ", body, flags=re.I)
    errs: list[str] = []
    alias: dict[str, str] = {}
    nodes: list[str] = []
    for _kw, name, al in re.findall(rf"\b(from|join)\s+(\w+)(?:\s+(?:as\s+)?(?!{_SQL_KEYWORDS_AFTER})(\w+))?", body, re.I):
        node = al or name
        alias[name] = name
        alias[node] = name
        if node not in nodes:
            nodes.append(node)
    for clause in re.findall(r"\bfrom\s+(.*?)(?=\bwhere\b|\bgroup\b|\border\b|\bhaving\b|\blimit\b|$)", body, re.I | re.S):
        if "(" in clause:
            continue
        for name, al in re.findall(rf",\s*(\w+)(?:\s+(?:as\s+)?(?!{_SQL_KEYWORDS_AFTER})(\w+))?", clause, re.I):
            node = al or name
            alias[name] = name
            alias[node] = name
            if node not in nodes:
                nodes.append(node)
    for a1, c1, a2, c2 in _QUALIFIED_EQ.findall(body):
        t1, t2 = alias.get(a1), alias.get(a2)
        e1, e2 = _sql_entity_of(t1, c1), _sql_entity_of(t2, c2)
        if e1 and e2 and e1 != e2:
            errs.append(f"{a1}.{c1} is a {e1} identifier and {a2}.{c2} is a {e2} identifier; they cannot be equal")
    if len(nodes) >= 2 and not _JOIN_SANITY_SKIP.search(body):
        bare = [(x, y) for x, y in _BARE_EQ.findall(body)
                if x.lower() not in _NOT_A_COLUMN and y.lower() not in _NOT_A_COLUMN]
        if not bare:
            parent = {n: n for n in nodes}

            def find(n):
                while parent[n] != n:
                    parent[n] = parent[parent[n]]
                    n = parent[n]
                return n

            for a1, _c1, a2, _c2 in _QUALIFIED_EQ.findall(body):
                if a1 in parent and a2 in parent:
                    parent[find(a1)] = find(a2)
            groups = {find(n) for n in nodes}
            if len(groups) > 1:
                errs.append(f"the tables {', '.join(nodes)} are not all connected by a join condition between their "
                            "columns (this is a cartesian product, and the bound identifiers do not restrict the "
                            "output); join them on identifier columns that the schema lists, or answer CANNOT_ANSWER if it lists none")
    return list(dict.fromkeys(errs))


# ------------------------------------------------------------------------------------------------ R7

_NON_ADDITIVE_SQL = re.compile(
    r"\bgroup\s+by\b|\bhaving\b|\blimit\b|\boffset\b|\bover\s*\(|\bfetch\s+(?:first|next)\b"
    r"|\b(?:count|sum|avg|min|max|array_agg|string_agg|bool_and|bool_or)\s*\(", re.I)
_NON_ADDITIVE_CYPHER = re.compile(
    r"\blimit\b|\bskip\b|\b(?:count|sum|avg|min|max|collect|percentile\w*)\s*\(", re.I)
_NON_ADDITIVE_DOC_KEYS = {"aggs", "aggregations", "size", "from", "terminate_after", "collapse"}

_R7_MESSAGE = ("a step with a bind must give the same result per bound identifier whatever the other identifiers are: "
               "the executor sends the bound set in batches and unites the results, so an aggregate, GROUP BY, HAVING, "
               "LIMIT, OFFSET or window function over the bound set gives a wrong answer for a large set; "
               "do the counting or limiting in a later step of the plan")


def additive_bind_errors(step, query: str) -> list[str]:
    """R7, from the premise PM (additive binds, Formal Core v1). A SET step with a bind whose query uses an aggregate,
    GROUP BY, HAVING, LIMIT, OFFSET or a window function (SQL; in Cypher an aggregate, LIMIT or SKIP; in the document
    store an aggregation, size or from key). No bind, or a scalar or ranked step, no error."""
    if not getattr(step, "bind", None):
        return []
    kind = getattr(getattr(step, "kind", None), "value", str(getattr(step, "kind", "set")))
    if kind != "set":
        return []
    store = getattr(getattr(step, "store", None), "value", str(getattr(step, "store", "")))
    if store == "sql":
        return [_R7_MESSAGE] if _NON_ADDITIVE_SQL.search(_mask(query, "--")) else []
    if store == "graph":
        return [_R7_MESSAGE] if _NON_ADDITIVE_CYPHER.search(_mask(query, "//")) else []
    if store == "doc":
        try:
            tree = json.loads(query)
        except (TypeError, ValueError):
            return []

        def walk(node):
            if isinstance(node, dict):
                return any(k in _NON_ADDITIVE_DOC_KEYS or walk(v) for k, v in node.items())
            if isinstance(node, list):
                return any(walk(v) for v in node)
            return False

        return [_R7_MESSAGE] if walk(tree) else []
    return []


# ------------------------------------------------------------------------------------------------ R6

_QUOTED = re.compile(r'"([^"\n]{1,80})"|\u201c([^\u201d\n]{1,80})\u201d')
_NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?(?![\w])")


def literal_coverage_errors(question: str, queries) -> list[str]:
    """R6, plan level (not a step check, so not part of CHECKS): every number and every quoted string of the question
    must occur in the final queries of the plan taken together, after case folding and removal of thousands
    separators. Numbers written as words are not covered. In-sample: suggested by two dev cases that lost a literal."""
    blob = " ".join(str(q) for q in (queries.values() if hasattr(queries, "values") else queries))
    low = blob.lower()
    low_nocomma = low.replace(",", "")
    missing: list[str] = []
    for m in _QUOTED.finditer(question or ""):
        lit = (m.group(1) or m.group(2) or "").strip()
        if lit and lit.lower() not in low:
            missing.append(f'"{lit}"')
    stripped = _QUOTED.sub(" ", question or "")
    for m in _NUMBER.finditer(stripped):
        lit = m.group(0).rstrip(".,").replace(",", "")
        if lit and lit not in low_nocomma:
            missing.append(lit)
    return [f"the question contains {', '.join(missing)}, which the queries of the plan do not contain"] if missing else []


# ------------------------------------------------------------------------------------------------ all

CHECKS = ("R1", "R2", "R3", "R4", "R5", "R7")


def semantic_errors(step, query: str, sigs: Signatures) -> list[str]:
    """Every error of the checks R1 to R3 for one translated step query, in a fixed order."""
    found = candidate_errors(step, query, sigs)
    return [e for c in CHECKS for e in found[c]]


def candidate_errors(step, query: str, sigs: Signatures) -> dict[str, list[str]]:
    """The errors of each candidate check for one translated step query (empty lists when it passes)."""
    out: dict[str, list[str]] = {c: [] for c in CHECKS}
    store = getattr(getattr(step, "store", None), "value", str(getattr(step, "store", "")))
    out["R4"] = negated_bind_errors(step, query)
    out["R7"] = additive_bind_errors(step, query)
    if store == "graph":
        out["R1"] = relationship_signature_errors(query, sigs)
        if step.bind:
            out["R2"] = cypher_bind_entity_errors(query, step.bind.name, step.bind.entity)
        out["R3"] = neutralised_condition_errors(query, "//")
    elif store == "sql":
        if step.bind:
            out["R2"] = sql_bind_entity_errors(query, step.bind.name, step.bind.entity)
        out["R3"] = neutralised_condition_errors(query)
        out["R5"] = join_sanity_errors(query)
    return out
