"""
benchmark_v2/natural_error_checks.py
====================================
Candidate static checks for translated queries, written after reading the silent errors that weak local
models produced on the CP1b questions (capability-floor runs, 2026-11-07). NOT part of the validator and not
used by any run: the module only adds functions that `replay_checks` applies to queries that were already
recorded, so that false rejections and catches can be counted without a model call.

Three checks, each aimed at one class seen in the rows:

  R1 relationship signature (this is the registered mechanism M2): a relationship pattern must follow a
     (start label, type, end label) triple of the graph descriptor. Seen: `(t:Tag)-[:TAGGED_WITH]->(:Question)`
     and `(q:Question)-[:ASKED]->(u)`, both reversed, which return the empty set.
  R2 bind entity: the column (SQL) or property (Cypher) that is compared with a bound placeholder must belong
     to the entity of the bind. Seen: `users.id = ANY(%(s1)s)` where s1 holds question identifiers.
  R3 neutralised condition: `OR TRUE` / `OR 1=1` makes a condition vacuous. Seen: `reputation >= 3648 AND
     (users.id = ANY(%(s1)s) OR TRUE)`, which keeps the placeholder the validator demands and ignores it.

R2 and R3 were identified after the rows were read (post hoc); R1 was named before (M2 of the F2 record).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from heterorag.federation.contract import LABEL_OF_ENTITY
from heterorag.federation.step_validator import _mask

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


# ------------------------------------------------------------------------------------------------ all

CHECKS = ("R1", "R2", "R3")


def candidate_errors(step, query: str, sigs: Signatures) -> dict[str, list[str]]:
    """The errors of each candidate check for one translated step query (empty lists when it passes)."""
    out: dict[str, list[str]] = {c: [] for c in CHECKS}
    store = getattr(getattr(step, "store", None), "value", str(getattr(step, "store", "")))
    if store == "graph":
        out["R1"] = relationship_signature_errors(query, sigs)
        if step.bind:
            out["R2"] = cypher_bind_entity_errors(query, step.bind.name, step.bind.entity)
        out["R3"] = neutralised_condition_errors(query, "//")
    elif store == "sql":
        if step.bind:
            out["R2"] = sql_bind_entity_errors(query, step.bind.name, step.bind.entity)
        out["R3"] = neutralised_condition_errors(query)
    return out
