"""
heterorag/federation/translator.py
==================================
Translates one retrieve step into one native query (Interface Specification v2.1, section 4).

The prompt carries the step's sub-question, the store's schema, the identifier contract for
the step's output entity and, for a bound step, the name of the bind placeholder. Values that
come from the question are written into the query as literals. Identifiers produced by other
steps are never shown to the model; they arrive through the bind (principle D16).

A query that fails the static checks is returned to the model once with the messages.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .contract import ID_CONTRACT, Catalogue
from .guarantee import Kind
from .llm_config import MeteredProvider, Usage
from .plan import Mode, Step, Store
from .step_validator import StepValidator


@dataclass
class TranslationResult:
    step_id: str
    query: str = ""
    errors: list[str] = field(default_factory=list)      # empty means the query passed the checks
    attempts: int = 0
    usage: Usage = field(default_factory=Usage)
    raw: list[str] = field(default_factory=list)
    store_rejections: int = 0                            # attempts the store's own check rejected
    notes: list[str] = field(default_factory=list)       # for example: the store check was unavailable

    @property
    def ok(self) -> bool:
        return bool(self.query) and not self.errors


# ----------------------------------------------------------------------------- schema text

def _sql_schema(desc) -> str:
    blocks = []
    for t in desc.schema_spec.tables:
        cols = []
        for c in t.columns:
            name = f'"{c.name}"' if c.name in ("class", "date", "count") else c.name
            note = f"  -- {c.description}" if c.description else ""
            cols.append(f"  {name} {c.data_type}{note}")
        blocks.append(f"TABLE {t.table_name}  -- {t.description or ''}\n" + "\n".join(cols))
    return "\n\n".join(blocks)


def _graph_schema(desc) -> str:
    s = desc.schema_spec
    nodes = [f"  (:{n.label} {{{', '.join(n.properties)}}})  // {n.description or ''}" for n in s.node_types]
    edges = [f"  (:{e.from_label})-[:{e.edge_type}{' {' + ', '.join(e.properties) + '}' if e.properties else ''}]->(:{e.to_label})"
             f"  // {e.description or ''}" for e in s.edge_types]
    return "NODES:\n" + "\n".join(nodes) + "\n\nRELATIONSHIPS:\n" + "\n".join(edges)


def _doc_schema(desc) -> str:
    out = []
    for t in desc.schema_spec.doc_types:
        fields = ", ".join(f"{f.field_name} ({f.field_type})" for f in t.fields)
        out.append(f"doc_type '{t.doc_type}': fields {fields}\n  {t.description or ''}")
    return "\n".join(out)


# ----------------------------------------------------------------------------- prompts

_COMMON_BIND = """BOUND IDENTIFIERS: this step is restricted to identifiers produced by an earlier step. They are
passed to the query as a list parameter named {name}; you never see them. {how}"""


def _output_block(step: Step) -> str:
    ident = ID_CONTRACT[step.store].get(step.output, "(no identifier defined)")
    if step.kind is Kind.SCALAR:
        return "OUTPUT: one row with one column holding the number the question asks for."
    ranked = ""
    if step.kind is Kind.RANKED:
        ranked = (f"\nThe result is ranked: return the {step.output} identifier first and the ranking value second, "
                  f"order by the value descending and then by the identifier ascending, and return exactly the "
                  f"top {step.k}.")
    return (f"OUTPUT: the first returned column or field is the identifier of each {step.output}: {ident}. "
            f"Return one row per distinct {step.output}.{ranked}")


def build_prompt(step: Step, desc) -> str:
    if step.store is Store.SQL:
        how = (f"Use it as `<identifier column> = ANY(%({step.bind.name})s)`." if step.bind else "")
        rules = f"""RULES
1. Write one PostgreSQL SELECT statement and nothing else.
2. Use only the tables and columns above.
3. Values that appear in the question (numbers, names, words, years) are written into the query as literals.
4. Do not write LIMIT{', except the ranking limit stated in OUTPUT' if step.kind is Kind.RANKED else ''}. Do not write OFFSET.
5. Do not use the percent sign anywhere except in the bind placeholder. For substring tests use strpos(lower(column), 'text') > 0.
6. Quote the column names class, date and count with double quotes.
7. For a year use EXTRACT(YEAR FROM creation_date)."""
        schema = _sql_schema(desc)
        lang = "SQL"
    elif step.store is Store.GRAPH:
        how = ""
        owned = ID_CONTRACT[Store.GRAPH].get(step.bind.entity) if step.bind else None
        if step.bind and not owned:
            how = f"Use it as `WHERE <variable>.<id property> IN ${step.bind.name}`."
        elif step.bind:
            label, _, prop = owned.partition(".")
            how = (f"Give the {label} node a variable and test its {prop} in a WHERE clause, for example "
                   f"`MATCH (v:{label}) ... WHERE v.{prop} IN ${step.bind.name}`. The parameter is used only in "
                   f"such a WHERE test, never inside a property map and never as a bare name.")
        rules = f"""RULES
1. Write one Cypher MATCH ... RETURN statement and nothing else.
2. Use only the labels, relationship types and properties above.
3. Values that appear in the question (ids, tag names) are written into the query as literals; tag names are strings, ids are integers.
4. Use RETURN DISTINCT. Do not write LIMIT{', except the ranking limit stated in OUTPUT' if step.kind is Kind.RANKED else ''}. Do not write SKIP.
5. Match CO_OCCURS_WITH without direction. DUPLICATE_OF points from the duplicate to the question it duplicates.
6. Do not write CREATE, MERGE, SET, DELETE, REMOVE or CALL.
7. Give every node you use a variable in MATCH, for example (n:Label), and refer to it as n.property. A label is not a variable. Return the identifier property of the output node, for example RETURN DISTINCT n.id, never the node itself and never a bare property name."""
        schema = _graph_schema(desc)
        lang = "Cypher"
    else:
        how = ""
        if step.bind:
            # the contract field is the right one only when the bound identifiers are of the output entity
            owned = ID_CONTRACT[Store.DOC].get(step.bind.entity, "") if step.bind.entity == step.output else ""
            m = re.search(r"identifier field (\w+)", owned)
            field_name = m.group(1) if m else "<identifier field>"
            token = f"$bind:{step.bind.name}"
            how = (f'Add exactly one clause `{{"terms": {{"{field_name}": "{token}"}}}}` inside bool.filter, together '
                   f'with the other conditions. The text "{token}" is written literally; it is replaced by the list '
                   f'before the query runs. A query without this clause is rejected.')
        if step.mode is Mode.SEARCH:
            mode = (f"MODE: relevance search. Return the {step.k} most relevant documents; use a match or multi_match "
                    f"query on the text fields.")
        else:
            mode = ("MODE: filter. Return every document that satisfies the condition. Use bool.filter with "
                    'a term on doc_type and a match query on the text field with "operator": "and" so that '
                    "every word occurs. Do not use query_string.")
        rules = f"""{mode}
RULES
1. Write one JSON object with exactly one key, "query", and nothing else.
2. Restrict doc_type to the document type of the output entity with a term clause.
3. Use only the fields above. Do not write size, from, sort, scripts or aggregations.
4. Values from the question (words, phrases) are written into the query as literals."""
        schema = _doc_schema(desc)
        lang = "Elasticsearch query"

    bind = ("\n" + _COMMON_BIND.format(name=step.bind.name, how=how) + "\n") if step.bind else ""
    return f"""You translate one sub-question into one {lang} for a single data service.

SERVICE: {desc.display_name}
{desc.capability_summary or ''}

{schema}

{_output_block(step)}
{bind}
{rules}

SUB-QUESTION: {step.question}

Reply with the {lang} only, without explanation or code fences. If the sub-question cannot be answered from this service, reply CANNOT_ANSWER."""


def build_repair_prompt(base: str, previous: str, errors: list[str]) -> str:
    listed = "\n".join(f"- {e}" for e in errors)
    return f"""{base}

YOUR PREVIOUS REPLY
{previous}

IT WAS REJECTED FOR THESE REASONS
{listed}

Reply again with the corrected query only."""


# ----------------------------------------------------------------------------- reply cleaning

_FENCE = re.compile(r"^```[a-zA-Z]*\s*\n?|\n?```\s*$")


def clean_reply(text: str) -> str:
    t = text.strip()
    t = _FENCE.sub("", t).strip()
    t = re.sub(r"^(SQL|CYPHER|QUERY|JSON)\s*:\s*", "", t, flags=re.I).strip()
    return t


# ----------------------------------------------------------------------------- translator

class Translator:
    def __init__(self, provider: MeteredProvider, descriptors: dict, catalogue: Catalogue,
                 *, max_tokens: int = 1500, dry_run: dict | None = None):
        """``descriptors`` maps Store to the Layer 1 ServiceDescriptor of that store. ``dry_run``
        maps Store to a function ``(step, query) -> list[str]`` that asks the store to check the
        query without running it (the clients' ``validate`` methods)."""
        self._llm, self._desc, self._max = provider, descriptors, max_tokens
        self._dry = dry_run or {}
        self._check = StepValidator(catalogue)

    @property
    def llm(self) -> MeteredProvider:
        return self._llm

    def translate(self, step: Step) -> TranslationResult:
        res = TranslationResult(step.id)
        base = build_prompt(step, self._desc[step.store])
        prompt, previous = base, ""
        for attempt in (1, 2):
            resp, u = self._llm.call(prompt, max_tokens=self._max, temperature=0.0)
            res.usage = res.usage + u
            res.attempts = attempt
            res.raw.append(resp.text)
            query = clean_reply(resp.text)
            if not query:
                errs = ["the reply was empty"]
            elif query.upper().startswith(("CANNOT_ANSWER", "-- CANNOT_ANSWER", "// CANNOT_ANSWER")):
                res.query, res.errors = "", ["the model reported that the service cannot answer this sub-question"]
                return res
            else:
                errs = self._check.check(step, query)
                if not errs and step.store in self._dry:
                    try:
                        errs = list(self._dry[step.store](step, query))
                        res.store_rejections += bool(errs)
                    except Exception as ex:                  # the store could not be asked: execution will tell
                        res.notes.append(f"store check unavailable: {type(ex).__name__}: {str(ex)[:120]}")
            res.query, res.errors = query, errs
            if not errs:
                return res
            prompt, previous = build_repair_prompt(base, query, errs), query
        return res
