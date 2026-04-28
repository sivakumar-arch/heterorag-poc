"""
heterorag/layer2/prompts.py
============================
Schema-aware prompt builders for Layer 2 query translation.

Each builder takes a ServiceDescriptor and a natural-language query and returns
a fully formed prompt string ready for the TranslationLLM.

Design principles:
  - Prompts embed the *full* SchemaSpec so the LLM has complete schema context.
  - Output format is specified precisely and unambiguously in the prompt.
  - Each prompt ends with a clear instruction to output ONLY the query, nothing else.
  - Fallback descriptions are derived from SchemaSpec when capability_summary is absent.

Reference: Foundation Doc v1.6 §6, OQ2 (capability_summary for LLM prompts).
"""

from __future__ import annotations

from heterorag.layer1.models import (
    DocumentSchemaSpec,
    GraphSchemaSpec,
    PersistenceType,
    SQLSchemaSpec,
    ServiceDescriptor,
)


# ---------------------------------------------------------------------------
# SQL prompt builder  (NL → PostgreSQL)
# ---------------------------------------------------------------------------

def build_sql_prompt(descriptor: ServiceDescriptor, natural_query: str) -> str:
    """
    Builds a prompt that instructs the LLM to translate a natural language query
    into a valid PostgreSQL SELECT statement against the described schema.
    """
    spec: SQLSchemaSpec = descriptor.schema_spec

    # Build a compact schema block from the TableSpec list
    schema_lines: list[str] = []
    for table in spec.tables:
        col_parts = []
        for col in table.columns:
            null_flag = "" if col.nullable else " NOT NULL"
            desc_part = f"  -- {col.description}" if col.description else ""
            col_parts.append(f"  {col.name} {col.data_type}{null_flag}{desc_part}")
        table_desc = f"  -- {table.description}" if table.description else ""
        schema_lines.append(
            f"TABLE {table.table_name}:{table_desc}\n" + "\n".join(col_parts)
        )

    schema_block = "\n\n".join(schema_lines)
    capability = descriptor.capability_summary or (
        f"A SQL database service exposing {len(spec.tables)} tables."
    )

    return f"""You are a precise SQL query translator for a Stack Overflow analytics database.

SERVICE: {descriptor.display_name}
CAPABILITY: {capability}

SCHEMA:
{schema_block}

RULES:
1. Write a single valid PostgreSQL SELECT statement only.
2. Use only the tables and columns listed above — do not invent any.
3. Always include a LIMIT clause (default LIMIT 20 unless the question implies otherwise).
4. Do NOT include any explanation, markdown, or code fences — output ONLY the SQL.
5. If the question cannot be answered from this schema alone, write: -- CANNOT_ANSWER

QUESTION: {natural_query}

SQL:"""


# ---------------------------------------------------------------------------
# Cypher prompt builder  (NL → Neo4j Cypher)
# ---------------------------------------------------------------------------

def build_cypher_prompt(descriptor: ServiceDescriptor, natural_query: str) -> str:
    """
    Builds a prompt that instructs the LLM to translate a natural language query
    into a valid Neo4j Cypher MATCH...RETURN statement.
    """
    spec: GraphSchemaSpec = descriptor.schema_spec

    # Node types block
    node_lines = []
    for n in spec.node_types:
        props = ", ".join(n.properties)
        desc = f"  // {n.description}" if n.description else ""
        node_lines.append(f"  (:{n.label} {{{props}}}){desc}")

    # Edge types block
    edge_lines = []
    for e in spec.edge_types:
        props = f" {{{', '.join(e.properties)}}}" if e.properties else ""
        desc = f"  // {e.description}" if e.description else ""
        edge_lines.append(
            f"  (:{e.from_label})-[:{e.edge_type}{props}]->(:{e.to_label}){desc}"
        )

    capability = descriptor.capability_summary or (
        f"A Neo4j graph service with {len(spec.node_types)} node types "
        f"and {len(spec.edge_types)} relationship types."
    )

    return f"""You are a precise Cypher query translator for a Stack Overflow knowledge graph.

SERVICE: {descriptor.display_name}
CAPABILITY: {capability}

NODE TYPES:
{chr(10).join(node_lines)}

RELATIONSHIP TYPES:
{chr(10).join(edge_lines)}

RULES:
1. Write a single valid Cypher MATCH...RETURN statement only.
2. Use only the node labels, relationship types, and properties listed above.
3. Always include LIMIT (default LIMIT 20 unless the question implies otherwise).
4. For CO_OCCURS_WITH edges use undirected match: (t1)-[:CO_OCCURS_WITH]-(t2)
5. Do NOT include any explanation, markdown, or code fences — output ONLY the Cypher.
6. If the question cannot be answered from this graph alone, write: // CANNOT_ANSWER

QUESTION: {natural_query}

CYPHER:"""


# ---------------------------------------------------------------------------
# BM25 prompt builder  (NL → Elasticsearch query string)
# ---------------------------------------------------------------------------

def build_bm25_prompt(descriptor: ServiceDescriptor, natural_query: str) -> str:
    """
    Builds a prompt that instructs the LLM to extract the optimal BM25 search
    terms from a natural language query for full-text retrieval.

    The output is an Elasticsearch query_string expression — not a full JSON body.
    The retrieval executor wraps it in the appropriate ES request.
    """
    spec: DocumentSchemaSpec = descriptor.schema_spec

    # List searchable fields
    searchable_fields: list[str] = []
    for doc_type in spec.doc_types:
        for field in doc_type.fields:
            if field.searchable and field.field_name not in searchable_fields:
                label = f"{field.field_name} ({field.field_type})"
                if field.description:
                    label += f" -- {field.description}"
                searchable_fields.append(label)

    fields_block = "\n".join(f"  - {f}" for f in searchable_fields)
    doc_types_block = "\n".join(
        f"  - {dt.doc_type}: {dt.description or 'no description'}"
        for dt in spec.doc_types
    )
    capability = descriptor.capability_summary or (
        f"A document store with {len(spec.doc_types)} document types."
    )

    return f"""You are a precise BM25 search query builder for a Stack Overflow full-text search index.

SERVICE: {descriptor.display_name}
CAPABILITY: {capability}

DOCUMENT TYPES IN INDEX:
{doc_types_block}

SEARCHABLE FIELDS:
{fields_block}

RULES:
1. Output ONLY an Elasticsearch query_string expression (not JSON, not an explanation).
2. Extract the core informational terms from the question.
3. Remove stop words and question phrasing — keep only content terms.
4. Use Lucene boolean operators (AND, OR, NOT) only when they improve precision.
5. Prefer terms that would appear in the body or title of a relevant Stack Overflow post.
6. Do NOT use field: prefix syntax — the executor applies multi_match across all text fields.
7. Do NOT include any explanation, markdown, or code fences — output ONLY the query string.

QUESTION: {natural_query}

QUERY STRING:"""


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def build_translation_prompt(
    descriptor:    ServiceDescriptor,
    natural_query: str,
) -> str:
    """
    Dispatch to the correct prompt builder based on the descriptor's persistence type.
    Raises ValueError for unrecognised persistence types.
    """
    pt = descriptor.persistence_type
    if pt == PersistenceType.SQL:
        return build_sql_prompt(descriptor, natural_query)
    elif pt == PersistenceType.GRAPH:
        return build_cypher_prompt(descriptor, natural_query)
    elif pt == PersistenceType.DOCUMENT:
        return build_bm25_prompt(descriptor, natural_query)
    else:
        raise ValueError(f"No prompt builder for persistence type: {pt}")
