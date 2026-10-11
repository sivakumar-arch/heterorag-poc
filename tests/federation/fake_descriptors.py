"""
Lightweight stand-ins for the three Layer 1 service descriptors.

They have the attributes the planner, the translator and the catalogue read, without the
pydantic models, so the federation tests run in environments without Layer 1 dependencies.
``test_fake_descriptors_match_the_real_ones`` (skipped when pydantic is missing) asserts that
the stand-ins agree with the real descriptors, so they cannot drift silently.
"""
from types import SimpleNamespace as NS

from heterorag.federation.contract import catalogue_from_descriptors
from heterorag.federation.plan import Store


def _col(name, typ="integer", desc=""):
    return NS(name=name, data_type=typ, description=desc)


def _table(name, cols, desc=""):
    return NS(table_name=name, description=desc, columns=[_col(*c) if isinstance(c, tuple) else _col(c) for c in cols])


SQL = NS(
    display_name="User & Activity Service", persistence_type="sql",
    capability_summary="Numeric attributes of users and posts: reputation, scores, view counts, badges, tag usage counts, "
                       "dates. No relationships and no text.",
    schema_spec=NS(tables=[
        _table("users", [("id",), ("reputation",), ("creation_date", "timestamptz"), ("display_name", "text"),
                         ("location", "text"), ("up_votes",), ("down_votes",), ("views",), ("account_id",)]),
        _table("posts", [("id",), ("post_type_id", "smallint", "1=Question, 2=Answer"), ("score",), ("view_count",),
                         ("answer_count",), ("comment_count",), ("creation_date", "timestamptz"),
                         ("last_activity_date", "timestamptz"), ("closed_date", "timestamptz")]),
        _table("votes", [("id",), ("post_id",), ("vote_type_id", "smallint"), ("creation_date", "timestamptz"),
                         ("user_id",), ("bounty_amount",)]),
        _table("badges", [("id",), ("user_id",), ("name", "text"), ("class", "smallint"), ("tag_based", "boolean"),
                          ("date", "timestamptz")]),
        _table("tags", [("id",), ("tag_name", "text"), ("count",)]),
        _table("comments", [("id",), ("post_id",), ("score",), ("user_id",), ("creation_date", "timestamptz")]),
    ]))

GRAPH = NS(
    display_name="Knowledge Graph Service", persistence_type="graph",
    capability_summary="Relationships: authorship, answers of a question, accepted answers, tags of a question, "
                       "tag co-occurrence, duplicates and links. No attributes and no text.",
    schema_spec=NS(
        node_types=[NS(label="User", properties=["id"], description="user"),
                    NS(label="Question", properties=["id"], description="question"),
                    NS(label="Answer", properties=["id"], description="answer"),
                    NS(label="Tag", properties=["name"], description="tag")],
        edge_types=[NS(edge_type=e, from_label=a, to_label=b, properties=p, description="")
                    for e, a, b, p in [("ASKED", "User", "Question", []), ("ANSWERED", "User", "Answer", []),
                                       ("ANSWERS", "Answer", "Question", []), ("ACCEPTED", "Question", "Answer", []),
                                       ("TAGGED_WITH", "Question", "Tag", []),
                                       ("CO_OCCURS_WITH", "Tag", "Tag", ["weight"]),
                                       ("LINKED_TO", "Question", "Question", []),
                                       ("DUPLICATE_OF", "Question", "Question", [])]]))


def _f(name, typ="keyword"):
    return NS(name=name, field_type=typ, searchable=typ == "text", description="")


DOC = NS(
    display_name="Content Service", persistence_type="document",
    capability_summary="Text: question titles and bodies, answer bodies, comments, user profiles, tag wikis. "
                       "No attributes and no relationships.",
    schema_spec=NS(doc_types=[
        NS(doc_type="question", description="question posts", fields=[_f("doc_type"), _f("post_id", "integer"), _f("title", "text"), _f("body", "text")]),
        NS(doc_type="answer", description="answer posts", fields=[_f("doc_type"), _f("post_id", "integer"), _f("body", "text")]),
        NS(doc_type="comment", description="comments", fields=[_f("doc_type"), _f("post_id", "integer"), _f("user_id", "integer"), _f("body", "text")]),
        NS(doc_type="user_about", description="user profiles", fields=[_f("doc_type"), _f("user_id", "integer"), _f("body", "text")]),
        NS(doc_type="tag_wiki", description="tag wikis", fields=[_f("doc_type"), _f("tag_name"), _f("body", "text")]),
    ]))

DESCRIPTORS = {Store.SQL: SQL, Store.GRAPH: GRAPH, Store.DOC: DOC}


def catalogue():
    return catalogue_from_descriptors(SQL, GRAPH, DOC)
