"""
heterorag/federation/guarantee.py
=================================
Guarantee levels for partial results and their propagation through the operators
(Interface Specification v2.1, section 6).

A guarantee states how a computed value relates to the true value:

    EXACT      the value is the true value
    SUBSET     every returned item is correct; items may be missing   (sets)
    SUPERSET   every correct item is returned; extra items possible   (sets)
    LOWER      the true value is at least the returned value          (scalars)
    UPPER      the true value is at most the returned value           (scalars)
    NONE       no relation can be stated

The rules follow from monotonicity. Intersection, union and filtered retrieval are monotone in
their inputs; set difference is monotone in its first input and antitone in its second;
cardinality is monotone. The tests check every rule against brute-force semantics.

Not implemented yet: top-k ordering over a set. It requires a value-fetching step and is added
together with the aggregate question classes.
"""

from __future__ import annotations

from enum import Enum
from functools import reduce


class Guarantee(str, Enum):
    EXACT = "exact"
    SUBSET = "subset"
    SUPERSET = "superset"
    LOWER = "lower"
    UPPER = "upper"
    NONE = "none"


class Status(str, Enum):
    COMPLETE = "complete"     # the whole result was read
    PARTIAL = "partial"       # reading stopped early after at least one row
    FAILED = "failed"         # nothing usable


class Kind(str, Enum):
    SET = "set"
    RANKED = "ranked"
    SCALAR = "scalar"


def seed(kind: Kind, status: Status) -> Guarantee:
    """Guarantee of a retrieve step from its read status."""
    if status is Status.COMPLETE:
        return Guarantee.EXACT
    if status is Status.PARTIAL and kind is Kind.SET:
        return Guarantee.SUBSET
    return Guarantee.NONE


def flip(g: Guarantee) -> Guarantee:
    """Guarantee of the complement direction (used for the subtracted operand)."""
    if g is Guarantee.SUBSET:
        return Guarantee.SUPERSET
    if g is Guarantee.SUPERSET:
        return Guarantee.SUBSET
    return g


_SET_LEVELS = (Guarantee.EXACT, Guarantee.SUBSET, Guarantee.SUPERSET, Guarantee.NONE)


def mono(a: Guarantee, b: Guarantee) -> Guarantee:
    """Combine two operands of an operator that is monotone in both."""
    for g in (a, b):
        if g not in _SET_LEVELS:
            raise ValueError(f"set operators take set guarantees, got {g.value}")
    if Guarantee.NONE in (a, b):
        return Guarantee.NONE
    if a is Guarantee.EXACT:
        return b
    if b is Guarantee.EXACT:
        return a
    return a if a is b else Guarantee.NONE


def intersect(gs: list[Guarantee]) -> Guarantee:
    return reduce(mono, gs)


def union(gs: list[Guarantee]) -> Guarantee:
    return reduce(mono, gs)


def subtract(a: Guarantee, b: Guarantee) -> Guarantee:
    """Set difference A minus B: monotone in A, antitone in B."""
    return mono(a, flip(b))


def count(g: Guarantee) -> Guarantee:
    if g is Guarantee.EXACT:
        return Guarantee.EXACT
    if g is Guarantee.SUBSET:
        return Guarantee.LOWER
    if g is Guarantee.SUPERSET:
        return Guarantee.UPPER
    return Guarantee.NONE


def retrieve_with_bind(bound_input: Guarantee, own: Guarantee) -> Guarantee:
    """A filtered retrieval restricted to a bound set is monotone in both the bound set and
    its own result."""
    return mono(bound_input, own)


def holds(level: Guarantee, returned, truth) -> bool:
    """Does the stated guarantee hold for a returned value against the truth?
    Sets are compared with set inclusion, scalars with numeric order."""
    if level is Guarantee.NONE:
        return True
    if level is Guarantee.EXACT:
        return returned == truth
    if level is Guarantee.SUBSET:
        return set(returned) <= set(truth)
    if level is Guarantee.SUPERSET:
        return set(returned) >= set(truth)
    if level is Guarantee.LOWER:
        return returned <= truth
    if level is Guarantee.UPPER:
        return returned >= truth
    raise ValueError(level)
