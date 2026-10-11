"""
Guarantee algebra, checked against brute-force set semantics.

For a universe of four elements, every true pair of sets (A, B) and every approximation of
each operand that is consistent with its stated guarantee is enumerated. The operator is
applied to the approximations and the propagated guarantee must hold against the true result.
A second check shows the rules are tight: where the rule says NONE, neither SUBSET nor SUPERSET
holds universally, so the rule is not needlessly weak.
"""
from itertools import chain, combinations

import pytest

from heterorag.federation import guarantee as G
from heterorag.federation.guarantee import Guarantee as Gt, Kind, Status

U = frozenset(range(4))
SET_LEVELS = (Gt.EXACT, Gt.SUBSET, Gt.SUPERSET)


def subsets(s):
    s = sorted(s)
    return [frozenset(c) for c in chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))]


ALL = subsets(U)


def approximations(truth: frozenset, level: Gt):
    if level is Gt.EXACT:
        return [truth]
    if level is Gt.SUBSET:
        return [a for a in ALL if a <= truth]
    if level is Gt.SUPERSET:
        return [a for a in ALL if a >= truth]
    raise ValueError(level)


def universally_holds(level, op, la, lb):
    """Does ``level`` hold for op(approx A, approx B) against op(A, B) in every case?"""
    for a in ALL:
        for b in ALL:
            truth = op(a, b)
            for xa in approximations(a, la):
                for xb in approximations(b, lb):
                    if not G.holds(level, op(xa, xb), truth):
                        return False
    return True


OPS = {
    "intersect": (lambda a, b: a & b, lambda la, lb: G.intersect([la, lb])),
    "union": (lambda a, b: a | b, lambda la, lb: G.union([la, lb])),
    "subtract": (lambda a, b: a - b, lambda la, lb: G.subtract(la, lb)),
    "bind": (lambda a, b: a & b, lambda la, lb: G.retrieve_with_bind(la, lb)),
}


@pytest.mark.parametrize("name", sorted(OPS))
@pytest.mark.parametrize("la", SET_LEVELS)
@pytest.mark.parametrize("lb", SET_LEVELS)
def test_propagated_guarantee_is_sound(name, la, lb):
    op, rule = OPS[name]
    level = rule(la, lb)
    assert universally_holds(level, op, la, lb), (name, la, lb, level)


@pytest.mark.parametrize("name", sorted(OPS))
@pytest.mark.parametrize("la", SET_LEVELS)
@pytest.mark.parametrize("lb", SET_LEVELS)
def test_none_is_only_returned_when_no_stronger_guarantee_exists(name, la, lb):
    op, rule = OPS[name]
    if rule(la, lb) is Gt.NONE:
        assert not universally_holds(Gt.SUBSET, op, la, lb)
        assert not universally_holds(Gt.SUPERSET, op, la, lb)


@pytest.mark.parametrize("level,expected", [(Gt.EXACT, Gt.EXACT), (Gt.SUBSET, Gt.LOWER),
                                            (Gt.SUPERSET, Gt.UPPER), (Gt.NONE, Gt.NONE)])
def test_count_rule(level, expected):
    assert G.count(level) is expected
    if level is not Gt.NONE:
        for truth in ALL:
            for approx in approximations(truth, level):
                assert G.holds(expected, len(approx), len(truth))


def test_multiway_intersection_and_union_fold():
    assert G.intersect([Gt.EXACT, Gt.SUBSET, Gt.SUBSET]) is Gt.SUBSET
    assert G.intersect([Gt.SUBSET, Gt.SUPERSET, Gt.EXACT]) is Gt.NONE
    assert G.union([Gt.SUPERSET, Gt.SUPERSET]) is Gt.SUPERSET


def test_none_absorbs():
    for lvl in SET_LEVELS:
        assert G.mono(Gt.NONE, lvl) is Gt.NONE
        assert G.mono(lvl, Gt.NONE) is Gt.NONE
    assert G.subtract(Gt.NONE, Gt.EXACT) is Gt.NONE
    assert G.subtract(Gt.EXACT, Gt.NONE) is Gt.NONE


def test_set_operators_reject_scalar_levels():
    with pytest.raises(ValueError):
        G.mono(Gt.LOWER, Gt.EXACT)


def test_seed_rules():
    assert G.seed(Kind.SET, Status.COMPLETE) is Gt.EXACT
    assert G.seed(Kind.SET, Status.PARTIAL) is Gt.SUBSET
    assert G.seed(Kind.SET, Status.FAILED) is Gt.NONE
    # a partial ranked or scalar read says nothing
    assert G.seed(Kind.RANKED, Status.PARTIAL) is Gt.NONE
    assert G.seed(Kind.SCALAR, Status.PARTIAL) is Gt.NONE
    assert G.seed(Kind.SCALAR, Status.COMPLETE) is Gt.EXACT


def test_flip_is_an_involution():
    for lvl in Gt:
        assert G.flip(G.flip(lvl)) is lvl
