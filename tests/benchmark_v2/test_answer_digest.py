from benchmark_v2.run_slice import answer_digest
from heterorag.federation.answer import Answer


def test_sets_compare_equal_whatever_their_order_and_differ_when_an_item_differs():
    a = answer_digest(Answer("ids", [3, 1, 2]))
    b = answer_digest(Answer("ids", [1, 2, 3]))
    c = answer_digest(Answer("ids", [1, 2, 4]))
    assert a == b and a["size"] == 3 and a["sha1"] != c["sha1"]


def test_ranked_order_matters_and_head_is_capped():
    a = answer_digest(Answer("ranked", list(range(30))))
    b = answer_digest(Answer("ranked", list(reversed(range(30)))))
    assert a["sha1"] != b["sha1"] and len(a["head"]) == 20


def test_scalar_and_abstain():
    assert answer_digest(Answer("scalar", 42)) == {"type": "scalar", "value": 42}
    assert answer_digest(Answer("abstain", None, reason="incomplete")) == {"type": "abstain", "reason": "incomplete"}
