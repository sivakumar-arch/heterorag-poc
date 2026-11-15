"""Premise RD: an Elasticsearch response that reports a timeout or failed shards is not a complete read.
The sampling a provider actually sent is recorded."""
from heterorag.federation.clients import ElasticsearchClient, _partial_reason
from heterorag.llm_provider import AnthropicProvider, OllamaProvider


def test_partial_reason_reads_timed_out_and_failed_shards():
    assert _partial_reason({"timed_out": False, "_shards": {"total": 3, "failed": 0}}) is None
    assert "timed_out" in _partial_reason({"timed_out": True})
    assert "1 of 3 shards" in _partial_reason({"_shards": {"total": 3, "failed": 1}})
    assert _partial_reason({"hits": {"hits": []}}) is None          # a response without the fields is unchanged


class _Es:
    def __init__(self, resp):
        self.resp = resp
    def search(self, **kw):
        return self.resp
    def open_point_in_time(self, **kw):
        return {"id": "p"}
    def close_point_in_time(self, **kw):
        pass


def test_a_search_with_failed_shards_is_not_exhausted_and_carries_the_reason():
    resp = {"hits": {"hits": [{"_source": {"id": 1}}]}, "_shards": {"total": 2, "failed": 1}, "timed_out": False}
    client = ElasticsearchClient(_Es(resp), "idx")
    out = client._search({"match_all": {}}, "id", 5)
    assert out.values == [1] and out.exhausted is False and "shards failed" in out.error


def test_a_filter_page_with_a_timeout_after_some_rows_is_partial():
    ok = {"hits": {"hits": [{"_source": {"id": 1}, "sort": [1]}]}, "timed_out": False, "_shards": {"failed": 0}}
    bad = {"hits": {"hits": [{"_source": {"id": 2}, "sort": [2]}]}, "timed_out": True}
    seq = iter([ok, bad])
    class Es(_Es):
        def search(self, **kw):
            return next(seq)
    out = ElasticsearchClient(Es(None), "idx", page=1)._filter({"match_all": {}}, "id", 100)
    assert out.values == [1] and out.exhausted is False and "timed_out" in out.error


def test_a_clean_filter_read_is_unchanged():
    seq = iter([{"hits": {"hits": [{"_source": {"id": 1}, "sort": [1]}]}}, {"hits": {"hits": []}}])
    class Es(_Es):
        def search(self, **kw):
            return next(seq)
    out = ElasticsearchClient(Es(None), "idx", page=1)._filter({"match_all": {}}, "id", 100)
    assert out.values == [1] and out.exhausted is True and out.error is None


class _Msgs:
    def __init__(self, reject):
        self.reject = reject
    def create(self, **kw):
        if self.reject and "temperature" in kw:
            raise TypeError("create() got an unexpected keyword argument 'temperature'")
        return object()


def _anthropic(reject):
    p = AnthropicProvider.__new__(AnthropicProvider)
    p.model, p._client = "m", type("C", (), {"messages": _Msgs(reject)})()
    return p


def test_sampling_sent_is_temperature_when_accepted_and_model_default_when_rejected():
    ok, rej = _anthropic(False), _anthropic(True)
    assert ok.sampling_sent() == "none"
    ok._create({}, 0.0)
    rej._create({}, 0.0)
    assert ok.sampling_sent() == "temperature" and rej.sampling_sent() == "model_default"


def test_ollama_reports_its_mode():
    p = OllamaProvider.__new__(OllamaProvider)
    p.model_default_sampling = True
    assert p.sampling_sent() == "model_default"
    p.model_default_sampling = False
    assert p.sampling_sent() == "temperature"
