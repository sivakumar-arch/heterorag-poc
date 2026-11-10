import json

from benchmark_v2 import probe_determinism as pd
from heterorag.llm_provider import LLMResponse

PLAN = {"steps": [
    {"id": "s1", "op": "retrieve", "store": "sql", "question": "users with reputation at least 100", "output": "user"},
    {"id": "s2", "op": "retrieve", "store": "graph", "question": "questions asked by these users", "output": "question",
     "bind": {"input": "s1", "entity": "user"}},
    {"id": "s3", "op": "intersect", "inputs": ["s1", "s2"], "output": "user"}]}


class Fake:
    """Replies from a script, cycling; records every prompt it received."""
    def __init__(self, replies):
        self.replies, self.prompts, self.temps, self.i = replies, [], [], 0

    def complete(self, prompt, *, max_tokens=1024, temperature=0.0):
        self.prompts.append(prompt)
        self.temps.append(temperature)
        r = self.replies[self.i % len(self.replies)]
        self.i += 1
        return LLMResponse(text=r)


def write(tmp_path, rows):
    f = tmp_path / "r.json"
    f.write_text(json.dumps({"rows": rows}))
    return str(f)


def test_only_retrieve_steps_are_probed_and_rows_without_a_plan_are_skipped():
    steps = pd.retrieve_steps([{"question_id": "Q1", "plan": PLAN}, {"question_id": "Q2", "plan": None}, {"question_id": "Q3"}])
    assert [(q, s.id) for q, s in steps] == [("Q1", "s1"), ("Q1", "s2")]


def test_identical_replies_count_as_no_difference_and_k_is_one(tmp_path, capsys):
    f = write(tmp_path, [{"question_id": "Q1", "plan": PLAN}])
    fake = Fake(["SELECT id FROM users"])
    assert pd.main([f, "--repeats", "3", "--prompts", "0"], provider=fake) == 0
    out = capsys.readouterr().out
    assert "2 prompts, each sent 3 times" in out and "K = 1" in out
    assert len(fake.prompts) == 6 and set(fake.temps) == {0.0}
    assert fake.prompts[0] == fake.prompts[1] == fake.prompts[2]        # one prompt, sent repeatedly


def test_a_difference_is_counted_after_cleaning_and_by_verdict(tmp_path, capsys):
    f = write(tmp_path, [{"question_id": "Q1", "plan": PLAN}])
    out_file = tmp_path / "o.json"
    # first prompt: a fenced and an unfenced copy of the same text (differs raw, not cleaned); second: a refusal vs a query
    fake = Fake(["SELECT id FROM users", "```sql\nSELECT id FROM users\n```", "SELECT id FROM users",
                 "CANNOT_ANSWER", "SELECT id FROM users", "CANNOT_ANSWER"])
    assert pd.main([f, "--repeats", "3", "--prompts", "0", "--out", str(out_file)], provider=fake) == 0
    t = json.loads(out_file.read_text())["tally"]
    assert t["prompts"] == 2 and t["differ_raw"] == 2 and t["differ_cleaned"] == 1 and t["differ_verdict"] >= 1
    assert "K = 5" in capsys.readouterr().out


def test_bad_input_is_refused(tmp_path):
    assert pd.main([str(tmp_path / "none.json")], provider=Fake(["x"])) == 2
    f = write(tmp_path, [{"question_id": "Q1", "plan": None}])
    assert pd.main([f], provider=Fake(["x"])) == 2
    assert pd.main([f, "--repeats", "1"], provider=Fake(["x"])) == 2


class FakeClient:
    def __init__(self):
        self.options = []

    def chat(self, **kw):
        self.options.append(kw["options"])
        return {"message": {"content": "SELECT 1"}, "prompt_eval_count": 1, "eval_count": 1}


def test_the_pipeline_provider_does_not_send_the_temperature_and_zero_sampling_does(monkeypatch):
    # the pipeline's provider for an Ollama model is built with model_default_sampling=True (make_provider)
    from heterorag.llm_provider import OllamaProvider
    c1, c0 = FakeClient(), FakeClient()
    OllamaProvider("m", client=c1, model_default_sampling=True).complete("p", temperature=0.0)
    assert "temperature" not in c1.options[0]
    p = pd.probe_provider("ollama:m", "zero", client=c0)
    p.complete("p", temperature=0.0)
    assert c0.options[0]["temperature"] == 0.0
