"""Long runs survive an unreachable API: infrastructure failures are not results, runs resume per question."""

import json

import pytest

from benchmark_v2 import agent_baseline as ab
from benchmark_v2 import resume, run_slice
from tests.benchmark_v2.test_agent_baseline import DESCRIPTORS, Stores


class APIConnectionError(Exception):
    """Same name as the SDK's exception when the network is down."""


def test_infrastructure_exceptions_are_told_apart_from_defects():
    assert resume.is_infra_exception(APIConnectionError("Connection error."))
    assert resume.is_infra_exception(TimeoutError("slow"))
    assert not resume.is_infra_exception(KeyError("x"))
    assert not resume.is_infra_exception(MemoryError("boom"))

    class Overloaded(Exception):
        status_code = 529
    assert resume.is_infra_exception(Overloaded("overloaded"))


def test_guard_trips_only_on_consecutive_infrastructure_failures():
    g = resume.Guard(3)
    assert [g.add({"infra": True}), g.add({"infra": True}), g.add({}), g.add({"infra": True}),
            g.add({"infra": True}), g.add({"infra": True})] == [False, False, False, False, False, True]


def test_save_is_atomic_and_load_drops_infrastructure_rows(tmp_path):
    p = str(tmp_path / "o.json")
    resume.save(p, {"n": 3}, [{"question_id": "a"}, {"question_id": "b", "infra": True}, {"question_id": "c"}], False)
    assert not (tmp_path / "o.json.tmp").exists()
    data = json.loads((tmp_path / "o.json").read_text())
    assert data["meta"]["complete"] is False and len(data["rows"]) == 3
    assert set(resume.load(p)) == {"a", "c"}
    (tmp_path / "bad.json").write_text("{not json")
    assert resume.load(str(tmp_path / "bad.json")) == {} and resume.load(str(tmp_path / "none.json")) == {}
    assert resume.is_complete([{}, {}], 2) and not resume.is_complete([{}, {"infra": True}], 2)
    assert not resume.is_complete([{}], 2)


# ----------------------------------------------------------------------------- the pipeline runner

def qrec(i):
    return {"question_id": f"Q{i}", "template": "T", "split": "dev", "text": f"q{i}", "stratum": "s"}


def fake_row(rec):
    return {"question_id": rec["question_id"], "template": "T", "split": "dev", "plan_valid": True, "plan_equal": True,
            "translation_ok": True, "plan_repairs": 0, "translation_repairs": 0, "store_rejections": 0,
            "outcome": "correct", "usage": {"planner": {"prompt_tokens": 1, "reply_tokens": 1},
                                            "translator": {"prompt_tokens": 1, "reply_tokens": 1}},
            "total_s": 0.1, "bind_decisions": [], "plan_abstain": None, "plan_errors": [], "reason": "",
            "translation_errors": {}, "step_errors": []}


class Fed:
    """Stands in for the Federation; ``run_one`` is patched to use ``fail``."""
    def __init__(self, fail=()):
        self.fail, self.calls = set(fail), []


def patch_run_one(monkeypatch, fed, exc=APIConnectionError("Connection error.")):
    def run_one(rec, federation):
        fed.calls.append(rec["question_id"])
        if rec["question_id"] in fed.fail:
            raise exc
        return fake_row(rec)
    monkeypatch.setattr(run_slice, "run_one", run_one)


def write_questions(tmp_path, n):
    q = tmp_path / "q.jsonl"
    q.write_text("\n".join(json.dumps(qrec(i)) for i in range(n)))
    return str(q)


def test_three_failures_in_a_row_stop_the_run_and_save_the_results_so_far(tmp_path, monkeypatch):
    fed = Fed(fail={"Q1", "Q2", "Q3"})
    patch_run_one(monkeypatch, fed)
    out = tmp_path / "o.json"
    code = run_slice.main(["--questions", write_questions(tmp_path, 6), "--out", str(out), "--resume"], federation=fed)
    assert code == resume.EXIT_INTERRUPTED
    data = json.loads(out.read_text())
    assert [r["question_id"] for r in data["rows"]] == ["Q0"] and data["meta"]["complete"] is False
    assert fed.calls == ["Q0", "Q1", "Q2", "Q3"]                      # Q4 and Q5 were not burned through


def test_resume_keeps_finished_rows_and_redoes_only_the_rest(tmp_path, monkeypatch):
    fed = Fed(fail={"Q1", "Q2", "Q3"})
    patch_run_one(monkeypatch, fed)
    out, q = tmp_path / "o.json", write_questions(tmp_path, 6)
    run_slice.main(["--questions", q, "--out", str(out), "--resume"], federation=fed)
    fed2 = Fed()
    patch_run_one(monkeypatch, fed2)
    code = run_slice.main(["--questions", q, "--out", str(out), "--resume"], federation=fed2)
    assert code == 0 and fed2.calls == ["Q1", "Q2", "Q3", "Q4", "Q5"]
    data = json.loads(out.read_text())
    assert data["meta"]["complete"] is True and data["meta"]["resumed_rows"] == 1
    assert [r["question_id"] for r in data["rows"]] == [f"Q{i}" for i in range(6)]
    assert not any(r.get("infra") for r in data["rows"])


def test_one_isolated_failure_leaves_the_run_incomplete_and_is_redone_on_resume(tmp_path, monkeypatch):
    fed = Fed(fail={"Q2"})
    patch_run_one(monkeypatch, fed)
    out, q = tmp_path / "o.json", write_questions(tmp_path, 4)
    assert run_slice.main(["--questions", q, "--out", str(out)], federation=fed) == resume.EXIT_INTERRUPTED
    data = json.loads(out.read_text())
    assert data["meta"]["complete"] is False and [r.get("infra", False) for r in data["rows"]] == [False, False, True, False]
    fed2 = Fed()
    patch_run_one(monkeypatch, fed2)
    assert run_slice.main(["--questions", q, "--out", str(out), "--resume"], federation=fed2) == 0
    assert fed2.calls == ["Q2"] and json.loads(out.read_text())["meta"]["complete"] is True


def test_a_defect_in_our_own_code_stays_a_result_and_does_not_loop(tmp_path, monkeypatch):
    fed = Fed(fail={"Q1"})
    patch_run_one(monkeypatch, fed, exc=KeyError("bug"))
    out = tmp_path / "o.json"
    assert run_slice.main(["--questions", write_questions(tmp_path, 3), "--out", str(out)], federation=fed) == 0
    rows = json.loads(out.read_text())["rows"]
    assert rows[1]["outcome"] == "abstain_unexpected" and "crash: KeyError" in rows[1]["reason"] and "infra" not in rows[1]


# ----------------------------------------------------------------------------- the agent runner

class Down:
    model = "m"
    models_seen: list = []

    def turn(self, *a):
        raise APIConnectionError("Connection error.")


def agent_questions(tmp_path, n):
    q = tmp_path / "aq.jsonl"
    q.write_text("\n".join(json.dumps({"question_id": f"D05-{i}", "template": "D05", "class": "D", "split": "dev",
                                        "stratum": "s", "text": "t?", "answer_type": "ids", "gold": [1]})
                           for i in range(n)))
    return str(q)


def test_agent_api_failure_is_flagged_as_infrastructure_and_stops_after_three(tmp_path):
    out = tmp_path / "a.json"
    code = ab.main(["--questions", agent_questions(tmp_path, 6), "--out", str(out), "--resume"], llm=Down(),
                   stores=Stores(), desc=DESCRIPTORS)
    assert code == resume.EXIT_INTERRUPTED
    data = json.loads(out.read_text())
    assert data["rows"] == [] and data["meta"]["complete"] is False


def test_agent_run_resumes_to_completion(tmp_path):
    from tests.benchmark_v2.test_agent_baseline import C, Scripted, T, final_set
    out, q = tmp_path / "a.json", agent_questions(tmp_path, 2)
    ab.main(["--questions", q, "--out", str(out), "--resume"], llm=Down(), stores=Stores(), desc=DESCRIPTORS)
    llm = Scripted([T(final_set(1)), T(final_set(1))])
    code = ab.main(["--questions", q, "--out", str(out), "--resume"], llm=llm, stores=Stores(), desc=DESCRIPTORS)
    assert code == 0
    data = json.loads(out.read_text())
    assert data["meta"]["complete"] is True and [r["outcome"] for r in data["rows"]] == ["correct", "correct"]
    assert not any(r.get("infra") for r in data["rows"])


def test_the_report_refuses_an_incomplete_run(tmp_path):
    from benchmark_v2 import cp1_report as cp
    p = tmp_path / "x.json"
    resume.save(str(p), {"n": 1}, [{"question_id": "a", "template": "T", "outcome": "correct",
                                    "usage": {"prompt_tokens": 1, "reply_tokens": 1}}], False)
    with pytest.raises(ValueError, match="not complete"):
        cp.System("x", [str(p)])


# ----------------------------------------------------------------------------- a run that cannot work as configured (CP1b, S5)

class NoKey:
    """The SDK raises this before any request when no API key is set."""
    model = "m"
    models_seen: list = []
    calls = 0

    def turn(self, *a):
        NoKey.calls += 1
        raise TypeError('"Could not resolve authentication method. Expected one of api_key, auth_token, or credentials to be set."')


def test_setup_failures_are_told_apart_from_results_and_from_the_network():
    assert resume.is_setup_failure({"stop": "error", "turns": 0, "reason": "TypeError: no key"})
    assert resume.is_setup_failure({"reason": "crash: KeyError: 'x'", "outcome": "abstain_unexpected"})
    assert not resume.is_setup_failure({"stop": "error", "turns": 0, "infra": True})           # the network: other guard
    assert not resume.is_setup_failure({"stop": "answered", "turns": 4, "reason": ""})
    assert not resume.is_setup_failure({"stop": "error", "turns": 2, "reason": "APIError after two replies"})
    g = resume.SetupGuard(3)
    bad = {"stop": "error", "turns": 0}
    assert [g.add(bad), g.add(bad), g.add({"stop": "answered", "turns": 3}), g.add(bad), g.add(bad), g.add(bad)] == \
        [False, False, False, False, False, True]


def test_agent_run_without_an_api_key_stops_after_three_and_saves_no_failed_row(tmp_path, capsys):
    NoKey.calls = 0
    out = tmp_path / "a.json"
    code = ab.main(["--questions", agent_questions(tmp_path, 6), "--out", str(out)], llm=NoKey(), stores=Stores(),
                   desc=DESCRIPTORS)
    assert code == resume.EXIT_SETUP and NoKey.calls == 3                  # questions 4 to 6 were not tried
    data = json.loads(out.read_text())
    assert data["rows"] == [] and data["meta"]["complete"] is False        # nothing that looks like a result
    err = capsys.readouterr().err
    assert "Could not resolve authentication method" in err and "ANTHROPIC_API_KEY" in err and "--resume" in err


def test_the_stop_keeps_earlier_good_rows_and_a_rerun_with_a_working_model_completes(tmp_path):
    from tests.benchmark_v2.test_agent_baseline import Scripted, T, final_set

    class OneThenNoKey(NoKey):
        def __init__(self):
            self.ok = Scripted([T(final_set(1))])

        def turn(self, *a):
            if self.ok.turns:
                return self.ok.turn(*a)
            return super().turn(*a)

    out, q = tmp_path / "a.json", agent_questions(tmp_path, 6)
    assert ab.main(["--questions", q, "--out", str(out), "--resume"], llm=OneThenNoKey(), stores=Stores(),
                   desc=DESCRIPTORS) == resume.EXIT_SETUP
    assert [r["question_id"] for r in json.loads(out.read_text())["rows"]] == ["D05-0"]
    llm = Scripted([T(final_set(1)) for _ in range(5)])
    assert ab.main(["--questions", q, "--out", str(out), "--resume"], llm=llm, stores=Stores(), desc=DESCRIPTORS) == 0
    data = json.loads(out.read_text())
    assert data["meta"]["complete"] is True and len(data["rows"]) == 6 and data["meta"]["resumed_rows"] == 1


def test_pipeline_run_with_every_question_crashing_stops_after_three(tmp_path, monkeypatch, capsys):
    fed = Fed(fail={"Q1", "Q2", "Q3", "Q4", "Q5"})
    patch_run_one(monkeypatch, fed, exc=TypeError("Could not resolve authentication method."))
    out = tmp_path / "o.json"
    code = run_slice.main(["--questions", write_questions(tmp_path, 6), "--out", str(out)], federation=fed)
    assert code == resume.EXIT_SETUP and fed.calls == ["Q0", "Q1", "Q2", "Q3"]
    assert [r["question_id"] for r in json.loads(out.read_text())["rows"]] == ["Q0"]
    assert "Could not resolve authentication method" in capsys.readouterr().err


def test_two_crashes_in_a_row_are_still_results(tmp_path, monkeypatch):
    fed = Fed(fail={"Q1", "Q2"})
    patch_run_one(monkeypatch, fed, exc=KeyError("bug"))
    out = tmp_path / "o.json"
    assert run_slice.main(["--questions", write_questions(tmp_path, 4), "--out", str(out)], federation=fed) == 0
    assert len(json.loads(out.read_text())["rows"]) == 4
