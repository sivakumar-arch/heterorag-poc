"""
benchmark_v2/probe_determinism.py
=================================
Does the translator give the same reply to the same prompt? Two replays of one plan gave different first replies
for the same step. This tool removes one explanation, a different prompt: for every retrieve step of the plans
stored in a result file it builds the translator prompt ONCE, sends that identical prompt K0 times, and reports
how often the replies differ.

Which sampling is used matters, and it is not the same for every model. The translator passes temperature 0.0 to
its provider, but the provider for a local (Ollama) model is built with model_default_sampling=True, which does
NOT send the temperature: local models sample at their own default (for example 1.0 for gpt-oss:20b, 0.15 for
Mistral Small 3.2). A hosted model gets temperature 0.0 while the API accepts it. Therefore:

    --sampling pipeline   (default) exactly what the pipeline did: model default for an Ollama model
    --sampling zero       send temperature 0.0 to an Ollama model (a hosted model is unchanged)

    python -m benchmark_v2.probe_determinism data/floor_gptoss20b_cost_1.json --repeats 5 --prompts 40 --out data/probe_det_gptoss.json
    python -m benchmark_v2.probe_determinism data/floor_gptoss20b_cost_1.json --repeats 5 --prompts 40 --sampling zero --out data/probe_det_gptoss_zero.json

It needs the translator model of the environment (HETERORAG_TRANSLATOR_MODEL, and HETERORAG_OLLAMA_NUM_CTX and
HETERORAG_OLLAMA_THINK as in the run) and nothing else: no store is queried and no plan is made. For each prompt it records the replies as sent back, after the cleaning the
translator applies, and the verdict of the validator (accepted, rejected, or CANNOT_ANSWER), with the same
semantic-check switch the pipeline uses (HETERORAG_SEMANTIC_CHECKS).

Reading the result: "differ" counts prompts, not calls. The prompts come from questions that share templates,
so the interval is optimistic. A count of zero over a finite sample does not prove that the model is
deterministic; it bounds how often it is not. The pre-registered rule for the number of replays K:
K = 1 if no prompt has two different cleaned replies, otherwise K = 5.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from benchmark_v2.floor_compare import SEED, pct

REFUSAL = ("CANNOT_ANSWER", "-- CANNOT_ANSWER", "// CANNOT_ANSWER")


def retrieve_steps(rows: list[dict]) -> list[tuple[str, object]]:
    """(question id, Step) for every retrieve step of the plans in the rows; rows without a plan are skipped."""
    from heterorag.federation.plan import Op, Plan
    out = []
    for r in rows:
        d = r.get("plan")
        if not d:
            continue
        plan = Plan.from_dict(d if isinstance(d, dict) else {"steps": d})
        out.extend((r["question_id"], s) for s in plan.steps if s.op is Op.RETRIEVE)
    return out


def probe_provider(model: str, sampling: str, client=None):
    """The provider the pipeline built for this model (sampling="pipeline"), or, for an Ollama model, the same
    provider with temperature 0.0 sent (sampling="zero")."""
    from heterorag.federation.llm_config import make_provider, ollama_settings, split_model
    kind, name = split_model(model)
    if kind != "ollama" or sampling == "pipeline":
        return make_provider(model, "translator")
    from heterorag.federation.llm_config import MeteredProvider
    from heterorag.llm_provider import OllamaProvider, RetryingProvider
    cfg = ollama_settings()
    inner = OllamaProvider(name, cfg["host"], num_ctx=cfg["num_ctx"], model_default_sampling=False, strip_think=True,
                           timeout=cfg["timeout"], think=cfg["think"], client=client)
    return MeteredProvider(RetryingProvider(inner), role="translator")


def verdict(step, cleaned: str, checker) -> str:
    """accepted | rejected | cannot_answer, as the translator would decide on a first attempt."""
    if not cleaned:
        return "rejected"
    if cleaned.upper().startswith(REFUSAL):
        return "cannot_answer"
    return "rejected" if checker.check(step, cleaned) else "accepted"


def probe_one(step, provider, descriptors, checker, repeats: int, max_tokens: int) -> dict:
    from heterorag.federation.translator import build_prompt, clean_reply
    prompt = build_prompt(step, descriptors[step.store])
    raw, cleaned, verdicts = [], [], []
    for _ in range(repeats):
        resp = provider.complete(prompt, max_tokens=max_tokens, temperature=0.0)
        raw.append(resp.text)
        c = clean_reply(resp.text)
        cleaned.append(c)
        verdicts.append(verdict(step, c, checker))
    return {"step": step.id, "store": step.store.value, "prompt_chars": len(prompt), "raw": raw, "cleaned": cleaned,
            "verdicts": verdicts}


def tally(probes: list[dict]) -> dict[str, int]:
    return {"prompts": len(probes),
            "differ_raw": sum(len(set(p["raw"])) > 1 for p in probes),
            "differ_cleaned": sum(len(set(p["cleaned"])) > 1 for p in probes),
            "differ_verdict": sum(len(set(p["verdicts"])) > 1 for p in probes)}


def recommended_k(t: dict[str, int]) -> int:
    return 5 if t["differ_cleaned"] else 1


def render(probes: list[dict], repeats: int, sampling: str = "pipeline") -> str:
    t = tally(probes)
    n = t["prompts"]
    how = "with the sampling of the pipeline" if sampling == "pipeline" else "with temperature 0.0 sent"
    lines = [f"{n} prompts, each sent {repeats} times {how} ({n * repeats} calls)"]
    if not n:
        return lines[0]
    lines += [f"  prompts with two or more different replies as sent back   {pct(t['differ_raw'], n)}",
              f"  ... after cleaning (fences and labels removed)            {pct(t['differ_cleaned'], n)}",
              f"  ... with different validator verdicts                     {pct(t['differ_verdict'], n)}",
              "  (prompts come from questions that share templates: read the intervals as optimistic)"]
    by_store: dict[str, list[int]] = {}
    for p in probes:
        c = by_store.setdefault(p["store"], [0, 0])
        c[0] += len(set(p["cleaned"])) > 1
        c[1] += 1
    for s, (k, m) in sorted(by_store.items()):
        lines.append(f"  store {s:6s} cleaned replies differ in {k} of {m} prompts")
    lines.append(f"  pre-registered rule: K = {recommended_k(t)}"
                 + ("  (no prompt differed; this bounds the rate, it does not prove determinism)" if recommended_k(t) == 1 else ""))
    return "\n".join(lines)


def main(argv: list[str] | None = None, provider=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("result_file", help="result file of a run that stored plans (rows[].plan)")
    ap.add_argument("--repeats", type=int, default=5, help="K0: replies per prompt (default 5)")
    ap.add_argument("--prompts", type=int, default=40, help="how many retrieve steps to probe, chosen at random (default 40; 0 = all)")
    ap.add_argument("--sampling", choices=("pipeline", "zero"), default="pipeline",
                    help="pipeline: as the runs were made (an Ollama model at its own default sampling); "
                         "zero: send temperature 0.0 to an Ollama model")
    ap.add_argument("--max-tokens", type=int, default=1500, help="as the translator (default 1500)")
    ap.add_argument("--out", help="write the replies and the tally to this JSON file")
    args = ap.parse_args(argv)
    if args.repeats < 2:
        print("error: --repeats must be at least 2", file=sys.stderr)
        return 2
    try:
        rows = json.loads(Path(args.result_file).read_text())["rows"]
    except (OSError, ValueError, KeyError) as e:
        print(f"error: cannot read {args.result_file}: {e}", file=sys.stderr)
        return 2
    steps = retrieve_steps(rows)
    if not steps:
        print("error: the file holds no stored retrieve steps", file=sys.stderr)
        return 2
    if args.prompts and args.prompts < len(steps):
        steps = random.Random(SEED).sample(steps, args.prompts)

    from heterorag.federation.contract import poc_catalogue
    from heterorag.federation.llm_config import RoleModels, semantic_checks_enabled
    from heterorag.federation.plan import Store
    from heterorag.federation.step_validator import StepValidator
    from heterorag.layer1.poc_descriptors import (CONTENT_SERVICE_DESCRIPTOR, KNOWLEDGE_GRAPH_DESCRIPTOR,
                                                  USER_ACTIVITY_DESCRIPTOR)
    desc = {Store.SQL: USER_ACTIVITY_DESCRIPTOR, Store.GRAPH: KNOWLEDGE_GRAPH_DESCRIPTOR, Store.DOC: CONTENT_SERVICE_DESCRIPTOR}
    checker = StepValidator(poc_catalogue(), semantic_checks=semantic_checks_enabled())
    model = RoleModels.from_env().translator
    if provider is None:
        provider = probe_provider(model, args.sampling)
    from heterorag.federation.llm_config import ollama_settings
    cfg = ollama_settings()
    print(f"translator model: {model}   sampling: {args.sampling}   prompts: {len(steps)}   repeats: {args.repeats}   "
          f"num_ctx: {cfg['num_ctx']}   think: {cfg['think']}", flush=True)

    probes = []
    for i, (qid, step) in enumerate(steps, 1):
        p = probe_one(step, provider, desc, checker, args.repeats, args.max_tokens)
        p["question_id"] = qid
        probes.append(p)
        flag = "  DIFFERS" if len(set(p["cleaned"])) > 1 else ""
        print(f"[{i}/{len(steps)}] {qid} {step.id} {step.store.value}{flag}", flush=True)
    print()
    print(render(probes, args.repeats, args.sampling))
    if args.out:
        Path(args.out).write_text(json.dumps({"translator_model": model, "sampling": args.sampling, "repeats": args.repeats,
                                              "semantic_checks": semantic_checks_enabled(), "num_ctx": cfg["num_ctx"], "think": cfg["think"], "source": args.result_file,
                                              "tally": tally(probes), "probes": probes}, indent=1))
        print(f"replies written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
