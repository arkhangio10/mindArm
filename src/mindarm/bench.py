"""Live planner benchmark: python -m mindarm.bench --n 10. It calls Token Factory; pytest never runs it."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from mindarm.config import PlannerOperation, default_config_path, load_settings
from mindarm.interfaces import MotionPlan, Option, Scene, home_plan
from mindarm.perception.fake import FakePerception
from mindarm.planner.fake import FakePlanner
from mindarm.planner.nemotron import DEFAULT_MODEL, NemotronPlanner, PlannerUnavailable, create_client
from mindarm.session_log import SessionLog

Prices = dict[str, float]


def model_prices(client: OpenAI, model: str) -> Prices | None:
    """Read USD-per-token prices from Token Factory's verbose model list, or None if unlisted."""
    for item in client.models.list(extra_query={"verbose": "true"}):
        pricing = (item.model_extra or {}).get("pricing")
        if item.id == model and pricing:
            return {key: float(pricing[key]) for key in ("prompt", "input_cache_read", "completion")}
    return None


def rejection_reason(error: str) -> str:
    """Condense a logged validation error into a countable reason, with list indexes as [i]."""
    try:
        details = json.loads(error)
    except ValueError:
        return error
    return "; ".join(dict.fromkeys(
        (".".join("[i]" if isinstance(part, int) else str(part) for part in item["loc"]) or "response")
        + ": " + item["msg"] for item in details))


def usage_totals(calls: list[dict[str, Any]]) -> dict[str, int]:
    """Sum prompt, cached, completion, and reasoning tokens over planner_call records."""
    totals = Counter({"prompt": 0, "cached": 0, "completion": 0, "reasoning": 0})
    for call in calls:
        usage = call["usage"] or {}
        totals["prompt"] += usage.get("prompt_tokens") or 0
        totals["completion"] += usage.get("completion_tokens") or 0
        totals["cached"] += (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        totals["reasoning"] += (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
    return dict(totals)


def cost(tokens: dict[str, int], prices: Prices | None) -> float | None:
    """Price uncached input, cached input, and completion tokens separately."""
    if prices is None:
        return None
    return ((tokens["prompt"] - tokens["cached"]) * prices["prompt"]
            + tokens["cached"] * prices["input_cache_read"] + tokens["completion"] * prices["completion"])


def bench_operation(
    operation: str, call: Callable[[int], Any], expected: Callable[[int, Any], bool], n: int,
    records: list[dict[str, Any]], prices: Prices | None,
) -> dict[str, Any]:
    """Run one planner operation n times and classify each run from its planner records."""
    latencies: list[float] = []
    outcome: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    calls: list[dict[str, Any]] = []
    outputs: list[Any] = []
    for index in range(n):
        first = len(records)
        start = perf_counter()
        try:
            result = call(index)
            outcome["expected"] += expected(index, result)
            outputs.append(result.model_dump(mode="json") if isinstance(result, MotionPlan)
                           else [option.model_dump(mode="json") for option in result])
        except PlannerUnavailable:
            outcome["api_error"] += 1
            outputs.append(None)
        latencies.append(perf_counter() - start)
        run = [record for record in records[first:] if record.get("operation") == operation]
        run_calls = [record for record in run if record["type"] == "planner_call"]
        calls.extend(run_calls)
        ok = [record["attempt"] for record in run_calls if record["status"] == "ok"]
        outcome["first_try"] += ok == [1]
        outcome["repaired"] += ok == [2]
        outcome["fallback"] += any(record["type"] == "planner_fallback" for record in run)
        reasons.update(rejection_reason(record["error"]) for record in run_calls if record["status"] == "invalid_output")
    tokens = usage_totals(calls)
    return {
        "operation": operation, "n": n, "calls": len(calls),
        "p50_s": float(np.percentile(latencies, 50)), "p95_s": float(np.percentile(latencies, 95)),
        "min_s": min(latencies), "max_s": max(latencies),
        **{key: outcome[key] for key in ("first_try", "repaired", "fallback", "api_error", "expected")},
        "tokens": tokens, "cost_usd": cost(tokens, prices), "rejections": dict(reasons), "outputs": outputs,
    }


def run(planner: NemotronPlanner, scene: Scene, n: int, records: list[dict[str, Any]],
        prices: Prices | None) -> list[dict[str, Any]]:
    """Benchmark n proposals, then n plans for those proposals, as the live loop plans them.

    Expected means every scene object is offered (proposals). For plans, an option naming one
    scene object must plan that object's five-step handover, and any other option exactly [HOME].
    """
    reference = FakePlanner()
    labels = [obj.label for obj in scene.objects]

    def offers_all(_: int, options: list[Option]) -> bool:
        return all(any(label.casefold() in option.label.casefold() for option in options) for label in labels)

    propose = bench_operation("propose_options", lambda _: planner.propose_options(scene), offers_all, n,
                              records, prices)
    proposals = ([[Option.model_validate(item) for item in output] for output in propose["outputs"] if output]
                 or [reference.propose_options(scene)])

    def goal(index: int) -> Option:
        options = proposals[index % len(proposals)]
        return options[index % len(options)]

    def matches(index: int, plan: MotionPlan) -> bool:
        named = [label for label in labels if label.casefold() in goal(index).label.casefold()]
        if len(named) != 1:
            return plan == home_plan()
        return plan == reference.plan_motion(Option(label=f"Pick up {named[0]}", description="reference"), scene)

    return [propose, bench_operation("plan_motion", lambda i: planner.plan_motion(goal(i), scene), matches,
                                     n, records, prices)]


def report(model: str, operations: dict[str, PlannerOperation], summaries: list[dict[str, Any]]) -> str:
    """Render a Markdown table plus rejection reasons."""
    lines = [f"Model: {model}", "",
             "| Operation | Reasoning | Format | N | p50 | p95 | Range | First try | Repaired | Fallback | API errors "
             "| Expected | Calls | Prompt (cached) / completion (reasoning) | Cost |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for item, options in zip(summaries, operations.values()):
        tokens = item["tokens"]
        price = "n/a" if item["cost_usd"] is None else f"${item['cost_usd']:.4f}"
        lines.append(
            f"| {item['operation']} | {'on' if options.reasoning else 'off'} "
            f"| {'json_schema' if options.strict_schema else 'json_object'} "
            f"| {item['n']} | {item['p50_s']:.2f} s | {item['p95_s']:.2f} s "
            f"| {item['min_s']:.2f}-{item['max_s']:.2f} s | {item['first_try']}/{item['n']} | {item['repaired']} "
            f"| {item['fallback']} | {item['api_error']} | {item['expected']}/{item['n']} | {item['calls']} "
            f"| {tokens['prompt']} ({tokens['cached']}) / "
            f"{tokens['completion']} ({tokens['reasoning']}) | {price} |")
    for item in summaries:
        lines += ["", f"Rejections in {item['operation']}:"]
        lines += [f"- {count}x {reason}" for reason, count in item["rejections"].items()] or ["- none"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Benchmark the live planner on the fixture scene and log every call."""
    parser = argparse.ArgumentParser(description="Benchmark the live Nemotron planner (calls Token Factory)")
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--n", type=int, default=10, help="Runs per operation")
    parser.add_argument("--model", help="Model for both operations (default: NEMOTRON_MODEL or Super)")
    for operation in ("propose", "plan"):
        parser.add_argument(f"--{operation}-reasoning", choices=("on", "off"), help="Override the configured setting")
        parser.add_argument(f"--{operation}-format", choices=("strict", "object"),
                            help="strict = json_schema, object = json_object (default: configured)")
    args = parser.parse_args(argv)
    settings = load_settings(args.config)
    load_dotenv(args.config.resolve().parent.parent / ".env", override=False)
    model = args.model or os.getenv("NEMOTRON_MODEL") or DEFAULT_MODEL
    operations = {}
    for operation, configured in (("propose", settings.planner_propose), ("plan", settings.planner_plan)):
        reasoning, response_format = getattr(args, f"{operation}_reasoning"), getattr(args, f"{operation}_format")
        operations[operation] = PlannerOperation(
            reasoning=configured.reasoning if reasoning is None else reasoning == "on",
            strict_schema=configured.strict_schema if response_format is None else response_format == "strict")
    client = create_client(settings.planner_timeout)
    log = SessionLog(settings.logs_dir)
    records: list[dict[str, Any]] = []

    def record(item: dict[str, Any]) -> None:
        records.append(item)
        log.write(item)

    log.write({"type": "bench_configuration", "model": model, "n": args.n, "max_tokens": settings.planner_max_tokens,
               **{name: value.model_dump() for name, value in operations.items()}})
    planner = NemotronPlanner(client, model=model, max_tokens=settings.planner_max_tokens, record=record,
                              **operations)
    summaries = run(planner, FakePerception(settings.scene_path).detect(), args.n, records, model_prices(client, model))
    log.write({"type": "bench_summary", "model": model, "summaries": summaries})
    print(report(model, operations, summaries))
    print(f"\nCalls logged to {log.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
