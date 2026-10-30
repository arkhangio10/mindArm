"""Nebius chat completions with one repair attempt and validated safe fallback."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from time import perf_counter
from typing import Any, TypeVar

from openai import APIConnectionError, APIStatusError, OpenAI, OpenAIError
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, ValidationError

from mindarm.config import PlannerOperation
from mindarm.interfaces import MotionPlan, Option, OptionsResponse, Scene, home_plan
from mindarm.planner.fake import safe_options
from mindarm.planner.prompts import OPTIONS_SCHEMA, motion_messages, motion_schema, options_messages

LOGGER = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)
RecordSink = Callable[[dict[str, Any]], None]
DEFAULT_BASE_URL = "https://api.tokenfactory.us-central1.nebius.com/v1/"
DEFAULT_MODEL = "nvidia/nemotron-3-super-120b-a12b"
# Two content attempts (the second one repairs) plus at most one retry after a transient failure.
MAX_CALLS = 3


def is_transient(error: OpenAIError) -> bool:
    """Timeouts, connection errors, and HTTP 5xx; APITimeoutError subclasses APIConnectionError."""
    return isinstance(error, APIConnectionError) or (isinstance(error, APIStatusError) and error.status_code >= 500)


class PlannerUnavailable(RuntimeError):
    """An API/transport failure; the orchestrator must disengage safely."""


def create_client(timeout: float, base_url: str | None = None) -> OpenAI:
    """Build a Token Factory client from NEBIUS_API_KEY."""
    key = os.getenv("NEBIUS_API_KEY", "").strip()
    if not key:
        raise ValueError("Set NEBIUS_API_KEY in .env, or use --planner fake for an offline run")
    # Disable SDK retries so the planner alone bounds the calls per operation (MAX_CALLS).
    return OpenAI(api_key=key, base_url=base_url or os.getenv("NEBIUS_BASE_URL") or DEFAULT_BASE_URL,
                  timeout=timeout, max_retries=0)


class NemotronPlanner:
    """Real text-only planner with schema, scene, and safe-intent validation."""

    def __init__(
        self, client: OpenAI | None = None, *, model: str | None = None,
        timeout: float = 30.0, max_tokens: int = 4096,
        record: RecordSink | None = None,
        propose: PlannerOperation | None = None, plan: PlannerOperation | None = None,
    ) -> None:
        self._client = client or create_client(timeout)
        self._model = model or os.getenv("NEMOTRON_MODEL") or DEFAULT_MODEL
        self._max_tokens = max_tokens
        self._record = record or self._log_record
        self._propose = propose or PlannerOperation(reasoning=False)
        self._plan = plan or PlannerOperation(reasoning=False)

    @staticmethod
    def _log_record(record: dict[str, Any]) -> None:
        LOGGER.info("planner %s", json.dumps(record))

    def _request(
        self, operation: str, options: PlannerOperation, messages: list[ChatCompletionMessageParam],
        schema: dict[str, Any], response_type: type[T], validate: Callable[[T], T] | None = None,
    ) -> T | None:
        response_format: Any = (
            {"type": "json_schema", "json_schema": {"name": operation, "strict": True, "schema": schema}}
            if options.strict_schema else {"type": "json_object"})
        attempt, retried = 1, False
        for call in range(1, MAX_CALLS + 1):
            start = perf_counter()
            usage: dict[str, Any] | None = None
            status = "invalid_output"
            error_text: str | None = None
            raw = ""
            try:
                response = self._client.chat.completions.create(
                    model=self._model, messages=messages, response_format=response_format,
                    temperature=0.1, max_tokens=self._max_tokens,
                    extra_body={"chat_template_kwargs": {"enable_thinking": options.reasoning}},
                )
                usage = response.usage.model_dump() if response.usage is not None else None
                if not response.choices:
                    raise ValueError("Response contained no choices")
                choice = response.choices[0]
                raw = choice.message.content or ""
                if choice.finish_reason != "stop":
                    raise ValueError(f"Incomplete response: finish_reason={choice.finish_reason}")
                output = response_type.model_validate_json(raw)
                if validate is not None:
                    output = validate(output)
                status = "ok"
                return output
            except OpenAIError as error:
                # API exception bodies may contain sensitive upstream data. Log type only.
                error_text = type(error).__name__
                if retried or not is_transient(error):
                    status = "api_error"
                    raise PlannerUnavailable(f"Nemotron request failed ({error_text})") from None
                status = "transient_error"
            except (ValidationError, ValueError) as error:
                error_text = (error.json(include_input=False, include_url=False)
                              if isinstance(error, ValidationError) else str(error))[:2000]
            finally:
                self._record({
                    "type": "planner_call", "operation": operation, "model": self._model,
                    "attempt": attempt, "call": call, "max_calls": MAX_CALLS,
                    "reasoning": options.reasoning, "strict_schema": options.strict_schema,
                    "latency_ms": (perf_counter() - start) * 1000,
                    "usage": usage, "status": status, "error": error_text,
                })
            if status == "transient_error":
                retried = True
            elif attempt == 1:
                attempt = 2
                messages = [*messages,
                            {"role": "assistant", "content": raw[:16000]},
                            {"role": "user", "content": f"Validation failed: {error_text}. Correct it and return JSON only."}]
            else:
                break
        self._record({"type": "planner_fallback", "operation": operation, "reason": "two_invalid_outputs"})
        LOGGER.warning("Nemotron %s: two invalid outputs; using safe fallback", operation)
        return None

    def propose_options(self, scene: Scene) -> list[Option]:
        response = self._request("propose_options", self._propose, options_messages(scene), OPTIONS_SCHEMA,
                                 OptionsResponse)
        return response.options if response is not None else safe_options()

    def plan_motion(self, option: Option, scene: Scene) -> MotionPlan:
        def validate(plan: MotionPlan) -> MotionPlan:
            plan.validate_scene(scene)
            if option.label in {item.label for item in safe_options()} and plan != home_plan():
                raise ValueError("Home and generic resting options require exactly [HOME]")
            return plan

        response = self._request("plan_motion", self._plan, motion_messages(option, scene), motion_schema(scene),
                                 MotionPlan, validate)
        return response if response is not None else home_plan()
