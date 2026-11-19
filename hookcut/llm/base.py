"""Provider-neutral LLM interface: every stage asks for JSON and gets validated data back.

Swapping models is a config change ("provider:model"); no stage imports a vendor SDK.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Type

from pydantic import BaseModel, ValidationError

from .cache import DiskCache


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResult:
    data: Any
    text: str
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    cached: bool = False
    attempts: int = 1


def extract_json(text: str) -> Any:
    """Parse JSON from a reply that may be fenced or wrapped in prose."""
    t = (text or "").strip()
    if (fence := re.search(r"```(?:json)?\s*(.*?)```", t, re.S)):
        t = fence.group(1).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError as first_error:
        starts = [i for i in (t.find("{"), t.find("[")) if i != -1]
        if not starts:
            raise
        end = max(t.rfind("}"), t.rfind("]"))
        try:
            return json.loads(t[min(starts): end + 1])
        except json.JSONDecodeError:
            pass
        try:  # last resort for truncated / slightly malformed JSON (unclosed brackets, trailing commas)
            from json_repair import repair_json
            fixed = repair_json(t[min(starts):], return_objects=True)
            if isinstance(fixed, (dict, list)) and fixed:
                return fixed
        except ImportError:
            pass
        raise first_error


class LLMClient:
    """Subclasses implement _call(system, user, json_mode) -> (text, input_tokens, output_tokens)."""

    provider = "base"

    def __init__(self, model: str, *, max_tokens: int = 8000, temperature: Optional[float] = None,
                 reasoning: Optional[str] = None, cache: Optional[DiskCache] = None):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.reasoning = reasoning
        self.cache = cache

    @property
    def spec(self) -> str:
        return f"{self.provider}:{self.model}"

    def _call(self, system: str, user: str, json_mode: bool) -> tuple[str, int, int]:
        raise NotImplementedError

    def complete_json(self, system: str, user: str, schema: Optional[Type[BaseModel]] = None, *,
                      validate: Optional[Callable[[Any], None]] = None, retries: int = 2,
                      cache_tag: str = "") -> LLMResult:
        """Ask for JSON; check it parses, matches `schema` and passes `validate` (raise ValueError);
        on failure, re-ask with the error appended. Cached by (model, prompts, settings)."""
        key = None
        if self.cache is not None:
            key = self.cache.key(self.spec, system, user, self.max_tokens, self.temperature, self.reasoning, cache_tag)
            hit = self.cache.get(key)
            if hit is not None:
                return LLMResult(data=hit["data"], text=hit["text"], provider=self.provider, model=self.model,
                                 input_tokens=hit.get("input_tokens", 0), output_tokens=hit.get("output_tokens", 0),
                                 latency_s=0.0, cached=True, attempts=0)
        prompt, last_err, tin, tout = user, "", 0, 0
        t0 = time.time()
        for attempt in range(1, retries + 2):
            text, i, o = self._call(system, prompt, json_mode=True)
            tin, tout = tin + i, tout + o
            try:
                data = extract_json(text)
                if schema is not None:
                    schema.model_validate(data)
                if validate is not None:
                    validate(data)
                result = LLMResult(data=data, text=text, provider=self.provider, model=self.model,
                                   input_tokens=tin, output_tokens=tout, latency_s=round(time.time() - t0, 2),
                                   attempts=attempt)
                if key is not None:
                    self.cache.put(key, {"data": data, "text": text, "input_tokens": tin, "output_tokens": tout})
                return result
            except (json.JSONDecodeError, ValidationError, ValueError) as e:
                last_err = str(e)[:1500]
                prompt = (f"{user}\n\nYOUR PREVIOUS REPLY WAS REJECTED: {last_err}\n"
                          "Return the COMPLETE corrected JSON object (every field, not just the fixed ones), "
                          "and nothing else.")
        raise LLMError(f"{self.spec}: no valid JSON after {retries + 1} attempts: {last_err}")
