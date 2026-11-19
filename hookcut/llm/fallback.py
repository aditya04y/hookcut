"""Try a list of model specs in order; free tiers rate-limit and time out, so a run should not die on one."""
from __future__ import annotations

from typing import Any, Callable, Optional, Type

from pydantic import BaseModel

from .base import LLMClient, LLMError, LLMResult


class FallbackClient(LLMClient):
    def __init__(self, clients: list[LLMClient]):
        if not clients:
            raise ValueError("FallbackClient needs at least one client")
        self.clients = clients
        first = clients[0]
        super().__init__(first.model, max_tokens=first.max_tokens, temperature=first.temperature,
                         reasoning=first.reasoning, cache=None)
        self.provider = first.provider

    @property
    def spec(self) -> str:
        return " -> ".join(c.spec for c in self.clients)

    def complete_json(self, system: str, user: str, schema: Optional[Type[BaseModel]] = None, *,
                      validate: Optional[Callable[[Any], None]] = None, retries: int = 2,
                      cache_tag: str = "") -> LLMResult:
        errors = []
        for c in self.clients:
            try:
                return c.complete_json(system, user, schema, validate=validate, retries=retries, cache_tag=cache_tag)
            except Exception as e:  # rate limit, timeout, bad JSON after retries, auth: try the next model
                errors.append(f"{c.spec}: {type(e).__name__}: {str(e)[:200]}")
        raise LLMError("all models failed: " + " | ".join(errors))
