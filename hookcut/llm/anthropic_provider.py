"""Claude through the official Anthropic SDK (not an OpenAI-compatible shim).

Credentials resolve the SDK's usual way (ANTHROPIC_API_KEY, or an `ant auth login` profile).
Opus 5.5 always thinks adaptively; depth is set with output_config.effort. Server-side refusal
fallbacks are enabled for the models that support them.
"""
from __future__ import annotations

from typing import Optional

import anthropic

from .base import LLMClient, LLMError

FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5", "claude-fable-5-2", "claude-fable-5-1"}


class AnthropicClient(LLMClient):
    provider = "anthropic"

    def __init__(self, model: str = "claude-opus-5-5", *, api_key: Optional[str] = None,
                 effort: str = "medium", **kw):
        kw.setdefault("max_tokens", 16000)
        super().__init__(model, **kw)
        self.effort = kw.get("reasoning") or effort
        self.client = anthropic.Anthropic(api_key=api_key, max_retries=4) if api_key else anthropic.Anthropic(max_retries=4)

    def _call(self, system: str, user: str, json_mode: bool) -> tuple[str, int, int]:
        params = dict(model=self.model, max_tokens=self.max_tokens, system=system,
                      messages=[{"role": "user", "content": user}],
                      output_config={"effort": self.effort})
        if self.model in FALLBACK_MODELS:
            resp = self.client.beta.messages.create(**params, betas=["server-side-fallback-2026-07-01"],
                                                    fallbacks="default")
        else:
            resp = self.client.messages.create(**params)
        if resp.stop_reason == "refusal":
            cat = getattr(resp.stop_details, "category", None) if resp.stop_details else None
            raise LLMError(f"anthropic:{self.model} refused (category={cat})")
        text = "".join(b.text for b in resp.content if b.type == "text")
        return text, resp.usage.input_tokens, resp.usage.output_tokens
