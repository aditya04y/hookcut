"""Any OpenAI-compatible chat endpoint: Gemini, Groq, OpenAI, OpenRouter, Ollama, or a custom URL
(e.g. a gateway the hiring team provides). Uses the official `openai` SDK with a base_url."""
from __future__ import annotations

import time
from typing import Optional

import openai
from openai import OpenAI

from .base import LLMClient

TPM_WAIT_S = 62  # Groq answers 413 when the rolling minute's token budget is full; the same request fits a minute later


class OpenAICompatClient(LLMClient):
    def __init__(self, provider: str, model: str, *, base_url: Optional[str], api_key: Optional[str],
                 json_mode: bool = True, **kw):
        super().__init__(model, **kw)
        self.provider = provider
        self.json_mode = json_mode
        self.client = OpenAI(base_url=base_url, api_key=api_key or "not-needed", max_retries=6, timeout=300)

    def _call(self, system: str, user: str, json_mode: bool) -> tuple[str, int, int]:
        kwargs = dict(model=self.model, max_tokens=self.max_tokens,
                      messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if json_mode and self.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if self.reasoning:
            kwargs["reasoning_effort"] = self.reasoning
        for attempt in range(4):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                break
            except openai.APIStatusError as e:  # the SDK does not retry 413
                msg = str(e).lower()
                if e.status_code == 413 and ("tpm" in msg or "tokens per minute" in msg) and attempt < 3:
                    time.sleep(TPM_WAIT_S)
                    continue
                if e.status_code == 400 and "failed to generate json" in msg and "response_format" in kwargs:
                    kwargs.pop("response_format")   # provider-side JSON check failed: let our parser handle it
                    continue
                raise
        text = (resp.choices[0].message.content or "") if resp.choices else ""
        u = resp.usage
        return text, (u.prompt_tokens or 0) if u else 0, (u.completion_tokens or 0) if u else 0
