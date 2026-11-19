"""Build an LLM client from a spec like "gemini:gemini-3.8-flash" or "groq:openai/gpt-oss-120b".

Secrets come from the process environment, then from the file named by HOOKCUT_ENV_FILE, then ./.env.
Only the variables listed here are ever read.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from .base import LLMClient
from .cache import DiskCache

PROVIDERS: dict[str, dict] = {
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/", "key_env": "GEMINI_API_KEY"},
    "groq": {"base_url": "https://api.groq.com/openai/v1", "key_env": "GROQ_API_KEY"},
    "openai": {"base_url": None, "key_env": "OPENAI_API_KEY"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "key_env": "OPENROUTER_API_KEY"},
    "ollama": {"base_url": "http://localhost:11434/v1", "key_env": None},
    "custom": {"base_url_env": "HOOKCUT_BASE_URL", "key_env": "HOOKCUT_API_KEY"},  # any OpenAI-compatible gateway
    "anthropic": {"sdk": "anthropic", "key_env": "ANTHROPIC_API_KEY"},
}
ALLOWED_VARS = sorted({p["key_env"] for p in PROVIDERS.values() if p.get("key_env")} | {"HOOKCUT_BASE_URL"})


def load_secrets(env_file: Optional[str] = None) -> dict[str, str]:
    secrets: dict[str, str] = {}
    files = [f for f in (env_file or os.environ.get("HOOKCUT_ENV_FILE"), ".env") if f]
    for f in files:
        if Path(f).is_file():
            from dotenv import dotenv_values
            vals = dotenv_values(f)
            for k in ALLOWED_VARS:
                if vals.get(k) and k not in secrets:
                    secrets[k] = vals[k]
    for k in ALLOWED_VARS:   # real environment wins
        if os.environ.get(k):
            secrets[k] = os.environ[k]
    return secrets


def parse_spec(spec: str) -> tuple[str, str]:
    if ":" not in spec:
        raise ValueError(f"model spec must look like provider:model, got {spec!r}")
    provider, model = spec.split(":", 1)
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}; choose from {sorted(PROVIDERS)}")
    return provider, model


def make_client(spec: str, *, secrets: Optional[dict] = None, cache: Optional[DiskCache] = None,
                **options) -> LLMClient:
    """options: max_tokens, temperature, reasoning (effort for Anthropic / reasoning_effort elsewhere)."""
    secrets = secrets if secrets is not None else load_secrets()
    provider, model = parse_spec(spec)
    p = PROVIDERS[provider]
    key = secrets.get(p["key_env"]) if p.get("key_env") else None
    if p.get("sdk") == "anthropic":
        from .anthropic_provider import AnthropicClient
        return AnthropicClient(model, api_key=key, cache=cache, **options)
    from .openai_compat import OpenAICompatClient
    base_url = secrets.get(p["base_url_env"]) if p.get("base_url_env") else p["base_url"]
    if p.get("key_env") and not key and provider != "custom":
        raise RuntimeError(f"{p['key_env']} is not set (env var, HOOKCUT_ENV_FILE, or .env)")
    return OpenAICompatClient(provider, model, base_url=base_url, api_key=key, cache=cache, **options)
