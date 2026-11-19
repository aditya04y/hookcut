"""Task-specific ranked model choices and explanations."""
from __future__ import annotations

from pathlib import Path

import yaml

from .quota import QuotaLedger

POLICY = Path(__file__).with_name("routing.yaml")
TIERS = {
    "balanced": ("balanced", "quality", "budget"),
    "quality": ("quality", "balanced", "budget"),
    "budget": ("budget", "balanced", "quality"),
}


class Router:
    def __init__(self, ledger: QuotaLedger | None = None, *, policy_path: str | Path = POLICY):
        self.ledger = ledger if ledger is not None else QuotaLedger()
        self.policy = yaml.safe_load(Path(policy_path).read_text(encoding="utf-8"))
        self._families = {model["spec"]: model["family"] for models in self.policy.values() for model in models}

    @property
    def tasks(self) -> tuple[str, ...]:
        return tuple(self.policy)

    def family(self, spec: str) -> str:
        """Resolve policy models, including aliases served by another gateway."""
        if spec in self._families:
            return self._families[spec]
        provider, _, model = spec.partition(":")
        model = model.rsplit("/", 1)[-1].lower()
        for prefix, family in (("gemini-", "gemini"), ("gpt-oss-", "gpt-oss"),
                               ("qwen", "qwen"), ("claude-", "claude")):
            if model.startswith(prefix):
                return family
        return provider.lower()

    def decisions(self, task: str, *, quality: str = "balanced", exclude_families=(),
                  max_cost: float | None = None) -> list[dict]:
        """Explain every choice. max_cost caps the USD price of 1M input + 1M output tokens."""
        if task not in self.policy:
            raise ValueError(f"unknown model task {task!r}; choose from {', '.join(self.tasks)}")
        if quality not in TIERS:
            raise ValueError(f"unknown quality {quality!r}; choose from {', '.join(TIERS)}")
        if max_cost is not None and max_cost < 0:
            raise ValueError("max_cost must be non-negative")
        excluded = {family.casefold() for family in exclude_families}
        exhausted = self.ledger.status()
        preference = TIERS[quality]
        models = sorted(self.policy[task], key=lambda model: preference.index(model["tier"]))
        choices = []
        for model in models:
            spec, tier, family = model["spec"], model["tier"], model["family"]
            price = model["price_in"] + model["price_out"]
            if family.casefold() in excluded:
                reason = f"Excluded family ({family})."
            elif spec in exhausted:
                reason = f"Daily quota resets at {exhausted[spec]['reset_at']}."
            elif max_cost is not None and price > max_cost:
                reason = f"${price:g} per 1M input + 1M output tokens exceeds the ${max_cost:g} cap."
            else:
                choices.append({**model, "eligible": True, "reason": f"{tier.title()} tier. {model['fit']}"})
                continue
            choices.append({**model, "eligible": False, "reason": reason})
        return choices

    def choose(self, task: str, *, quality: str = "balanced", exclude_families=(),
               max_cost: float | None = None) -> list[str]:
        return [row["spec"] for row in self.decisions(task, quality=quality,
                                                      exclude_families=exclude_families, max_cost=max_cost)
                if row["eligible"]]
