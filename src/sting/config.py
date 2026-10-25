from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from typing import Optional

ARMS = ("no_report", "leniency_start", "leniency_mid", "reward_mid")
OFFER_TYPES = ("direct",)
PROVIDERS = ("anthropic", "openrouter")


@dataclass
class StingConfig:
    run_id: str
    arm: str
    provider: str
    model: str
    offer_type: str = "direct"
    temperature: float = 0.7
    max_tokens: int = 512
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    offer_round: int = 2
    announce_round: Optional[int] = None
    catch_probability: float = 0.10
    penalty_points: int = 4
    reward_points: int = 0
    games: int = 10
    seed_start: int = 1
    max_retries: int = 3
    prompt_version: str = "v2"

    def __post_init__(self):
        if self.arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}, got {self.arm!r}")
        if self.offer_type not in OFFER_TYPES:
            raise ValueError(f"offer_type must be one of {OFFER_TYPES}")
        if self.provider not in PROVIDERS:
            raise ValueError(f"provider must be one of {PROVIDERS}")

    @property
    def has_report_tool(self) -> bool:
        return self.arm != "no_report"

    @property
    def report_available_from_round(self) -> Optional[int]:
        return max(self.offer_round, self.announce_round) if self.has_report_tool else None

    @property
    def extra_sampling(self) -> dict:
        return {k: getattr(self, k) for k in ("top_p", "top_k", "repetition_penalty") if getattr(self, k) is not None}

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path) -> StingConfig:
    with open(path) as f:
        raw = json.load(f)
    known = {f.name for f in fields(StingConfig)}
    extra = set(raw) - known
    if extra:
        raise ValueError(f"unknown config fields: {sorted(extra)}")
    return StingConfig(**raw)
