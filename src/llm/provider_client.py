from __future__ import annotations

import builtins
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from dotenv import load_dotenv

PURPOSES = {
    "play", "challenge", "reflect", "offer_response", "report_window",
    "undercover_play", "undercover_challenge", "pilot",
}
MAX_TRIES = 5

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def split_messages(messages: list[dict]) -> tuple[str, list[dict]]:
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    rest = [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"]
    return system, rest


class ProviderClient:
    def __init__(self, provider: str, model: str, temperature: float, max_tokens: int, call_log: "CallLog",
                 extra_sampling: Optional[dict] = None):
        if provider not in ("anthropic", "openrouter"):
            raise ValueError(f"unknown provider {provider!r}")
        self.provider, self.model = provider, model
        self.temperature, self.max_tokens = temperature, max_tokens
        self.call_log = call_log
        self.extra_sampling = {k: v for k, v in (extra_sampling or {}).items() if v is not None}
        self.last_call_id: Optional[str] = None
        if provider == "anthropic":
            import anthropic
            self._sdk = anthropic
            self._client = anthropic.Anthropic()
            self._fatal = (anthropic.AuthenticationError, anthropic.NotFoundError,
                           anthropic.BadRequestError, anthropic.PermissionDeniedError)
        else:
            import openai
            self._client = openai.OpenAI(base_url="https://openrouter.ai/api/v1",
                                         api_key=os.environ["OPENROUTER_API_KEY"])
            self._fatal = (openai.AuthenticationError, openai.NotFoundError,
                           openai.BadRequestError, openai.PermissionDeniedError)

    def _call(self, messages: list[dict]) -> tuple[str, int, int]:
        if self.provider == "anthropic":
            system, rest = split_messages(messages)
            kwargs = dict(model=self.model, messages=rest, max_tokens=self.max_tokens)
            if system:
                kwargs["system"] = system
            resp = self._client.messages.create(**kwargs)
            text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
            return text, resp.usage.input_tokens, resp.usage.output_tokens
        resp = self._client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=self.max_tokens, temperature=self.temperature,
            **({"extra_body": self.extra_sampling} if self.extra_sampling else {}))
        usage = resp.usage
        return (resp.choices[0].message.content or "",
                getattr(usage, "prompt_tokens", None), getattr(usage, "completion_tokens", None))

    def chat(self, messages: list[dict], *, game_id: str, round: int, player: str, purpose: str,
             retry_index: int = 0,
             parser: Optional[Callable[[str], tuple[Optional[dict], bool]]] = None) -> str:
        if purpose not in PURPOSES:
            raise ValueError(f"bad purpose {purpose!r}")
        for attempt in range(MAX_TRIES):
            t0 = time.time()
            text, tin, tout, error = "", None, None, None
            try:
                text, tin, tout = self._call(messages)
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
                fatal = isinstance(e, self._fatal + (TypeError, ValueError))
            parsed, ok = (parser(text) if (parser and error is None) else (None, error is None))
            self.last_call_id = self.call_log.append(
                game_id=game_id, round=round, player=player, purpose=purpose, provider=self.provider,
                model=self.model, temperature=None if self.provider == "anthropic" else self.temperature,
                max_tokens=self.max_tokens, messages=messages, raw_response=text, parsed=parsed, parse_ok=ok,
                retry_index=retry_index, error=error, tokens_in=tin, tokens_out=tout,
                latency_s=builtins.round(time.time() - t0, 2),
                timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
            if error is None:
                return text
            if fatal or attempt == MAX_TRIES - 1:
                raise RuntimeError(error)
            time.sleep(2 ** attempt)

