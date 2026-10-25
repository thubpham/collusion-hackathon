from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts" / "sting"
RETRY_LINE = "Your last reply was not valid JSON. Reply with the JSON object only."


def load(name: str) -> str:
    return (PROMPT_DIR / f"{name}.txt").read_text()


def fill(template: str, **values) -> str:
    names = set(re.findall(r"\{([a-z_]+)\}", template))
    missing = names - set(values)
    if missing:
        raise KeyError(f"unfilled placeholders: {sorted(missing)}")
    for k in names:
        template = template.replace("{" + k + "}", str(values[k]))
    return template


def catch_percent(catch_probability: float) -> int:
    return round(catch_probability * 100)


def _extract_json(raw: str) -> Optional[dict]:
    text = re.sub(r"```(?:json)?", "", raw)
    i, j = text.find("{"), text.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        obj = json.loads(text[i:j + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def parse_offer(raw: str) -> tuple[Optional[dict], bool]:
    obj = _extract_json(raw)
    if obj is None or not isinstance(obj.get("decision"), str):
        return None, False
    decision = obj["decision"].strip().upper()
    reason = obj.get("reason", "")
    if decision not in ("ACCEPT", "REFUSE") or not isinstance(reason, str):
        return None, False
    return {"decision": decision, "reason": reason}, True


def parse_report(raw: str) -> tuple[Optional[dict], bool]:
    obj = _extract_json(raw)
    if obj is None:
        return None, False
    val, reason = obj.get("report"), obj.get("reason", "")
    if isinstance(val, str) and val.strip().lower() in ("true", "false"):
        val = val.strip().lower() == "true"
    if not isinstance(val, bool) or not isinstance(reason, str):
        return None, False
    return {"report": val, "reason": reason}, True


def ask_structured(client, messages: list[dict], parser, max_retries: int, **meta) -> tuple[Optional[dict], bool, int, str]:
    retries = 0
    msgs = messages
    while True:
        raw = client.chat(msgs, retry_index=retries, parser=parser, **meta)
        parsed, ok = parser(raw)
        if ok or retries >= max_retries:
            return parsed, ok, retries, client.last_call_id
        retries += 1
        msgs = messages[:-1] + [{"role": "user", "content": messages[-1]["content"] + "\n" + RETRY_LINE}]
