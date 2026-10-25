import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm.provider_client import ProviderClient, split_messages
from src.sting.call_log import KEYS, CallLog

MSGS = [{"role": "system", "content": "A"}, {"role": "user", "content": "hi"},
        {"role": "system", "content": "B"}, {"role": "assistant", "content": "yo"},
        {"role": "user", "content": "again"}]


def test_split_messages():
    system, rest = split_messages(MSGS)
    assert system == "A\n\nB"
    assert [m["role"] for m in rest] == ["user", "assistant", "user"]


def test_mocked_call_writes_one_valid_line(tmp_path):
    log = CallLog(tmp_path / "calls.jsonl")
    client = ProviderClient("anthropic", "claude-haiku-5-5", 0.7, 512, log)
    client._client = MagicMock()
    client._client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="O"), SimpleNamespace(type="text", text="K")],
        usage=SimpleNamespace(input_tokens=5, output_tokens=2))
    out = client.chat(MSGS, game_id="g", round=1, player="Lily", purpose="pilot",
                      parser=lambda t: ({"x": t}, True))
    assert out == "OK"
    kwargs = client._client.messages.create.call_args.kwargs
    assert kwargs["system"] == "A\n\nB" and "temperature" not in kwargs and kwargs["max_tokens"] == 512
    lines = (tmp_path / "calls.jsonl").read_text().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert list(rec) == KEYS
    assert rec["temperature"] is None and rec["call_id"] == "c_000001" and rec["parsed"] == {"x": "OK"} and rec["tokens_in"] == 5


def test_openrouter_extra_sampling_goes_in_extra_body(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    log = CallLog(tmp_path / "c.jsonl")
    c = ProviderClient("openrouter", "m", 0.8, 256, log,
                       extra_sampling={"top_p": 0.95, "top_k": 40, "repetition_penalty": 1.1, "x": None})
    c._client = MagicMock()
    c._client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="OK"))],
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=1))
    c.chat([{"role": "user", "content": "hi"}], game_id="g", round=1, player="p", purpose="pilot")
    kw = c._client.chat.completions.create.call_args.kwargs
    assert kw["extra_body"] == {"top_p": 0.95, "top_k": 40, "repetition_penalty": 1.1}
    assert kw["temperature"] == 0.8 and kw["max_tokens"] == 256
