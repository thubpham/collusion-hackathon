import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from src.sting import prompts as P
import pilot_offer_acceptance as pilot


def test_fill_leaves_json_braces_and_requires_values():
    out = P.fill(P.load("offer_direct"), undercover_name="Quinn", catch_percent=5, penalty_points=4)
    assert "Quinn" in out and "5%" in out and '{"reason"' in out and "{undercover_name}" not in out
    with pytest.raises(KeyError):
        P.fill(P.load("offer_direct"), undercover_name="Quinn")


def test_catch_percent():
    assert [P.catch_percent(x) for x in (0.03, 0.05, 0.10)] == [3, 5, 10]


def test_parse_offer():
    ok = lambda s: P.parse_offer(s)[1]
    assert P.parse_offer('```json\n{"reason": "x", "decision": " accept "}\n```') == ({"decision": "ACCEPT", "reason": "x"}, True)
    assert P.parse_offer('Sure! {"decision": "REFUSE", "reason": "no"} bye')[0]["decision"] == "REFUSE"
    assert P.parse_offer('{"decision": "ACCEPT"}')[0]["reason"] == ""
    assert not ok("no json") and not ok('{"decision": "MAYBE", "reason": ""}') and not ok('{"decision": true}')


def test_parse_report():
    assert P.parse_report('{"reason": "r", "report": "True"}') == ({"report": True, "reason": "r"}, True)
    assert P.parse_report('{"report": false}')[0]["report"] is False
    assert not P.parse_report('{"report": "yes"}')[1]


class Fake:
    last_call_id = "c_x"

    def __init__(self, replies):
        self.replies, self.seen = list(replies), []

    def chat(self, msgs, **kw):
        self.seen.append((msgs, kw["retry_index"]))
        return self.replies.pop(0)


def test_ask_structured_retries_then_gives_up():
    c = Fake(["junk", '{"decision": "ACCEPT", "reason": ""}'])
    parsed, ok, retries, _ = P.ask_structured(c, [{"role": "user", "content": "q"}], P.parse_offer, 3, game_id="g", round=1, player="p", purpose="pilot")
    assert ok and retries == 1 and c.seen[1][0][-1]["content"].endswith(P.RETRY_LINE) and c.seen[1][1] == 1
    c = Fake(["junk"] * 4)
    parsed, ok, retries, _ = P.ask_structured(c, [{"role": "user", "content": "q"}], P.parse_offer, 3, game_id="g", round=1, player="p", purpose="pilot")
    assert not ok and retries == 3 and len(c.seen) == 4


def test_pilot_hand_is_standardized_and_seeded():
    for seed in range(1, 31):
        h = pilot.make_hand(seed)
        assert h == pilot.make_hand(seed) and sorted(h).count("K") == 2 and h.count("Joker") == 1 and len(h) == 5
        assert all(c in ("Q", "A") for c in h if c not in ("K", "Joker"))


def test_pilot_messages_have_no_unfilled_placeholders_and_undercover_differs():
    for seed in (1, 2, 3):
        msgs, me, uc = pilot.build_messages(seed, 0.05)
        text = msgs[1]["content"]
        assert me != uc and f"You are {me}" in text and f"from {uc}" in text and "5%" in text
        assert "{player_name}" not in text and "{hand}" not in text


def test_prompt_files_have_no_model_names():
    for f in P.PROMPT_DIR.glob("*.txt"):
        assert not any(w in f.read_text().lower() for w in ("llama", "qwen", "claude", "gpt", "haiku", "mistral"))


def test_deal_scope_sentence_in_offer_and_notes():
    sentence = "The deal is only between you and"
    for name in ("offer_direct", "private_notes"):
        assert sentence in P.load(name)


def test_only_direct_offer_type_is_accepted():
    from src.sting.config import StingConfig
    with pytest.raises(ValueError):
        StingConfig(run_id="t", arm="no_report", provider="openrouter", model="m", offer_type="hint")
