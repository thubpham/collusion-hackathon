import json
import random
import re
from datetime import datetime, timezone


class ScriptedClient:
    def __init__(self, call_log, accepters=None, report_fn=None, challenge_prob=0.3, seed=0):
        self.call_log = call_log
        self.accepters = accepters
        self.report_fn = report_fn or (lambda player, round_id: False)
        self.challenge_prob = challenge_prob
        self.rng = random.Random(seed)
        self.last_call_id = None
        self.calls = []

    def reply(self, messages, round_id, player, purpose):
        if purpose in ("play", "undercover_play"):
            line = re.search(r"Your current hand is: (.*)", messages[-1]["content"]).group(1)
            card = [c.strip() for c in line.split(",")][0]
            return json.dumps({"played_cards": [card], "behavior": "calm", "play_reason": "scripted"})
        if purpose in ("challenge", "undercover_challenge"):
            challenge = self.rng.random() < self.challenge_prob
            return json.dumps({"was_challenged": challenge, "challenge_reason": "scripted"})
        if purpose == "offer_response":
            accept = self.accepters is None or player in self.accepters
            return json.dumps({"reason": "scripted", "decision": "ACCEPT" if accept else "REFUSE"})
        if purpose == "report_window":
            return json.dumps({"reason": "scripted", "report": bool(self.report_fn(player, round_id))})
        return "Nothing notable."

    def chat(self, messages, *, game_id, round, player, purpose, retry_index=0, parser=None):
        text = self.reply(messages, round, player, purpose)
        parsed, ok = parser(text) if parser else (None, True)
        self.calls.append({"round": round, "player": player, "purpose": purpose, "messages": messages})
        self.last_call_id = self.call_log.append(
            game_id=game_id, round=round, player=player, purpose=purpose, provider="fake",
            model="fake", temperature=None, max_tokens=None, messages=messages,
            raw_response=text, parsed=parsed, parse_ok=ok, retry_index=retry_index, error=None,
            tokens_in=0, tokens_out=0, latency_s=0.0,
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        return text
