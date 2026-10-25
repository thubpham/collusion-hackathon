import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sting.game_sting import StingPlayer


class Scripted:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def chat(self, messages, model=None, max_new_tokens=4096):
        self.prompts.append(messages[-1]["content"])
        return self.replies.pop(0), ""


def make_player(hand, replies):
    game = SimpleNamespace(rules_for=lambda player: "rules", cfg=SimpleNamespace(offer_round=2),
                           undercover_name="Quinn")
    player = StingPlayer("Lily", game)
    player.hand = list(hand)
    player.llm_client = Scripted(replies)
    return player


def play_json(cards):
    return '{"played_cards": %s, "behavior": "b", "play_reason": "r"}' % str(cards).replace("'", '"')


def test_overcount_play_is_rejected_without_touching_the_hand():
    player = make_player(["K", "Q", "A", "A", "Joker"], [play_json(["K", "A", "K"]), play_json(["K", "Q"])])
    result, _ = player.choose_cards_to_play("base", "actions", "decision")
    assert result["played_cards"] == ["K", "Q"]
    assert sorted(player.hand) == ["A", "A", "Joker"]
    assert "more copies" in player.llm_client.prompts[1]


def test_non_card_entries_are_rejected():
    player = make_player(["K", "Q", "A", "A", "Joker"], [play_json([1, 2, 3]), play_json(["A", "A"])])
    result, _ = player.choose_cards_to_play("base", "actions", "decision")
    assert result["played_cards"] == ["A", "A"] and sorted(player.hand) == ["Joker", "K", "Q"]


def test_five_bad_answers_fall_back_to_one_card_with_hand_intact():
    player = make_player(["K", "Q"], [play_json(["K", "K"])] * 5)
    result, _ = player.choose_cards_to_play("base", "actions", "decision")
    assert result["played_cards"] == ["K"] and player.hand == ["Q"]
