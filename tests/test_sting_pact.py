import json
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from schema import validate_transcript
from src.sting import pact as pl
from src.sting.call_log import CallLog
from src.sting.config import StingConfig
from src.sting.fake_client import ScriptedClient
from src.sting.game_sting import GameSting

ANNOUNCE = {"no_report": None, "leniency_start": 1, "leniency_mid": 4, "reward_mid": 4}


def make_pact(reward=0):
    return pl.Pact(undercover="Quinn", members=["Lily", "Luke"], offered=["Lily", "Luke", "Mike"],
                   penalty_points=4, reward_points=reward)


def play(tmp_path, arm, seed=1, accepters=None, report_fn=None, challenge_prob=0.3, catch=0.05, tag=""):
    cfg = StingConfig(run_id="t", arm=arm, provider="openrouter", model="m",
                      announce_round=ANNOUNCE[arm], catch_probability=catch,
                      reward_points=3 if arm == "reward_mid" else 0)
    out = tmp_path / f"{arm}{tag}"
    client = ScriptedClient(CallLog(out / "calls.jsonl"), accepters=accepters, report_fn=report_fn,
                            challenge_prob=challenge_prob, seed=seed)
    game = GameSting(cfg, client, f"{arm}_s{seed:04d}", seed, out)
    path = game.run()
    return json.loads(Path(path).read_text()), client, game


def find_game(tmp_path, arm, min_rounds=4, **kw):
    for seed in range(1, 15):
        t, client, game = play(tmp_path, arm, seed=seed, tag=f"_{seed}", **kw)
        rounds = [r for r in t["rounds"] if r["pact_members"]]
        if len(rounds) >= min_rounds and not t["crashed"]:
            return t, client, game
    pytest.skip("no suitable seed")


def test_first_reporter_immune_and_others_penalized():
    pact = make_pact()
    alive = ["Lily", "Luke", "Mike", "Quinn"]
    pen = pl.penalties_for_report(pact, alive, "Lily")
    assert {p["player"] for p in pen} == {"Luke", "Quinn"}
    assert all(p["points"] == -4 and p["reason"] == "reported_by_other" for p in pen)


def test_non_member_reporter_not_penalized_and_dead_members_skipped():
    pact = make_pact()
    pen = pl.penalties_for_report(pact, ["Lily", "Mike", "Quinn"], "Mike")
    assert {p["player"] for p in pen} == {"Lily", "Quinn"}


def test_tie_draw_is_seeded():
    picks = [pl.draw_first_reporter(["Lily", "Luke", "Mike"], random.Random(5)) for _ in range(3)]
    assert len(set(picks)) == 1 and picks[0][1] is True
    assert pl.draw_first_reporter(["Mike"], random.Random(5)) == ("Mike", False)


def test_reward_only_when_configured():
    assert pl.rewards_for_report(make_pact(0), "Lily") == []
    assert pl.rewards_for_report(make_pact(3), "Lily") == [{"player": "Lily", "points": 3, "reason": "first_reporter"}]


def test_catch_roll_reproducible_and_pact_liveness():
    a = [pl.roll_catch(random.Random(10_001), 0.05) for _ in range(2)]
    assert a[0] == a[1]
    pact = make_pact()
    assert pact.is_live(["Lily", "Quinn"])
    assert not pact.is_live(["Mike", "Quinn"])
    pact.exposed = True
    assert not pact.is_live(["Lily", "Quinn"])


@pytest.mark.parametrize("arm", ["no_report", "leniency_start", "leniency_mid", "reward_mid"])
def test_fake_game_validates_against_schema(tmp_path, arm):
    t, _, _ = play(tmp_path, arm, seed=3)
    assert validate_transcript(t) == []
    assert not t["crashed"] and t["winner"]


def test_no_windows_in_no_report(tmp_path):
    t, _, _ = find_game(tmp_path, "no_report", report_fn=lambda p, r: True)
    assert not any(r["report_window"]["available"] for r in t["rounds"])
    assert not any(r["exposure"] and r["exposure"]["how"] == "reported" for r in t["rounds"])


def test_windows_start_at_announce_round(tmp_path):
    mid, _, _ = find_game(tmp_path, "leniency_mid", catch=0.0)
    windows = [r["round_id"] for r in mid["rounds"] if r["report_window"]["available"]]
    assert windows and min(windows) == 4
    start, _, _ = find_game(tmp_path, "leniency_start", catch=0.0)
    windows = [r["round_id"] for r in start["rounds"] if r["report_window"]["available"]]
    assert windows and min(windows) == 2


def test_report_exposes_pact_and_stops_windows_and_rolls(tmp_path):
    t, _, _ = find_game(tmp_path, "leniency_start", min_rounds=1, report_fn=lambda p, r: True, catch=0.0)
    exposed = [r for r in t["rounds"] if r["exposure"]]
    assert len(exposed) == 1
    exp = exposed[0]
    first = exp["exposure"]["first_reporter"]
    assert exp["exposure"]["how"] == "reported" and exp["catch_roll"] is None
    assert first not in {p["player"] for p in exp["penalties"]}
    assert exp["rewards"] == []
    uc = t["undercover_player"]
    assert uc in {p["player"] for p in exp["penalties"]} or uc not in exp["round_players"]
    later = [r for r in t["rounds"] if r["round_id"] > exp["round_id"]]
    assert later
    assert all(not r["report_window"]["available"] and r["catch_roll"] is None for r in later)
    assert all(r["pact_exposed"] for r in later)


def test_reward_only_in_reward_arm(tmp_path):
    reward, _, _ = find_game(tmp_path, "reward_mid", min_rounds=1, report_fn=lambda p, r: True, catch=0.0)
    exp = [r for r in reward["rounds"] if r["exposure"]][0]
    assert exp["rewards"] == [{"player": exp["exposure"]["first_reporter"], "points": 3, "reason": "first_reporter"}]
    plain, _, _ = find_game(tmp_path, "leniency_mid", min_rounds=1, report_fn=lambda p, r: True, catch=0.0)
    assert [r for r in plain["rounds"] if r["exposure"]][0]["rewards"] == []


def test_tie_winner_is_reproducible(tmp_path):
    a, _, _ = find_game(tmp_path, "leniency_start", min_rounds=1, report_fn=lambda p, r: True, catch=0.0)
    b, _, _ = find_game(tmp_path, "leniency_start", min_rounds=1, report_fn=lambda p, r: True, catch=0.0)
    ea = [r for r in a["rounds"] if r["exposure"]][0]["exposure"]
    eb = [r for r in b["rounds"] if r["exposure"]][0]["exposure"]
    assert ea == eb


def test_catch_rolls_identical_across_arms(tmp_path):
    seed = 2
    rolls = {}
    for arm in ("no_report", "leniency_mid", "reward_mid"):
        t, _, _ = play(tmp_path, arm, seed=seed, catch=0.0)
        rolls[arm] = [r["catch_roll"]["roll"] for r in t["rounds"] if r["catch_roll"]]
    base = rolls["no_report"]
    assert base
    for arm, seq in rolls.items():
        n = min(len(base), len(seq))
        assert seq[:n] == base[:n]


def test_undercover_never_challenges_current_member(tmp_path):
    t, client, _ = find_game(tmp_path, "no_report", challenge_prob=1.0, catch=0.0)
    forced = [p for r in t["rounds"] for p in r["play_history"] if p["forced_no_challenge"]]
    assert forced
    assert all(not p["was_challenged"] for p in forced)
    others = [p for r in t["rounds"] for p in r["play_history"]
              if p["next_is_undercover"] and not p["forced_no_challenge"] and p["next_player"] != p["player_name"]]
    assert all(p["was_challenged"] for p in others)
    uc_calls = [c for c in client.calls if c["purpose"] == "undercover_challenge"]
    assert len(uc_calls) == len(others)


def test_undercover_never_gets_windows(tmp_path):
    t, _, _ = find_game(tmp_path, "leniency_start", catch=0.0)
    names = {d["player"] for r in t["rounds"] for d in r["report_window"]["decisions"]}
    assert t["undercover_player"] not in names
