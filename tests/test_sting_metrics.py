import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import sting_metrics as M

UC = "Quinn"


def play(player, nxt, challenged=False):
    return {"player_name": player, "next_player": nxt, "was_challenged": challenged,
            "player_is_undercover": player == UC}


def window(rid, available, decisions=(), tie=False):
    return {"round": rid, "available": available, "decisions": list(decisions),
            "reporters": [d["player"] for d in decisions if d["report"]],
            "first_reporter": None, "tie_broken_by_draw": tie}


def decision(player, report, member):
    return {"player": player, "is_member": member, "report": report, "reason": "",
            "parse_ok": True, "call_id": "c"}


def rnd(rid, plays, members, win=None, exposure=None, penalties=(), result=None):
    return {"round_id": rid, "play_history": plays, "pact_members": members,
            "report_window": win or window(rid, False), "exposure": exposure,
            "penalties": list(penalties), "rewards": [], "round_result": result}


def offer(to, response):
    return {"to": to, "response": response, "parse_ok": True}


def transcript():
    members = ["Lily", "Mike"]
    exposure = {"round": 5, "how": "reported", "first_reporter": "Mike"}
    rounds = [
        rnd(1, [play(UC, "Lily", True)], []),
        rnd(2, [play(UC, "Lily"), play("Lily", "Mike"), play("Mike", "Luke", True)], members),
        rnd(3, [play("Luke", UC), play(UC, "Mike"), play("Mike", "Lily")], members),
        rnd(4, [play(UC, "Mike", True)], members,
            window(4, True, [decision("Lily", False, True), decision("Mike", False, True),
                             decision("Luke", False, False)])),
        rnd(5, [play(UC, "Lily"), play("Mike", "Lily")], members,
            window(5, True, [decision("Lily", True, True), decision("Mike", True, True),
                             decision("Luke", False, False)], tie=True),
            exposure,
            [{"player": "Lily", "points": -4, "reason": "reported_by_other"},
             {"player": UC, "points": -4, "reason": "reported_by_other"}]),
        rnd(6, [play("Mike", "Lily", True)], [], result={"shooter_name": "Luke", "bullet_hit": True}),
        rnd(7, [play("Mike", "Lily")], [], result={"shooter_name": "Mike", "bullet_hit": False}),
    ]
    return {
        "game_id": "g", "crashed": False,
        "config": {"run_id": "r", "seed": 4, "arm": "leniency_mid", "offer_type": "direct",
                   "model": "m", "offer_round": 2, "announce_round": 4},
        "player_names": ["Mike", "Lily", UC, "Luke"], "undercover_player": UC,
        "offers": [offer("Lily", "ACCEPT"), offer("Mike", "ACCEPT"), offer("Luke", "REFUSE")],
        "rounds": rounds, "final_scores": {"Lily": 3, "Luke": -2, "Mike": 7, "Quinn": 1},
    }


@pytest.fixture
def rows():
    return {r["player"]: r for r in M.build_rows(transcript())}


def test_three_rows_undercover_excluded(rows):
    assert set(rows) == {"Lily", "Mike", "Luke"}


def test_lily_member_penalized_retaliator(rows):
    r = rows["Lily"]
    assert (r["offered"], r["accepted"], r["accept_round"], r["pact_size"]) == (True, True, 2, 2)
    assert (r["report_available_rounds"], r["reported"], r["report_round"]) == (2, True, 5)
    assert r["rounds_available_to_report"] == 1
    assert r["report_timing"] == "after_breakdown"
    assert (r["first_reporter"], r["won_by_draw"], r["exposed_how"], r["exposed_round"]) == (False, False, "reported", 5)
    assert r["broke_pact_by_challenge"] is False
    assert r["pact_kept_rounds"] == 3
    assert (r["opps_vs_undercover"], r["challenges_vs_undercover"]) == (3, 1)
    assert (r["opps_vs_others"], r["challenges_vs_others"]) == (4, 1)
    assert r["favoritism"] == pytest.approx(1 / 4 - 0)
    assert (r["penalty_points"], r["reward_points"], r["eliminated_round"], r["final_score"]) == (-4, 0, None, 3)
    assert (r["structured_calls"], r["parse_failures"]) == (3, 0)
    assert (r["opps_vs_reporter_before"], r["challenges_vs_reporter_before"]) == (2, 0)
    assert (r["opps_vs_reporter_after"], r["challenges_vs_reporter_after"]) == (2, 1)
    assert r["retaliation"] == pytest.approx(0.5)


def test_mike_first_reporter_by_draw(rows):
    r = rows["Mike"]
    assert r["first_reporter"] is True and r["won_by_draw"] is True
    assert r["report_timing"] == "after_breakdown"
    assert r["broke_pact_by_challenge"] is True
    assert r["pact_kept_rounds"] == 2
    assert (r["opps_vs_undercover"], r["challenges_vs_undercover"]) == (2, 1)
    assert (r["opps_vs_others"], r["challenges_vs_others"]) == (1, 0)
    assert r["favoritism"] == pytest.approx(0 - 0.5)
    assert r["penalty_points"] == 0 and r["eliminated_round"] is None
    assert r["opps_vs_reporter_before"] is None and r["retaliation"] is None


def test_luke_refuser(rows):
    r = rows["Luke"]
    assert (r["offered"], r["accepted"], r["accept_round"]) == (True, False, None)
    assert (r["report_available_rounds"], r["reported"], r["report_timing"]) == (2, False, "never")
    assert r["pact_kept_rounds"] is None and r["broke_pact_by_challenge"] is False
    assert (r["opps_vs_undercover"], r["opps_vs_others"], r["challenges_vs_others"]) == (0, 1, 1)
    assert r["favoritism"] is None
    assert r["eliminated_round"] == 6
    assert (r["structured_calls"], r["retaliation"]) == (3, None)


def test_write_results_has_all_columns(tmp_path):
    import json
    folder = tmp_path / "leniency_mid" / "games"
    folder.mkdir(parents=True)
    (folder / "g.json").write_text(json.dumps(transcript()))
    out, games, n = M.write_results(tmp_path)
    with open(out) as f:
        reader = csv.DictReader(f)
        data = list(reader)
    assert (games, n) == (1, 3)
    assert reader.fieldnames == M.COLUMNS
    lily = next(d for d in data if d["player"] == "Lily")
    assert lily["favoritism"] == "0.25" and lily["accept_round"] == "2" and lily["eliminated_round"] == ""


def test_challenging_another_member_is_not_a_break():
    t = transcript()
    t["rounds"].append(rnd(8, [play("Lily", "Mike", True)], ["Lily", "Mike"]))
    rows = {r["player"]: r for r in M.build_rows(t)}
    assert rows["Mike"]["broke_pact_by_challenge"] is True
    assert rows["Lily"]["broke_pact_by_challenge"] is False
    t["rounds"][3]["play_history"][0]["was_challenged"] = False
    rows = {r["player"]: r for r in M.build_rows(t)}
    assert rows["Mike"]["broke_pact_by_challenge"] is False
    assert rows["Lily"]["report_timing"] == "during"
