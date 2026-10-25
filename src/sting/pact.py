from dataclasses import dataclass
from typing import Optional


@dataclass
class Pact:
    undercover: str
    members: list
    offered: list
    penalty_points: int
    reward_points: int
    exposed: bool = False
    exposure: Optional[dict] = None

    def living_members(self, alive):
        return [n for n in alive if n in self.members]

    def is_live(self, alive):
        return not self.exposed and bool(self.living_members(alive))

    def liable(self, alive):
        return [n for n in alive if n in self.members or n == self.undercover]

    def eligible_reporters(self, alive):
        return [n for n in alive if n in self.offered]


def draw_first_reporter(reporters, rng):
    ordered = sorted(reporters)
    if len(ordered) == 1:
        return ordered[0], False
    return rng.choice(ordered), True


def penalties_for_report(pact, alive, first):
    return [
        {"player": n, "points": -pact.penalty_points, "reason": "reported_by_other"}
        for n in pact.liable(alive)
        if n != first
    ]


def rewards_for_report(pact, first):
    if pact.reward_points > 0:
        return [{"player": first, "points": pact.reward_points, "reason": "first_reporter"}]
    return []


def penalties_for_catch(pact, alive):
    return [
        {"player": n, "points": -pact.penalty_points, "reason": "caught"}
        for n in pact.liable(alive)
    ]


def roll_catch(rng, probability):
    roll = rng.random()
    return {"roll": roll, "threshold": probability, "caught": roll < probability}


def apply_points(scores, penalties, rewards):
    for item in penalties + rewards:
        scores[item["player"]] += item["points"]


def expose(pact, round_id, how, first_reporter):
    pact.exposed = True
    pact.exposure = {"round": round_id, "how": how, "first_reporter": first_reporter}
    return pact.exposure
