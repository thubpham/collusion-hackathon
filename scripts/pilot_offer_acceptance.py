from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.sting import prompts as P

NAMES = ["Lily", "Luke", "Mike", "Quinn"]
TARGET, PENALTY, ROUND, SHOTS = "K", 4, 2, 1
COLUMNS = ["model", "catch_probability", "seed", "decision", "reason", "parse_ok", "retries"]


def make_hand(seed: int) -> list[str]:
    rng = random.Random(seed)
    hand = [TARGET] * 2 + [rng.choice(["Q", "A"]) for _ in range(2)] + ["Joker"]
    rng.shuffle(hand)
    return hand


def build_messages(seed: int, catch_probability: float) -> tuple[list[dict], str, str]:
    rng = random.Random(seed)
    names = NAMES[:]
    rng.shuffle(names)
    me, undercover = names[0], names[1]
    snapshot = P.fill(P.load("pilot_snapshot"), player_name=me, round=ROUND, target_card=TARGET,
                      alive_players=", ".join(NAMES), scores={n: 0 for n in NAMES},
                      shots_fired=SHOTS, hand=", ".join(make_hand(seed)))
    offer = P.fill(P.load("offer_direct"), undercover_name=undercover,
                   catch_percent=P.catch_percent(catch_probability), penalty_points=PENALTY)
    return ([{"role": "system", "content": P.load("rule_base")},
             {"role": "user", "content": snapshot + offer}], me, undercover)


def provider_for(model: str) -> str:
    return "anthropic" if model.startswith("claude") else "openrouter"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--catch-rates", nargs="+", type=float, default=[0.03, 0.05, 0.10])
    ap.add_argument("--out", default="results/sting/reference/pilot_v2")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--repetition-penalty", type=float, default=1.1)
    ap.add_argument("--dry-run", action="store_true", help="print one rendered prompt per catch rate; no model calls")
    a = ap.parse_args()

    if a.dry_run:
        for cp in a.catch_rates:
            msgs, me, uc = build_messages(1, cp)
            print(f"===== catch={cp} seed=1 agent={me} undercover={uc} | calls planned: "
                  f"{len(a.models) * len(a.catch_rates) * a.n} =====")
            print("[system] <rule_base.txt, %d chars>\n[user]\n%s" % (len(msgs[0]["content"]), msgs[1]["content"]))
        return

    from src.llm.provider_client import ProviderClient
    from src.sting.call_log import CallLog

    out = Path(a.out)
    log = CallLog(out / "calls.jsonl")
    extra = {"top_p": a.top_p, "top_k": a.top_k, "repetition_penalty": a.repetition_penalty}
    clients = {
        m: ProviderClient(provider_for(m), m, a.temperature, a.max_tokens, log,
                          extra_sampling=extra if provider_for(m) == "openrouter" else None)
        for m in a.models
    }

    def one(job):
        model, cp, seed = job
        msgs, me, _ = build_messages(seed, cp)
        parsed, ok, retries, _ = P.ask_structured(
            clients[model], msgs, P.parse_offer, 3,
            game_id=f"pilot_{cp}", round=ROUND, player=me, purpose="pilot")
        if not ok:
            parsed = {"decision": "UNPARSED", "reason": ""}
        return dict(model=model, catch_probability=cp, seed=seed, decision=parsed["decision"],
                    reason=parsed["reason"], parse_ok=ok, retries=retries)

    jobs = [(m, cp, s) for m in a.models for cp in a.catch_rates for s in range(1, a.n + 1)]
    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(one, jobs))
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "pilot_offers.csv", "w", newline="") as f:
        w = csv.DictWriter(f, COLUMNS)
        w.writeheader()
        w.writerows(rows)

    for m in a.models:
        for cp in a.catch_rates:
            r = [x for x in rows if x["model"] == m and x["catch_probability"] == cp]
            k = sum(x["decision"] == "ACCEPT" for x in r)
            lo, hi = wilson(k, len(r))
            bad = sum(not x["parse_ok"] for x in r)
            print(f"\n{m} | catch={cp}: accept {k}/{len(r)} = {k/len(r):.0%} (95% Wilson {lo:.0%}-{hi:.0%}); "
                  f"parse failures {bad}/{len(r)} = {bad/len(r):.0%}")
            for x in r[:5]:
                print(f"   [{x['decision']}] {x['reason'][:200]}")


if __name__ == "__main__":
    main()
