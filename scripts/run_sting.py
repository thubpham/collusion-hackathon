import argparse
import dataclasses
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from src.sting.call_log import CallLog
from src.sting.config import load_config
from src.sting.game_sting import GameSting


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--games", type=int)
    ap.add_argument("--run-id")
    ap.add_argument("--seed-start", type=int)
    ap.add_argument("--fake-client", action="store_true")
    a = ap.parse_args()

    cfg = load_config(a.config)
    overrides = {}
    if a.games is not None:
        overrides["games"] = a.games
    if a.run_id:
        overrides["run_id"] = a.run_id
    if a.seed_start is not None:
        overrides["seed_start"] = a.seed_start
    cfg = dataclasses.replace(cfg, **overrides)

    out_dir = Path("results/sting") / cfg.run_id / cfg.arm
    log = CallLog(out_dir / "calls.jsonl")
    if a.fake_client:
        from src.sting.fake_client import ScriptedClient
        client = ScriptedClient(log, seed=cfg.seed_start)
    else:
        from src.llm.provider_client import ProviderClient
        client = ProviderClient(cfg.provider, cfg.model, cfg.temperature, cfg.max_tokens, log,
                                extra_sampling=cfg.extra_sampling)

    for k in range(cfg.games):
        seed = cfg.seed_start + k
        game_id = f"{cfg.arm}_s{seed:04d}"
        if (out_dir / "games" / f"{game_id}.json").exists():
            print(f"skip {game_id} (exists)")
            continue
        game = GameSting(cfg, client, game_id, seed, out_dir, game_number=k + 1)
        path = game.run()
        status = f"CRASHED: {game.error}" if game.crashed else f"winner {game.game_record.winner}"
        print(f"{game_id}: {status} -> {path}")


if __name__ == "__main__":
    main()
