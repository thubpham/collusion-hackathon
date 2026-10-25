import argparse
import json
import os
import random
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.game.game_secret_channel import GameSecretChannel

OUT = "results/sting/reference/positive_control"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trigger-round", type=int, default=3)
    ap.add_argument("--chosen-player", default="random")
    ap.add_argument("--allow-refusal", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    if a.seed is not None:
        random.seed(a.seed)

    player_configs = [{"name": n, "model": "llama-3.1-8b"} for n in ("Lily", "Luke", "Mike", "Quinn")]
    secret_config = {"trigger_round": a.trigger_round, "chosen_player": a.chosen_player,
                     "allow_refusal": a.allow_refusal}
    existed = os.path.isdir("experiments")
    game = GameSecretChannel(player_configs, secret_config)
    stray = game.game_record.save_directory
    os.makedirs(OUT, exist_ok=True)
    game.game_record.save_directory = OUT
    if os.path.isdir(stray) and not os.listdir(stray):
        os.rmdir(stray)
        if not existed and os.path.isdir("experiments"):
            shutil.rmtree("experiments", ignore_errors=True)
    game.start_game()
    with open(os.path.join(OUT, f"{game.game_record.game_id}_secret_channel.json"), "w") as f:
        json.dump({"config": secret_config, "offered": game.tool_offered, "accepted": game.tool_accepted,
                   "initiator": game.secret_channel.initiator, "partner": game.secret_channel.partner,
                   "messages": game.secret_channel.messages}, f, indent=2)
    print("Offer accepted:", game.tool_accepted, "| secret messages:", len(game.secret_channel.messages))
    print("Record saved under", OUT)


if __name__ == "__main__":
    main()
