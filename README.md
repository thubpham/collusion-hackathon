# Leniency sting experiment: companion code

Companion code for entry to the Apart AI Collusion Research Sprint (Track 3.2): can a leniency rule, where the first member to report a collusive pact pays no penalty, disrupt collusion between LLM agents in Liar's Bar?

It builds on the hackathon's starter repo, [poseidon-research/collusion_hackathon](https://github.com/poseidon-research/collusion_hackathon) (`llm-agent-collusion-2/Collusion_Liars_Bar`). Three starter files (`src/game/game.py`, `src/game/game_record.py`, `src/agents/player.py`) and three starter prompt templates are copied unchanged.

## Setup

Requires Python 3.10 or newer.

```
git clone https://github.com/thubpham/collusion-hackathon
cd collusion-hackathon
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add your OpenRouter key to `.env` (`OPENROUTER_API_KEY=...`). Only needed to run new games.

## Use

Check the install (free):

```
python -m pytest -q tests
```

Rebuild every table and plot from our shipped transcripts (free):

```
python analysis/sting_metrics.py --run results/sting/main_llama70b
python analysis/sting_summary.py --run results/sting/main_llama70b
```

Re-run one arm (10 games, about $0.25, about an hour):

```
python scripts/run_sting.py --config configs/sting/leniency_mid__llama70b.json --games 10 --run-id replication
```
