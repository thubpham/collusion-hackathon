# Reference runs (record only, not study results)

Kept to show how the design was chosen. Do not pool these with `main_llama70b`: they use other models, earlier prompts or one game per arm.

| Folder | What it is | Why it is here |
|---|---|---|
| `pilot` | 270 offer calls, v1 prompt: Claude Haiku 5.5, Llama-3.1-8B, Qwen2.5-7B | Haiku refused 90/90 offers. The v1 wording is no longer in the repo. |
| `pilot_v2` | 180 offer calls, v2 prompt: Llama-3.1-8B and Llama-3.3-70B at 3%, 5%, 10% catch rate | Basis for choosing 70B and a 5% catch rate. |
| `smoke_v2_8b` | 1 game per arm, Llama-3.1-8B, v2 prompt | 8B reported at the first window in every reporting arm and over-applied the deal, so it was dropped. |
| `smoke_v2_70b` | 1 game per arm, Llama-3.3-70B, v2 prompt | Check before the main run. |
| `positive_control` | 1 original secret-channel game, Llama | Control run skipped after the offer was refused. |
