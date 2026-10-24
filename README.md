# Child v0.1: toy simulations

Small, dependency-free simulations accompanying **"Child"**, a concept for an embodied agent whose goals arise from internal needs rather than from an assigned task, guided by the ethic *"Do not use. Recognize."*

- Blueprint (public, commit dated Oct 31, 2025): https://huggingface.co/spaces/Marrishka/ChildAI/blob/main/Blueprint.pdf
- Specification draft and one-page reviewer summary: see the [`Docs/`](Docs/) folder

**Status: early, exploratory, feedback welcome.** These experiments show feasibility in a toy world. They do not show that the approach works in general.

## What is tested

A 5x5 grid world with a charger. The agent has internal variables (energy, a toy "boredom" signal, and in one experiment integrity). It is never told where to go. Behavior is learned with tabular Q-learning from a reward built only from its own internal state.

"Boredom", "pain" and similar words are used as **functional analogues of needs and signals**. Nothing here claims the agent experiences anything.

## Results (5 seeds each, hand-picked parameters)

| Experiment | Variant | Survival | Object visits per life |
| --- | --- | --- | --- |
| 2. Charger only | Homeostasis (no hint) | 91% | n/a |
| 2. Charger only | Hand-shaped reward | 96% | n/a |
| 2. Charger only | Sparse reward | 20% | n/a |
| 3. Charger + object | Homeostasis, weights 1:1 | 68% | 3.4 |
| 3. Charger + object | Hand-made task + energy penalty | 92% | 7.5 |
| 3. Charger + object | Task only | 0% | n/a |
| Relief reward | Weights 3:2 / 3:4 (energy:boredom) | 89% / 0% | not reported |
| Convex deficit (power 3) | Weights 1:1 | 91% | 4.8 |
| Pain threshold, strength 1 | Weights 1:4 | 92% | 5.1 |
| Pain threshold, strength 3 | Weights 1:1 | 94% | 0.2 (over-passive) |
| Hazard on the direct path | Pain strength 0 / 1 / 3 | 0% / 97% / 97% | 2.5 / 0.4 / 0.1 |

Takeaways, with caveats:

- A homeostatic reward can learn to survive without hints, but a hand-tuned reward still explores more.
- Outcomes are knife-edge in the need weights (3:2 gives 89% survival, 3:4 gives 0%).
- A nonlinear "pain" threshold removes the need for an explicit death penalty in balanced settings, but too much pain makes the agent passive.
- In the hazard test the detour is too long for a round trip, so the observed "caution" depends on that geometry, and exploration collapsed.

## Limits

Tiny world, tabular learner, 5 seeds, hand-picked parameters, no hyperparameter search, no comparison with established homeostatic RL baselines. An architectural safeguard (not only reward tuning) is still needed, and the "integrity" signal is not yet protected from sycophancy.

## Files

| File | What it runs |
| --- | --- |
| `child_experiment1.py` | First mini-experiment: homeostasis vs outer reward, fixed start |
| `child_experiment2.py` | Fair comparison: homeostasis vs hand-shaped vs sparse reward, random starts |
| `child_experiment3.py` | Two needs (energy, boredom) vs hand-made task rewards. **Other scripts import this file.** |
| `child_weights.py` | Sweep of energy:boredom weights |
| `child_drive_reduction.py` | Reward = relief (decrease of the deficit) |
| `child_convex_drive.py` | Nonlinear (convex) energy deficit, no death penalty |
| `child_pain_threshold.py` | "Hunger pain" below a charge threshold |
| `child_hazards.py` | Hazard cell and integrity |
| `app.py` | Optional toy demo with sliders (needs `pip install gradio`; run `python app.py`). No language model. |

## How to run

Python 3 only, nothing to install. Keep all files in one folder, then for example:

```
python child_experiment3.py
python child_hazards.py
```

Longer experiments may take several minutes. Results use fixed seeds, so reruns should reproduce the tables above.

## Authorship and AI assistance

Written by Marina Vasileva, an independent author who is not a professional programmer. The Blueprint was co-written with Qwen3-Max. The v0.1 specification and all code here were written with Claude. The code has not had an independent review; that is one of the things I am asking for.

## Questions for reviewers

1. Is relief of a deficit plus a nonlinear "pain" threshold a sound way to avoid an explicit death penalty, or a hidden penalty by another name? How does it differ from existing homeostatic reinforcement learning?
2. How should an agent's "integrity" be protected from sycophancy if it depends on approval or criticism, while keeping it corrigible through a maintenance-mode trust model?
3. What would be the most convincing next experiment: hazards that do not cut off the goal, more seeds, or Crafter?

I have not done a systematic prior-art search. Pointers to closer work are very welcome.

## License

Code: MIT (see `LICENSE`). Text: CC BY 4.0. *(Choose the licenses you want before publishing; these are suggestions.)*