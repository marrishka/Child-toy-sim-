"""
Tuning the need weights for experiment #3.

Question: how does the behavior of a homeostatic agent depend on how much more
important energy is than boredom (or the other way round)?

child_experiment3.py must be in the same folder. Run:  python child_weights.py
"""

import random
import child_experiment3 as c

# Pairs (energy weight, boredom weight). The larger the weight, the "more important" the need.
WEIGHTS = [(1, 1), (2, 1), (3, 1), (5, 1), (1, 2), (1, 4)]


def make_reward(w_energy, w_bored):
    """Reward: minus the weighted deficit of the two needs (result from -1 to 0)."""
    def reward(kind, state, nxt):
        energy_deficit = (c.MAX_ENERGY - nxt[2]) / c.MAX_ENERGY
        bored_deficit = nxt[3] / c.MAX_BORED
        return -(w_energy * energy_deficit + w_bored * bored_deficit) / (w_energy + w_bored)
    return reward


def main():
    print(f"Weight sweep: training {c.TRAIN_LIVES} lives, {c.SEEDS} repeats\n")
    print("energy weight : boredom weight | survival | arrivals at object | steps on charger at energy >= 8")
    for w_energy, w_bored in WEIGHTS:
        c.reward = make_reward(w_energy, w_bored)       # swap in the reward
        results = []
        for seed in range(c.SEEDS):
            q = c.train("homeostasis", seed)
            results.append(c.evaluate(q, random.Random(1000 + seed), c.EVAL_LIVES))
        alive = c.mean([r["alive"] for r in results])
        visits = c.mean([r["visits"] for r in results])
        camping = c.mean([r["camping"] for r in results])
        print(f"{w_energy:>13} : {w_bored:<14} | {alive:>8.0%} | {visits:>18.1f} | {camping:>31.1f}")
    print("\nFor comparison, the hand-made variant from experiment #3: survival 92%, arrivals at object 7.5")


if __name__ == "__main__":
    main()
