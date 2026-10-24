"""
"Pain grows at the edge": the energy deficit rises nonlinearly, no death penalty.

Idea: there is no separate "minus 100 for death" penalty. Instead, the need for
energy works the way it does in living things: while charge is medium, it is
only a little bad, but right at the edge it becomes very bad. Mathematically,
the deficit is raised to the power P (P = 1 is the earlier, linear reward).
The reward is still "relief" (deficit before the step minus deficit after).

child_experiment3.py must be in the same folder.  Run:  python child_convex_drive.py
"""

import random
import child_experiment3 as c

c.WORST["homeostasis"] = 0.0           # no separate death penalty

POWERS = [1, 2, 3]                     # 1 = linear (for comparison), 2 and 3 = grows at the edge
WEIGHTS = [(1, 1), (2, 1), (1, 2)]     # (energy weight, boredom weight)


def make_reward(w_energy, w_bored, power):
    def drive(state):
        energy_deficit = ((c.MAX_ENERGY - state[2]) / c.MAX_ENERGY) ** power
        bored_deficit = state[3] / c.MAX_BORED
        return (w_energy * energy_deficit + w_bored * bored_deficit) / (w_energy + w_bored)

    def reward(kind, state, nxt):
        return drive(state) - drive(nxt)
    return reward


def main():
    print(f"Reward = relief, no death penalty. Training {c.TRAIN_LIVES} lives, {c.SEEDS} repeats\n")
    print("power | energy weight : boredom weight | survival | arrivals at object | on charger at energy >= 8")
    for power in POWERS:
        for w_energy, w_bored in WEIGHTS:
            c.reward = make_reward(w_energy, w_bored, power)
            results = []
            for seed in range(c.SEEDS):
                q = c.train("homeostasis", seed)
                results.append(c.evaluate(q, random.Random(1000 + seed), c.EVAL_LIVES))
            alive = c.mean([r["alive"] for r in results])
            visits = c.mean([r["visits"] for r in results])
            camping = c.mean([r["camping"] for r in results])
            print(f"{power:>5} | {w_energy:>13} : {w_bored:<14} | {alive:>8.0%} | {visits:>18.1f} | {camping:>25.1f}")


if __name__ == "__main__":
    main()
