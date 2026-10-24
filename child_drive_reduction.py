"""
Reward = relief (the decrease of the deficit per step).

Before, the agent was punished for feeling bad RIGHT NOW (minus the deficit).
Now it is rewarded for things getting EASIER: reward = deficit before the step
minus deficit after the step. The distance to the charger or the object does
not take part in the reward.

Examples:
  charge rose from 3 to 6 out of 10 -> energy deficit fell from 0.7 to 0.4 -> reward +0.3
  boredom was 8 and reset to zero   -> deficit fell from 0.8 to 0          -> reward +0.8
  an ordinary step                  -> energy 1 lower, boredom 1 higher    -> a small minus

child_experiment3.py must be in the same folder.  Run:  python child_drive_reduction.py
"""

import random
import child_experiment3 as c

# Death is not punished separately: the agent only loses future relief.
c.WORST["homeostasis"] = 0.0

WEIGHTS = [(1, 1), (2, 1), (1, 2)]    # (energy weight, boredom weight)


def deficit(state, w_energy, w_bored):
    """Total deficit of the two needs: 0 is all good, 1 is the worst possible."""
    energy_deficit = (c.MAX_ENERGY - state[2]) / c.MAX_ENERGY
    bored_deficit = state[3] / c.MAX_BORED
    return (w_energy * energy_deficit + w_bored * bored_deficit) / (w_energy + w_bored)


def make_reward(w_energy, w_bored):
    def reward(kind, state, nxt):
        return deficit(state, w_energy, w_bored) - deficit(nxt, w_energy, w_bored)
    return reward


def main():
    print(f"Reward = relief. Training {c.TRAIN_LIVES} lives, {c.SEEDS} repeats\n")
    print("energy weight : boredom weight | survival | arrivals at object | steps on charger at energy >= 8")
    for w_energy, w_bored in WEIGHTS:
        c.reward = make_reward(w_energy, w_bored)
        results = []
        for seed in range(c.SEEDS):
            q = c.train("homeostasis", seed)
            results.append(c.evaluate(q, random.Random(1000 + seed), c.EVAL_LIVES))
        alive = c.mean([r["alive"] for r in results])
        visits = c.mean([r["visits"] for r in results])
        camping = c.mean([r["camping"] for r in results])
        print(f"{w_energy:>13} : {w_bored:<14} | {alive:>8.0%} | {visits:>18.1f} | {camping:>31.1f}")

    print("\nFor comparison: hand-made variant (+1 for the object, penalty for low charge): 92%, 7.5 arrivals")
    print("Earlier reward 'minus the current deficit' at weights 1:1: 68%, 3.4 arrivals")


if __name__ == "__main__":
    main()
