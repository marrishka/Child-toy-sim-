"""
Hunger pain: the energy deficit gets a threshold below which it starts to hurt.

Like a stomach ache when hungry: while charge is above the threshold there is
no pain, below it the pain grows stronger the lower the charge. There is no
separate death penalty. The reward is still "relief" (total deficit before the
step minus after).

Total deficit = ordinary deficit of the two needs + PAIN_STRENGTH * pain
where pain = ((THRESHOLD - energy) / THRESHOLD) squared if energy is below the threshold, else 0.

child_experiment3.py must be in the same folder.  Run:  python child_pain_threshold.py
"""

import random
import child_experiment3 as c

c.WORST["homeostasis"] = 0.0           # no separate death penalty

THRESHOLD = 5                          # pain starts below this charge
PAIN_STRENGTHS = [0, 1, 3]             # 0 = no pain (for comparison)
WEIGHTS = [(1, 1), (1, 2), (1, 4)]     # (energy weight, boredom weight)


def make_reward(w_energy, w_bored, strength):
    def drive(state):
        energy_deficit = (c.MAX_ENERGY - state[2]) / c.MAX_ENERGY
        bored_deficit = state[3] / c.MAX_BORED
        base = (w_energy * energy_deficit + w_bored * bored_deficit) / (w_energy + w_bored)
        pain = max(0.0, (THRESHOLD - state[2]) / THRESHOLD) ** 2
        return base + strength * pain

    def reward(kind, state, nxt):
        return drive(state) - drive(nxt)
    return reward


def main():
    print(f"Hunger pain below charge {THRESHOLD}. Training {c.TRAIN_LIVES} lives, {c.SEEDS} repeats\n")
    print("pain strength | energy weight : boredom weight | survival | arrivals at object | on charger at energy >= 8")
    for strength in PAIN_STRENGTHS:
        for w_energy, w_bored in WEIGHTS:
            c.reward = make_reward(w_energy, w_bored, strength)
            results = []
            for seed in range(c.SEEDS):
                q = c.train("homeostasis", seed)
                results.append(c.evaluate(q, random.Random(1000 + seed), c.EVAL_LIVES))
            alive = c.mean([r["alive"] for r in results])
            visits = c.mean([r["visits"] for r in results])
            camping = c.mean([r["camping"] for r in results])
            print(f"{strength:>13} | {w_energy:>13} : {w_bored:<14} | {alive:>8.0%} | {visits:>18.1f} | {camping:>25.1f}")


if __name__ == "__main__":
    main()
