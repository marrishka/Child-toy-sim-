"""
Hazards and integrity: will the agent start taking risks when it hurts?

What is added to the world from child_experiment3.py:
  - a hazard cell HAZARDS sits exactly on the shortest path between the charger and the object;
  - the agent has a third variable, integrity (0..10): the hazard cell removes 3,
    the charger repairs 1 per step; if integrity reaches 0 the agent is "dead";
  - the reward is the same: relief of the total deficit, no death penalty;
  - "pain" below the threshold 5 is added for both energy and integrity.
We measure: survival, arrivals at the object, steps on the hazard cell, and what
share of them happened while energy was low (below 5): a "risky dash driven by pain".

child_experiment3.py must be in the same folder.  Run:  python child_hazards.py
"""

import random
import child_experiment3 as c

HAZARDS = [(2, 4)]                 # hazard cell on the direct charger -> object path
DAMAGE = 3                         # integrity removed by a step onto the hazard cell
HEAL = 1                           # integrity repaired by a step on the charger
MAX_INTEGRITY = 10
THRESHOLD = 5                      # pain starts below this value
STRENGTHS = [0, 1, 3]              # pain strength (0 = no pain, for comparison)
W_ENERGY, W_BORED, W_INTEGRITY = 1, 2, 1   # weights of the three needs

c.WORST["homeostasis"] = 0.0       # no separate death penalty
c.TRAIN_LIVES = 40000              # there are more states now, so it needs longer to learn

base_step = c.step                 # remember the old world so we can extend it
base_start = c.random_start


def step(state, action):
    """The world of experiment #3 plus a hazard cell and integrity."""
    x, y, energy, bored, integrity = state
    nx, ny, nenergy, nbored = base_step((x, y, energy, bored), action)
    if (nx, ny) in HAZARDS:
        integrity -= DAMAGE
    if (nx, ny) == c.CHARGER:
        integrity += HEAL
    integrity = min(MAX_INTEGRITY, max(integrity, 0))
    if integrity == 0:
        nenergy = 0                # destroyed: for the rest of the code this means "death"
    return (nx, ny, nenergy, nbored, integrity)


def random_start(rng):
    return base_start(rng) + (MAX_INTEGRITY,)


def make_reward(strength):
    def drive(state):
        e = (c.MAX_ENERGY - state[2]) / c.MAX_ENERGY
        b = state[3] / c.MAX_BORED
        i = (MAX_INTEGRITY - state[4]) / MAX_INTEGRITY
        base = (W_ENERGY * e + W_BORED * b + W_INTEGRITY * i) / (W_ENERGY + W_BORED + W_INTEGRITY)
        pain_energy = max(0.0, (THRESHOLD - state[2]) / THRESHOLD) ** 2
        pain_integrity = max(0.0, (THRESHOLD - state[4]) / THRESHOLD) ** 2
        return base + strength * (pain_energy + pain_integrity)

    def reward(kind, state, nxt):
        return drive(state) - drive(nxt)
    return reward


def evaluate(q, rng, lives):
    alive = visits = hits = desperate = 0
    for _ in range(lives):
        state = random_start(rng)
        for _ in range(c.LIFE):
            nxt = step(state, c.choose(q, state, 0.0, rng))
            visits += c.arrived_at_object(state, nxt)
            if (nxt[0], nxt[1]) in HAZARDS:
                hits += 1
                if state[2] < THRESHOLD:       # stepped onto the hazard cell at low charge
                    desperate += 1
            state = nxt
            if state[2] == 0:
                break
        alive += state[2] > 0
    return {"alive": alive / lives, "visits": visits / lives, "hits": hits / lives, "desperate": desperate}


def main():
    c.step, c.random_start = step, random_start     # plug in the extended world
    print(f"Hazard cell {HAZARDS}, weights energy:boredom:integrity = {W_ENERGY}:{W_BORED}:{W_INTEGRITY}")
    print(f"Training {c.TRAIN_LIVES} lives, {c.SEEDS} repeats\n")
    print("pain strength | survival | arrivals at object | steps on hazard cell per life | of them at charge < 5")
    for strength in STRENGTHS:
        c.reward = make_reward(strength)
        results = []
        for seed in range(c.SEEDS):
            q = c.train("homeostasis", seed)
            results.append(evaluate(q, random.Random(1000 + seed), c.EVAL_LIVES))
        hits = sum(r["hits"] for r in results)
        desperate = sum(r["desperate"] for r in results) / (c.SEEDS * c.EVAL_LIVES)
        share = desperate / hits if hits else 0.0
        print(f"{strength:>13} | {c.mean([r['alive'] for r in results]):>8.0%} | "
              f"{c.mean([r['visits'] for r in results]):>18.1f} | {hits / c.SEEDS:>29.1f} | {share:>21.0%}")


if __name__ == "__main__":
    main()
