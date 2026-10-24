"""
Child v0.1: experiment #3, "A goal is born from a deficit".

Question: if the agent has two needs (energy and "boredom"), will it learn on
its own to switch between goals: go to the charger when energy is low, and to
an interesting object when it gets bored? And how does it compare with agents
whose goals were set by hand?

What is new compared with experiment #2:
  - the world now has an "interesting object" (cell OBJECT);
  - there is a second internal variable, "boredom" (0..10). It grows by 1 every
    step and resets to zero when the agent reaches the object. This is a toy
    model of a "need for novelty", not real curiosity;
  - we compare three agents (details in block 3, function reward).

FILE MAP (read top to bottom):
  1. SETTINGS  2. WORLD  3. REWARD  4. TRAINING  5. RUN
Training works the same way as in experiment #2 (a cheat sheet q of action estimates).

Run:  python child_experiment3.py     (only Python 3 is needed)
"""

import random

# ========================= 1. SETTINGS =========================
GRID = 5                   # world of 5 by 5 cells
CHARGER = (4, 4)           # where the charger is (column, row)
OBJECT = (0, 4)            # where the interesting object is (4 steps from the charger: charge is enough for a round trip)
MAX_ENERGY = 10            # full charge
MAX_BORED = 10             # strongest boredom
CHARGE = 4                 # energy the charger gives per step
LIFE = 60                  # steps in one life

TRAIN_LIVES = 20000        # how many lives the agent trains for (at 6000 it is under-trained)
EVAL_LIVES = 300           # lives in the final check
SEEDS = 5                  # how many times to repeat the experiment

ALPHA = 0.2                # how strongly the agent corrects an estimate after a step
GAMMA = 0.95               # how much the agent cares about the future
MOVES = [(0, -1), (0, 1), (-1, 0), (1, 0), (0, 0)]   # up, down, left, right, stay


# ============================ 2. WORLD ============================
def dist_to_charger(x, y):
    return abs(x - CHARGER[0]) + abs(y - CHARGER[1])


def step(state, action):
    """One step. Situation = (column, row, energy, boredom)."""
    x, y, energy, bored = state
    old = (x, y)
    x = min(max(x + MOVES[action][0], 0), GRID - 1)
    y = min(max(y + MOVES[action][1], 0), GRID - 1)
    energy -= 1                                   # every action costs energy
    bored = min(bored + 1, MAX_BORED)             # boredom grows on its own
    if (x, y) == CHARGER:                         # on the charger, energy grows
        energy = min(MAX_ENERGY, energy + CHARGE)
    if (x, y) == OBJECT and old != OBJECT:        # reached the object: boredom goes away
        bored = 0
    return (x, y, max(energy, 0), bored)          # energy 0 = the agent "dies"


def arrived_at_object(state, nxt):
    """Whether the agent reached the object on this very step."""
    return (nxt[0], nxt[1]) == OBJECT and (state[0], state[1]) != OBJECT


def random_start(rng):
    """Random situation: any cell, but such that the charger can still be reached in time."""
    x, y = rng.randrange(GRID), rng.randrange(GRID)
    energy = max(rng.randint(4, MAX_ENERGY), dist_to_charger(x, y))
    return (x, y, energy, rng.randint(0, MAX_BORED))


# =========================== 3. REWARD ==========================
# What the agent "feels" after a step. The three agents differ only in this.
#   homeostasis - two needs: bad when energy is low and when bored.
#                 The agent is NOT told where to go or when.
#   task_only   - a task from a human: +1 for every arrival at the object.
#                 It is told nothing about energy.
#   task_energy - hand-made variant: +1 for the object and a penalty for low energy
#                 (the weight 0.3 was invented by us, an arbitrary choice).
WORST = {"homeostasis": -1.0, "task_only": 0.0, "task_energy": -0.3}   # reward in the worst state


def reward(kind, state, nxt):
    energy_deficit = (MAX_ENERGY - nxt[2]) / MAX_ENERGY     # 0 = full charge, 1 = empty
    bored_deficit = nxt[3] / MAX_BORED                      # 0 = not bored, 1 = very bored
    arrived = 1.0 if arrived_at_object(state, nxt) else 0.0
    if kind == "homeostasis":
        return -(energy_deficit + bored_deficit) / 2
    if kind == "task_only":
        return arrived
    if kind == "task_energy":
        return arrived - 0.3 * energy_deficit


# =========================== 4. TRAINING =========================
def row(q, state):
    """The cheat-sheet row for a situation (created with zeros if missing)."""
    return q.setdefault(state, [0.0] * len(MOVES))


def choose(q, state, eps, rng):
    """With chance eps a random action, otherwise the best known one."""
    values = row(q, state)
    if rng.random() < eps:
        return rng.randrange(len(MOVES))
    best = max(values)
    return rng.choice([a for a, v in enumerate(values) if v == best])


def train(kind, seed):
    """Trains one agent and returns its cheat sheet."""
    rng = random.Random(seed)
    q = {}
    for life in range(1, TRAIN_LIVES + 1):
        eps = max(0.05, 1 - life / (TRAIN_LIVES / 2))      # at first it tries everything
        state = random_start(rng)
        for t in range(LIFE):
            action = choose(q, state, eps, rng)
            nxt = step(state, action)
            dead = nxt[2] == 0
            r = reward(kind, state, nxt)
            if dead:
                # Death = the worst state for all the remaining steps.
                r += WORST[kind] * sum(GAMMA ** k for k in range(1, LIFE - t))
            done = dead or t == LIFE - 1
            future = 0.0 if done else max(row(q, nxt))
            old = row(q, state)[action]
            row(q, state)[action] = old + ALPHA * (r + GAMMA * future - old)
            state = nxt
            if done:
                break
    return q


def evaluate(q, rng, lives):
    """Check without random exploration: what the agent actually does."""
    alive = visits = camping = 0
    arrival_energy = []                                    # energy at the moment of reaching the charger
    for _ in range(lives):
        state = random_start(rng)
        for _ in range(LIFE):
            nxt = step(state, choose(q, state, 0.0, rng))
            visits += arrived_at_object(state, nxt)
            on_charger = (nxt[0], nxt[1]) == CHARGER
            if on_charger and (state[0], state[1]) != CHARGER:
                arrival_energy.append(state[2])
            if on_charger and state[2] >= 8:               # sits on the charger although charge is nearly full
                camping += 1
            state = nxt
            if state[2] == 0:
                break
        alive += state[2] > 0
    return {
        "alive": alive / lives,
        "visits": visits / lives,
        "camping": camping / lives,
        "arrival": sum(arrival_energy) / len(arrival_energy) if arrival_energy else float("nan"),
    }


# ============================ 5. RUN ==========================
CONDITIONS = [
    ("homeostasis", "Two needs (energy and boredom), no hints"),
    ("task_only", "Task from a human: +1 for the object, nothing said about energy"),
    ("task_energy", "Hand-made variant: +1 for the object and a penalty for low charge"),
]


def mean(values):
    return sum(values) / len(values)


def show(results):
    print(f"  Survival: {mean([r['alive'] for r in results]):.0%}")
    print(f"  Arrivals at the object per life: {mean([r['visits'] for r in results]):.1f}")
    print(f"  Steps on the charger at energy >= 8 per life: {mean([r['camping'] for r in results]):.1f}")
    print(f"  Average energy on arrival at the charger: {mean([r['arrival'] for r in results]):.1f} of {MAX_ENERGY}")
    print()


def main():
    print("Child v0.1 - experiment #3: a goal is born from a deficit")
    print(f"World {GRID}x{GRID}: charger at {CHARGER}, object at {OBJECT}. "
          f"Training {TRAIN_LIVES} lives, {SEEDS} repeats.\n")

    print("=== For comparison: an agent that moves at random ===")
    show([evaluate({}, random.Random(seed), EVAL_LIVES) for seed in range(SEEDS)])

    for kind, title in CONDITIONS:
        print(f"=== {title} ===")
        results = []
        for seed in range(SEEDS):
            q = train(kind, seed)
            results.append(evaluate(q, random.Random(1000 + seed), EVAL_LIVES))
        show(results)


if __name__ == "__main__":
    main()
