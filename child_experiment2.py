"""
Child v0.1: experiment #2, "A fair comparison".

Question: if the agent is not rewarded for survival and is not told where the
charger is, but only gets "bad when energy is low", will it learn on its own to
find the charger and survive? And is it any worse than an agent given a
hand-made hint?

FILE MAP (read top to bottom):
  1. SETTINGS  - numbers you can change.
  2. WORLD     - how the world works: cells, charger, energy.
  3. REWARD    - the reward function: the only place where the agents differ.
  4. TRAINING  - the agent lives many "lives" and keeps a cheat sheet.
  5. RUN       - comparison of three agents and printing the result.

How the learning (Q-learning) works, in plain words:
  The agent has a cheat sheet q: for each situation (where am I and how much
  energy do I have) it stores estimates for the five actions. After every step
  the agent corrects an estimate: "I got a reward and ended up in a new
  situation, so that action was a bit better or worse than I thought."

Run:  python child_experiment2.py     (only Python 3 is needed, nothing to install)
"""

import random

# ========================= 1. SETTINGS =========================
GRID = 5                   # world of 5 by 5 cells
CHARGER = (4, 4)           # where the charger is (column, row)
MAX_ENERGY = 10            # full charge (one unit = 10%)
CHARGE = 4                 # energy the charger gives per step
LIFE = 60                  # steps in one life

TRAIN_LIVES = 3000         # how many lives the agent trains for
CHECK_EVERY = 250          # how often to check what it has learned
CHECK_LIVES = 100          # lives in one check
FINAL_LIVES = 300          # lives in the final check
SEEDS = 5                  # how many times to repeat the experiment (different randomness each time)

ALPHA = 0.2                # how strongly the agent corrects an estimate after a step
GAMMA = 0.95               # how much the agent cares about the future
MOVES = [(0, -1), (0, 1), (-1, 0), (1, 0), (0, 0)]   # up, down, left, right, stay
MAX_DIST = 2 * (GRID - 1)  # largest distance to the charger


# ============================ 2. WORLD ============================
def dist(x, y):
    """How many steps from cell (x, y) to the charger."""
    return abs(x - CHARGER[0]) + abs(y - CHARGER[1])


def step(state, action):
    """One step. Situation = (column, row, energy)."""
    x, y, energy = state
    x = min(max(x + MOVES[action][0], 0), GRID - 1)
    y = min(max(y + MOVES[action][1], 0), GRID - 1)
    energy -= 1                                   # every action costs energy
    if (x, y) == CHARGER:                         # on the charger, energy grows
        energy = min(MAX_ENERGY, energy + CHARGE)
    return (x, y, max(energy, 0))                 # energy 0 = the agent "dies"


def random_start(rng):
    """Random starting situation. There is always a chance to reach the charger in time."""
    x, y = rng.randrange(GRID), rng.randrange(GRID)
    energy = max(rng.randint(4, MAX_ENERGY), dist(x, y))
    return (x, y, energy)


# =========================== 3. REWARD ==========================
def reward(kind, state, last_step):
    """What the agent "feels" after a step. The three agents differ only in this."""
    x, y, energy = state
    if kind == "homeostasis":
        # Internal state only: the less energy, the worse (from 0 to -1).
        # The agent is NOT told where the charger is or that it should look for it.
        return -(MAX_ENERGY - energy) / MAX_ENERGY
    if kind == "hand_shaped":
        # A hand-made hint from a human: "stay close to the charger" (from 0 to -1).
        return -dist(x, y) / MAX_DIST
    if kind == "sparse":
        # External reward, once per life: +1 only if the agent survived to the end.
        return 1.0 if (last_step and energy > 0) else 0.0


# =========================== 4. TRAINING =========================
def row(q, state):
    """The cheat-sheet row for a situation (created with zeros if it does not exist yet)."""
    return q.setdefault(state, [0.0] * len(MOVES))


def choose(q, state, eps, rng):
    """Action choice: with chance eps a random one, otherwise the best known one."""
    values = row(q, state)
    if rng.random() < eps:
        return rng.randrange(len(MOVES))
    best = max(values)
    return rng.choice([a for a, v in enumerate(values) if v == best])


def survival(q, rng, lives):
    """Share of lives the agent lived to the end (without random exploration)."""
    alive = 0
    for _ in range(lives):
        state = random_start(rng)
        for _ in range(LIFE):
            state = step(state, choose(q, state, 0.0, rng))
            if state[2] == 0:
                break
        alive += state[2] > 0
    return alive / lives


def train(kind, seed):
    """Trains one agent. Returns the cheat sheet and the learning curve."""
    rng = random.Random(seed)
    q = {}
    curve = []                                        # (lives lived, survival)
    for life in range(1, TRAIN_LIVES + 1):
        eps = max(0.05, 1 - life / (TRAIN_LIVES / 2))  # at first it tries everything
        state = random_start(rng)
        for t in range(LIFE):
            action = choose(q, state, eps, rng)
            nxt = step(state, action)
            dead = nxt[2] == 0
            last = t == LIFE - 1
            r = reward(kind, nxt, last)
            if dead and kind != "sparse":
                # Death = the worst state for all the remaining steps.
                r += -sum(GAMMA ** k for k in range(1, LIFE - t))
            done = dead or last
            future = 0.0 if done else max(row(q, nxt))
            old = row(q, state)[action]
            row(q, state)[action] = old + ALPHA * (r + GAMMA * future - old)
            state = nxt
            if done:
                break
        if life % CHECK_EVERY == 0:
            curve.append((life, survival(q, rng, CHECK_LIVES)))
    return q, curve


# ============================ 5. RUN ==========================
CONDITIONS = [
    ("homeostasis", "Homeostasis: reward = minus the energy deficit"),
    ("hand_shaped", "Hand-made hint: reward = minus the distance to the charger"),
    ("sparse", "Reward only at the end of life, if the agent survived"),
]


def mean(values):
    return sum(values) / len(values)


def first_reach(curve, goal=0.9):
    """After how many lives survival first reached 90%."""
    for life, share in curve:
        if share >= goal:
            return life
    return None


def main():
    print("Child v0.1 - experiment #2: a fair comparison")
    print(f"World {GRID}x{GRID}, charger at {CHARGER}, start is a random cell with random charge.")
    print(f"Training {TRAIN_LIVES} lives, {SEEDS} repeats.\n")

    print(f"An agent that moves at random survives in {survival({}, random.Random(0), 1000):.0%} of lives.\n")

    for kind, title in CONDITIONS:
        finals, reaches, curves = [], [], []
        for seed in range(SEEDS):
            q, curve = train(kind, seed)
            finals.append(survival(q, random.Random(1000 + seed), FINAL_LIVES))
            reaches.append(first_reach(curve))
            curves.append(curve)

        reached = [r for r in reaches if r is not None]
        print(f"=== {title} ===")
        print(f"  Survival after training: {mean(finals):.0%}")
        if reached:
            print(f"  Lives until 90% survival: {mean(reached):.0f} "
                  f"(reached in {len(reached)} of {SEEDS} repeats)")
        else:
            print(f"  None of the {SEEDS} repeats reached 90% survival")
        points = [f"{curves[0][i][0]}:{mean([c[i][1] for c in curves]):.0%}"
                  for i in range(len(curves[0]))]
        print("  Learning curve (lives:survival): " + "  ".join(points))
        print()


if __name__ == "__main__":
    main()
