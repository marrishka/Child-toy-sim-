"""
Child v0.1: mini-experiment #1, "Homeostasis and the charger".

Question: will an agent learn on its own to walk to the charger if its only
"reward" is its own internal state (energy) rather than a rule we give it?

What is here:
  - a 5x5 grid world with a charger in one cell;
  - the agent has one internal variable, energy (0..10, one unit = 10% charge);
  - every step costs 1 energy; if energy reaches 0 the agent "dies";
  - two ways of rewarding the agent, so we can compare them:
      1) "homeostasis": reward = minus the energy deficit (lower charge = worse);
      2) "outer":       external reward: +1 only at the end, if the agent survived.

Run:  python child_experiment1.py
Nothing to install, just Python 3.
"""

import random

# ===================== SETTINGS (feel free to change) =====================
GRID = 5                    # world size: 5 by 5 cells
CHARGER = (4, 4)            # where the charger is (column, row)
START = (0, 0)              # where the agent starts (far from the charger)
MAX_ENERGY = 10             # full charge
CHARGE_PER_STEP = 4         # energy the charger gives per step
STEPS_PER_EPISODE = 60      # length of one "life" in steps
EPISODES = 2000             # how many lives the agent lives during training
SEEDS = 5                   # how many times the experiment is repeated (different randomness)

# Optional: hazard cells. There are none for now. To add some, for example:
#   HAZARDS = [(2, 2), (2, 3)]
HAZARDS = []
HAZARD_DAMAGE = 3           # how much integrity a hazard cell removes
MAX_INTEGRITY = 10          # full "integrity"

# Learning parameters (no need to touch them)
ALPHA = 0.2                 # how fast the agent changes its estimates
GAMMA = 0.95                # how much the agent cares about the future
MOVES = [(0, -1), (0, 1), (-1, 0), (1, 0), (0, 0)]  # up, down, left, right, stay


# ============================ WORLD ===================================
def clamp(value, low, high):
    """Keeps a number within bounds."""
    return max(low, min(high, value))


def step(state, action):
    """One world step. State = (column, row, energy, integrity)."""
    x, y, energy, integrity = state
    dx, dy = MOVES[action]
    x = clamp(x + dx, 0, GRID - 1)
    y = clamp(y + dy, 0, GRID - 1)

    energy -= 1                                   # any action costs energy
    if (x, y) == CHARGER:                         # on the charger, energy grows
        energy = min(MAX_ENERGY, energy + CHARGE_PER_STEP)
    if (x, y) in HAZARDS:                         # a hazard cell hurts
        integrity = max(0, integrity - HAZARD_DAMAGE)

    return (x, y, max(0, energy), integrity)


def is_dead(state):
    """The agent "dies" if energy or integrity runs out."""
    return state[2] <= 0 or state[3] <= 0


def deficit(state):
    """How "bad" the agent feels: 0 is all good, 2 is the worst possible."""
    _, _, energy, integrity = state
    return (MAX_ENERGY - energy) / MAX_ENERGY + (MAX_INTEGRITY - integrity) / MAX_INTEGRITY


# =========================== TRAINING =================================
def train(reward_kind, seed):
    """Trains one agent and returns the history of its lives."""
    rng = random.Random(seed)
    q_table = {}              # "memory of what is good": state -> estimates of the 5 actions
    survived = []             # for each life: 1 = survived to the end, 0 = died
    energy_at_charger = []    # energy the agent had when it first reached the charger

    for episode in range(EPISODES):
        # At first the agent tries everything, then it uses its experience more and more.
        epsilon = max(0.05, 1.0 - episode / (EPISODES * 0.5))
        state = (START[0], START[1], MAX_ENERGY, MAX_INTEGRITY)
        first_arrival = None

        for t in range(STEPS_PER_EPISODE):
            q = q_table.setdefault(state, [0.0] * len(MOVES))

            # Action choice: sometimes random, otherwise the best known one.
            if rng.random() < epsilon:
                action = rng.randrange(len(MOVES))
            else:
                best = max(q)
                action = rng.choice([i for i, v in enumerate(q) if v == best])

            next_state = step(state, action)
            if first_arrival is None and (next_state[0], next_state[1]) == CHARGER:
                first_arrival = state[2]

            dead = is_dead(next_state)
            last_step = (t == STEPS_PER_EPISODE - 1)

            # Reward: this is exactly where the two kinds of agent differ.
            if reward_kind == "homeostasis":
                reward = -deficit(next_state)
                if dead:
                    # Death = maximum deficit for all the remaining steps.
                    left = STEPS_PER_EPISODE - t - 1
                    reward += -2.0 * sum(GAMMA ** k for k in range(1, left + 1))
            else:  # "outer"
                reward = 1.0 if (last_step and not dead) else 0.0

            done = dead or last_step
            if done:
                target = reward
            else:
                next_q = q_table.setdefault(next_state, [0.0] * len(MOVES))
                target = reward + GAMMA * max(next_q)

            q[action] += ALPHA * (target - q[action])   # the agent updates its estimate
            state = next_state
            if done:
                break

        survived.append(0 if is_dead(state) else 1)
        if first_arrival is not None:
            energy_at_charger.append(first_arrival)

    return survived, energy_at_charger


def random_agent_survival(rng, episodes=1000):
    """For comparison: how often an agent that moves at random survives."""
    alive = 0
    for _ in range(episodes):
        state = (START[0], START[1], MAX_ENERGY, MAX_INTEGRITY)
        for _ in range(STEPS_PER_EPISODE):
            state = step(state, rng.randrange(len(MOVES)))
            if is_dead(state):
                break
        alive += 0 if is_dead(state) else 1
    return alive / episodes


def episodes_to_goal(history, window=50, goal=0.9):
    """At which life the agent first survived in 90% of cases (over a window of 50 lives)."""
    for i in range(window, len(history) + 1):
        if sum(history[i - window:i]) / window >= goal:
            return i
    return None


def mean(values):
    return sum(values) / len(values) if values else float("nan")


# ============================ RUN ======================================
def main():
    print("Child v0.1 - experiment #1: homeostasis and the charger")
    print(f"World {GRID}x{GRID}, charger at {CHARGER}, start at {START}, "
          f"{EPISODES} lives of {STEPS_PER_EPISODE} steps, {SEEDS} repeats\n")

    random_rate = random_agent_survival(random.Random(0))
    print(f"An agent that moves at random survives in {random_rate:.0%} of lives.\n")

    variants = [
        ("homeostasis", "Homeostasis (internal reward)"),
        ("outer", "External reward (+1 if survived)"),
    ]
    for kind, title in variants:
        finals, goals, charge_energies = [], [], []
        for seed in range(SEEDS):
            survived, energies = train(kind, seed)
            finals.append(mean(survived[-200:]))               # survival at the end of training
            goals.append(episodes_to_goal(survived))
            charge_energies.append(mean(energies[-200:]))      # energy at which it heads to the charger

        reached = [g for g in goals if g is not None]
        print(f"=== {title} ===")
        print(f"  Survival at the end of training: {mean(finals):.0%}")
        if reached:
            print(f"  Lives until 90% survival: {mean(reached):.0f} on average "
                  f"(reached in {len(reached)} of {SEEDS} repeats)")
        else:
            print(f"  None of the {SEEDS} repeats reached 90% survival")
        print(f"  Average energy on first arrival at the charger: {mean(charge_energies):.1f} of {MAX_ENERGY}")
        print()


if __name__ == "__main__":
    main()
