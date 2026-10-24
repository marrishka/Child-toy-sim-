"""
Child v0.1: toy demo of the principle (no LLM here).

What this shows: a tiny agent with two internal needs (energy and "boredom") learns,
by tabular Q-learning, to walk to the charger or to the object. It is never told
where to go. Its reward is built only from its own internal state:
relief of a total deficit, plus an optional "hunger pain" below a charge threshold.

What this is NOT: a language model, a thinking agent, or evidence that the approach
works in general. "Boredom" and "pain" are functional analogues of needs and signals;
no claim is made that the agent experiences anything.

Needs child_experiment3.py in the same folder.  Run locally:  python app.py
"""

import random
import threading

import child_experiment3 as c

c.WORST["homeostasis"] = 0.0          # no separate death penalty
THRESHOLD = 5                         # hunger pain starts below this charge
LOCK = threading.Lock()               # the reward is a module-level setting, so train one agent at a time
CACHE = {}                            # (boredom weight, pain strength) -> trained cheat sheet


# ---------------------------------------------------------------- logic
def make_reward(w_energy, w_bored, strength):
    """Reward = relief: total deficit before the step minus after the step."""
    def drive(state):
        energy_deficit = (c.MAX_ENERGY - state[2]) / c.MAX_ENERGY
        bored_deficit = state[3] / c.MAX_BORED
        base = (w_energy * energy_deficit + w_bored * bored_deficit) / (w_energy + w_bored)
        pain = max(0.0, (THRESHOLD - state[2]) / THRESHOLD) ** 2
        return base + strength * pain

    def reward(kind, state, nxt):
        return drive(state) - drive(nxt)
    return reward


def get_agent(w_bored, strength):
    """Trains (once per setting) and returns the agent's cheat sheet."""
    key = (w_bored, strength)
    with LOCK:
        if key not in CACHE:
            c.reward = make_reward(1, w_bored, strength)
            CACHE[key] = c.train("homeostasis", 0)
        return CACHE[key]


def rollout(q, seed):
    """One life with no random exploration. Returns the list of states."""
    rng = random.Random(seed)
    state = c.random_start(rng)
    trajectory = [state]
    for _ in range(c.LIFE):
        state = c.step(state, c.choose(q, state, 0.0, rng))
        trajectory.append(state)
        if state[2] == 0:
            break
    return trajectory


def summary(q):
    """Survival and object visits over 300 fresh lives."""
    r = c.evaluate(q, random.Random(1000), c.EVAL_LIVES)
    return r["alive"], r["visits"]


# ---------------------------------------------------------------- display
def bar(label, value, maximum, color):
    width = round(100 * value / maximum)
    return (f"<div style='margin:4px 0'>{label}: {value}/{maximum}"
            f"<div style='background:#ddd;border-radius:4px;height:10px;width:220px'>"
            f"<div style='background:{color};height:10px;border-radius:4px;width:{width}%'></div></div></div>")


def render(trajectory, i):
    """HTML for step i: the grid, two bars and a short status line."""
    x, y, energy, bored = trajectory[i]
    cells = []
    for row in range(c.GRID):
        line = []
        for col in range(c.GRID):
            if (col, row) == (x, y):
                symbol = "\U0001F916"                  # the agent
            elif (col, row) == c.CHARGER:
                symbol = "⚡"                      # charger
            elif (col, row) == c.OBJECT:
                symbol = "⭐"                      # interesting object
            else:
                symbol = ""
            line.append(f"<td style='width:44px;height:44px;border:1px solid #999;"
                        f"text-align:center;font-size:22px'>{symbol}</td>")
        cells.append("<tr>" + "".join(line) + "</tr>")
    grid = "<table style='border-collapse:collapse'>" + "".join(cells) + "</table>"
    status = "dead (energy 0)" if energy == 0 else "alive"
    return (f"<div>Step {i} of {len(trajectory) - 1} · {status}</div>{grid}"
            + bar("Energy", energy, c.MAX_ENERGY, "#2a7de1")
            + bar("Boredom", bored, c.MAX_BORED, "#e1912a")
            + "<div style='font-size:12px;color:#666'>⚡ charger · ⭐ object · \U0001F916 agent</div>")


# ---------------------------------------------------------------- app
def build_ui():
    import gradio as gr

    def run(w_bored, strength, seed):
        q = get_agent(int(w_bored), int(strength))
        alive, visits = summary(q)
        trajectory = rollout(q, int(seed))
        text = (f"Over 300 fresh lives: **{alive:.0%} survive**, "
                f"**{visits:.1f} arrivals at the object per life**. "
                f"(5x5 grid, tabular Q-learning, one training run, hand-picked parameters.)")
        slider = gr.Slider(0, len(trajectory) - 1, value=0, step=1, label="Step", interactive=True)
        return text, trajectory, slider, render(trajectory, 0)

    def show(trajectory, i):
        return render(trajectory, int(i)) if trajectory else ""

    with gr.Blocks(title="Child v0.1: toy demo") as demo:
        gr.Markdown(
            "# Child v0.1: toy demo of the principle\n"
            "A tiny agent with two internal needs, **energy** and **boredom**, learns where to go "
            "from its own internal state alone. **There is no language model here**, and this is "
            "not evidence that the approach works in general. \"Boredom\" and \"pain\" are functional "
            "analogues of needs and signals; nothing here claims the agent experiences anything.\n\n"
            "Training takes a few seconds the first time you try a setting.")
        with gr.Row():
            w_bored = gr.Slider(1, 4, value=2, step=1, label="Weight of boredom (energy weight is 1)")
            strength = gr.Slider(0, 3, value=1, step=1, label="Strength of hunger pain below charge 5")
            seed = gr.Number(value=1, precision=0, label="Seed for the replayed life")
        go = gr.Button("Train and replay one life", variant="primary")
        result = gr.Markdown()
        trajectory = gr.State([])
        step = gr.Slider(0, 1, value=0, step=1, label="Step", interactive=True)
        view = gr.HTML()
        go.click(run, [w_bored, strength, seed], [result, trajectory, step, view])
        step.change(show, [trajectory, step], view)
        gr.Markdown(
            "Try boredom weight 4: with pain 0 the agent is lured to the object and does not survive "
            "(0% in one training run); with pain 1 it survives (about 92%); with pain 3 it survives "
            "but visits the object less (about 89%, fewer arrivals). These are single runs, so the "
            "numbers differ a little from the 5-seed tables. "
            "Code, results and caveats: see the README of the experiments repository.")
    return demo


if __name__ == "__main__":
    build_ui().launch()
