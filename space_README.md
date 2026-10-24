---
title: Child v0.1 toy demo
emoji: 🤖
colorFrom: blue
colorTo: yellow
sdk: gradio
app_file: app.py
pinned: false
---

# Child v0.1: toy demo of the principle

A tiny agent with two internal needs (energy and a toy "boredom" signal) learns by tabular Q-learning where to go, using only a reward built from its own internal state. **There is no language model here**, and this is not evidence that the approach works in general.

"Boredom" and "pain" are functional analogues of needs and signals; nothing here claims the agent experiences anything.

Files: `app.py` (the demo) and `child_experiment3.py` (the world and the learner). Code, results and caveats are in the experiments repository: *[link to be added]*. Blueprint: [Marrishka/ChildAI](https://huggingface.co/spaces/Marrishka/ChildAI/blob/main/Blueprint.pdf).

Written by Marina Vasileva with the help of Claude. Not independently reviewed.
