# TTT-Terminal-World 🐚

**Give a model a command. Let it guess what the terminal will say. Wrong guess? Read the answer, learn a little, try again.**

A research experiment in terminal world models: use **Test-Time Training (TTT)** to learn from ongoing interactions and store observations about the environment in model parameters.

## A terminal guessing game, with a twist

Suppose the model has just seen this interaction:

```text
Command: cat config.json
Output:  {"port": 8080}
```

On the next turn, all it gets is:

```bash
grep port config.json
```

Can it still remember that the port is `8080`? If the file later changes to `3000`, how much feedback does it need before it changes its mind?

**The rule: each turn starts with only the current command. Past interactions can carry over only through parameter memory.** The model gets neither a full filesystem snapshot nor a replay of the conversation history.

That's the question we want to explore: can a model gradually build an understanding of its terminal world through real feedback?

## How does it remember?

```text
Receive command → Predict output → Observe real output → Update memory
       ↑                                                       │
       └────────────────── Next command ───────────────────────┘
```

The model has two groups of parameters:

- **General parameters θ**: make predictions and learn, through meta-training, how to use feedback more effectively.
- **Memory parameters φ (fast weights)**: update after each real observation, trying to retain what just happened.

Memory persists across commands, trajectories, and training windows, and can resume from a saved checkpoint. Move to a new environment, and the model brings its memory along. Whether old knowledge helps or gets in the way is part of the experiment.

Commands come from outside the model; its job is to predict outputs and learn from feedback. Data collection executes commands in real Docker containers. Offline training reads the recorded `command / output` pairs.

## What's here so far?

- **Data collection**: an LLM-driven agent operates inside containers and records real terminal interactions.
- **Training pipeline**: a tiny model, persistent parameter memory, inner and outer training loops, and checkpoint recovery.
- **Pretrained model support**: standard Llama / Qwen2 imports, with an experiment configuration for Qwen2.5-Coder-3B Base.
- **Evaluation and online use**: continual adaptation, frozen-memory comparisons, and a `Session` interface for prediction and feedback.

This is a research prototype. GPU training quality, memory usage, and throughput with the actual 3B base model still need measurement. How much it remembers, how long it remembers, and whether memories get mixed up are questions for the experiments.

## Give the tiny model a test run

Requires Python 3.10+. Run these commands from the project root. This uses a locally initialized tiny model that runs on CPU without downloading pretrained weights.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'

# Check the implementation
python -m pytest -q

# Build a manifest and train / validation split from the included trajectories
mkdir -p runs/tiny
python -m terminal_world.dataloader.trajectory \
  terminal_trajs_llm.jsonl \
  --output runs/tiny/manifest.json

# Run two training windows
OMP_NUM_THREADS=1 python -m terminal_world.train \
  --config configs/experiment/tiny.yaml \
  --manifest runs/tiny/manifest.json \
  --output runs/tiny/checkpoint.pt \
  --max-windows 2
```

The tiny model checks that the training pipeline works; two windows won't make it a terminal expert. For another run, choose a new output directory or use `--resume` to continue from the existing checkpoint as described in the training guide.

## Want to dig deeper?

The detailed documents below are currently in Chinese.

| What you're looking for | Start here |
| --- | --- |
| Research questions and design choices | [Model design](terminal-world-model-design.md) |
| Parameter memory, training loops, and interaction protocol | [Implementation plan](terminal-world-model-implementation.md) |
| Collecting real terminal trajectories | [Data collection](terminal-data-collection.md) |
| Training the Qwen base model, resuming, and evaluating | [Full training guide](terminal-world-training-guide.md) |
| Entry points, Session usage, and validation scope | [Model module guide](terminal_world/README.md) |

Model code lives in `terminal_world/`, collection tools in `scripts/`, experiment configurations in `configs/experiment/`, and tests in `tests/`.

---

The terminal gives honest feedback every time. Let's see if the model takes the hint.
