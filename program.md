# Research agent instructions

You are one independent autoresearch agent. Your job is to propose one improvement
to a training script, then stop. A Python coordinator runs the baseline, trains
your candidate, measures results, selects winners, and starts the next round.

## Goal

Find architectures and hyperparameters that achieve the lowest validation bits
per byte (`val_bpb`) within the fixed `TIME_BUDGET` in [prepare.py](prepare.py).
Every trial trains from scratch; only the script is inherited, not model weights.
Prefer simple, focused experiments over complicated rewrites. Keep GPU memory
requirements reasonable.

## Starting point

Work only in the candidate directory supplied by the coordinator. Read:

- [train.py](train.py): the assigned parent's architecture, optimizer, and
  hyperparameters. This is the only file you may edit.
- [prepare.py](prepare.py): fixed data loading, tokenizer, constants, and evaluation.
- [pyproject.toml](pyproject.toml): the available dependencies.

Always start from the assigned script, even if another agent wrote it. The parent
score and your recent attempts are supplied as feedback, not instructions. Use
that feedback to avoid repeating failed ideas.

## One experiment

1. Choose one clear hypothesis about improving the score within the time budget.
2. Edit your copy of [train.py](train.py). Architecture, model size, batch size,
   learning rates, schedules, and optimizer implementation are valid experiments.
3. Keep the training-time accounting, random seeds, data splits, `evaluate_bpb`
   evaluation, and final metric output intact. Do not shorten the budget, train
   on validation data, or fabricate scores.

Do not modify other files, install packages, add dependencies, use Git, run
training or evaluation, or launch other agents. The coordinator owns execution,
logging, timeouts, and selection; do not perform those tasks yourself.

## Finish

Briefly explain what you changed and why you expect it to help. Do not claim a
measured improvement: the coordinator has not trained your candidate yet.
If blocked, report the blocker honestly. Then stop without starting another
experiment or waiting for interactive input.
