# Autoresearch

Single-GPU LLM training experiments powered by Copilot SDK agents. Agents change
[train.py](train.py) to search for better architectures and hyperparameters
within a fixed training time budget. Validation bits per byte (**val_bpb**)
is the score; lower is better. The same population loop supports one or many agents.

## Setup

Requires Linux, Bash, Git, an NVIDIA GPU compatible with CUDA 12.8,
[uv](https://docs.astral.sh/uv/getting-started/installation/), and Copilot access.
Python 3.11 is selected by [.python-version](.python-version).

From the repository root:

```bash
bash scripts/pod-init.sh
```

This installs dependencies and prepares the data using the shared `/opt` paths
expected by the launcher. It is safe to rerun and saves both uv path settings
in `~/.bashrc` for new shells. It does not change the current shell's environment.

Alternatively, run the setup commands manually:

```bash
export UV_PROJECT_ENVIRONMENT=/opt/autoresearch-venv
export UV_CACHE_DIR=/opt/autoresearch-cache/uv
uv sync --locked
/opt/autoresearch-venv/bin/python prepare.py
```

The `/opt` locations must be writable. Keep both `UV_` variables set when using
`uv`. Data and tokenizer files are cached in `~/.cache/autoresearch/`.

The development dependencies include `ipykernel` for notebooks. Register the
project kernel once:

```bash
/opt/autoresearch-venv/bin/python -m ipykernel install --user \
  --name autoresearch --display-name "Python (autoresearch)"
```

In VS Code, select `/opt/autoresearch-venv/bin/python` for Python and
`Python (autoresearch)` for notebooks. The workspace settings use this Python
environment by default.

The [workspace settings](.vscode/settings.json) set
`python.analysis.typeCheckingMode` to `off` for these experimental scripts.
Syntax and missing-import diagnostics remain enabled; no local type stubs are needed.

Sign in with the Copilot CLI, or set `COPILOT_GITHUB_TOKEN` in the local,
Git-ignored `agent/.env`. The launcher loads it automatically; existing
environment variables take precedence. VS Code sign-in alone is not sufficient.
Never commit tokens.

## Run

```bash
# Preview without authentication or GPU work
bash agent/run.sh --dry-run

# List available models and reasoning levels
bash agent/run.sh --list-models

# Two different agents, keeping the best two scripts
bash agent/run.sh --allow-all-tools \
  --models MODEL_A MODEL_B --keep 2 --rounds 2

# Run the same loop with just one agent
bash agent/run.sh --allow-all-tools --agents 1 --keep 1 --rounds 2 \
  --model MODEL_ID --reasoning-effort high

# All options
bash agent/run.sh --help
```

Replace the model placeholders with IDs from `--list-models`. `--models` takes one
ID per agent; its order stays fixed across rounds. Repeating an ID creates another
independent agent using that model. Omit `--agents` to use the list length, or set it
to the same count. Use either `--models` or the shared `--model`, not both.
Without either option, agents use SDK defaults.

An optional `--reasoning-effort` applies to every agent and must be supported by
every selected model. Omit it to use each model's default. All selected models
are validated before any training. Add `--dry-run` to preview the assignments
without authentication or GPU work.

The launcher now always uses the population loop. Existing
`bash agent/run.sh population ...` commands still work, but the prefix is optional.
The old Git-commit/TSV workflow and `--experiments` option have been retired.
Use `--rounds`, or `--rounds 0` for a baseline only.

To run training without the agent:

```bash
/opt/autoresearch-venv/bin/python -u train.py
```

**This is not a sandbox:** autonomous tools and generated training code have your
account's access. Use a dedicated machine and reserve the GPU for one research run
at a time. Live research requires `--allow-all-tools`.

## How the loop works

The coordinator is ordinary Python in [agent/population.py](agent/population.py),
not another AI agent. [agent/sdk.py](agent/sdk.py) contains only environment loading,
Copilot authentication/model checks, and proposal sessions. One `Trial` record
follows each script through proposal, training, and selection. The baseline runs
once before the round loop. Both use the same evaluation and logging functions:

1. Copy the current working tree's training script and run one fresh baseline.
2. At the start of each round, each agent independently samples a parent from the
   previous winners, uniformly **with replacement**. Initially only the baseline
   is available.
3. Give each agent its own directory containing a copy of the complete parent
   script. Fresh Copilot sessions reason and edit **concurrently**. Each gets the
   parent score and its own five most recent attempts, not other agents' reasoning.
   Each keeps its assigned model, even when inheriting code written by another
   agent using a different model.
4. After the proposals finish, Python runs their training jobs **one at a time**.
   Every job starts from fresh weights and uses the same fixed evaluator and
   configured training budget. The Python environment and cached data are shared,
   not copied. Agents have only file-view/edit tools; they do not launch training.
5. Keep up to `--keep` distinct scripts with the lowest scores from **previous
   winners plus successful new candidates**. Exact source duplicates occupy one
   slot; incumbent winners win score ties. Copy winners for the next round rather
   than editing their archived scripts.

Edit [program.md](program.md) to change the agents' shared research instructions.
The coordinator reads it once at the start of each run and supplies its complete
text as the system-message instructions for every proposal session. Each agent
also gets its parent assignment and its own recent history. Changes made during
a run apply only to the next run. `--dry-run` displays the current instructions;
a missing or empty file is an error. Python still handles training and selection.

Without `--models`, defaults are four agents, two winners, five rounds, and
sampling seed 42. This is one baseline plus at most 20 candidate training runs.
`--keep` must be between one and the agent count; the pool can contain fewer
winners until enough distinct successful scripts exist. `--rounds 0` runs only
the baseline. The seed reproduces
parent sampling given the same winners, not LLM proposals or wall-clock GPU runs.

Each agent gets one attempt per round. Proposal errors, empty/unchanged proposals,
timeouts, nonzero training exits, missing/duplicate/nonfinite/nonpositive metrics,
or changes to the fixed helper files are failures, never winners. There are no automatic retries.
A failed baseline stops the run; failed candidates leave previous winners eligible.
Proposal turns have a 15-minute timeout and training commands a 600-second timeout.
The agents must preserve the training/evaluation protocol within their script;
this is an experiment workflow, not an adversarial-code sandbox.

## Results

Everything is stored under a new Git-ignored `results/population-<timestamp>/`:

- `config.json`: run settings, including the model list in agent order.
- `program.md`: the exact shared instructions used for this run.
- `baseline/` and `rNNN-aNNN/`: individual script snapshots and fixed helpers;
  `train.log` contains training output. Candidate directories also contain the
  proposal prompt in `prompt.txt` and SDK output in `agent.log`.
- `results.jsonl`: one record per completed attempt, with agent, round, parent,
  score, explanation, model, and explicit success/failure.
  Failed scores are `null`, never a misleading zero. Each candidate records the
  model that proposed it, not its parent's model. The baseline has `model: null`
  because it is run by Python without an AI proposal.
- `pool.json`: atomically updated after each completed round, listing the current
  winners, their scores, and script paths relative to the run directory.

The final console output identifies the best script. No Git branches or commits
are created, and the repository's original [train.py](train.py), `run.log`, and
`results.tsv` are not replaced. Existing uncommitted code is copied as the baseline,
not reset. Ctrl+C aborts active proposal sessions and stops the coordinator-owned
training process group; completed results and partial logs remain. There is no
automatic resume.

Only one GPU training worker is implemented. It uses the first GPU visible to
the process (respecting `CUDA_VISIBLE_DEVICES`). Multi-GPU scheduling can later
replace the serial training stage without changing parent selection or reasoning.

Open [analysis.ipynb](analysis.ipynb) and run all cells to read the existing
`results/*/results.jsonl` files directly, without an export or helper script.
The notebook summarizes runs and models, compares candidates with their recorded
parents, and plots the latest run by default; set `RUN` to view another run.
It saves `progress.png` without changing the logs or the legacy `results.tsv`.

## Credits

Based on Andrej Karpathy's [autoresearch](https://github.com/karpathy/autoresearch)
and [nanochat](https://github.com/karpathy/nanochat). MIT license.
