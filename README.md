# Autoresearch

Single-GPU LLM training experiments powered by a Copilot SDK agent. The agent
changes [train.py](train.py), measures validation bits per byte (**val_bpb**),
and keeps or rejects each change. Lower is better.

## Setup

Requires Linux, Bash, Git, GNU coreutils, an NVIDIA GPU compatible with CUDA 12.8,
[uv](https://docs.astral.sh/uv/getting-started/installation/), and Copilot access.
Python 3.11 is selected by [.python-version](.python-version).

From the repository root:

```bash
export UV_PROJECT_ENVIRONMENT=/opt/autoresearch-venv
export UV_CACHE_DIR=/opt/autoresearch-cache/uv
uv sync --locked
/opt/autoresearch-venv/bin/python prepare.py
```

The `/opt` locations must be writable. Keep both `UV_` variables set when using
`uv`. Data and tokenizer files are cached in `~/.cache/autoresearch/`.

In VS Code, select `/opt/autoresearch-venv/bin/python` for Python and notebooks.
The workspace settings use this environment by default.

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

# Run a baseline plus up to three candidate attempts (default)
bash agent/run.sh --allow-all-tools

# Choose the attempt count and model
bash agent/run.sh --allow-all-tools --experiments 1 \
  --model MODEL_ID --reasoning-effort high

# All options
bash agent/run.sh --help
```

Replace `MODEL_ID` with an available model and choose a supported reasoning level.
Omit both options for SDK defaults. Use `--experiments 0` for a baseline only.
Crashes and repair retries count toward the attempt limit.

To run training without the agent:

```bash
/opt/autoresearch-venv/bin/python -u train.py
```

**This is not a sandbox:** autonomous tools have your account's access. Use a
dedicated machine and run only one training process or agent on the GPU.
Live research requires `--allow-all-tools`.

## Results and limits

```bash
tail -F run.log
```

- `run.log`: current training output.
- `results/<run-tag>/`: archived baseline and attempt logs.
- `results.tsv`: appended metrics and `keep`, `discard`, or `crash` outcomes.
- [analysis.ipynb](analysis.ipynb): results analysis and plotting.

Each launch requests a fresh baseline and a bounded set of candidates on a new
branch. The agent follows [program.md](program.md) with launcher overrides,
editing and committing only [train.py](train.py). Start with no edits to that
file and no staged changes.

Training has a 300-second budget plus startup and evaluation, with a 600-second
command timeout. The SDK allows 15 minutes per planned invocation.
Experiment rules are agent instructions, not independently enforced checks;
a successful session exit does not mean every experiment succeeded.

Ctrl+C requests an abort, but tool-launched training may remain running.
Check remaining processes before restarting. There is no automatic retry or resume.

## Credits

Based on Andrej Karpathy's [autoresearch](https://github.com/karpathy/autoresearch)
and [nanochat](https://github.com/karpathy/nanochat). MIT license.
