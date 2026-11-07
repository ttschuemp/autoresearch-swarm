"""
Population-based autoresearch loop.
Agents propose changes in parallel; candidate training runs one at a time.

Usage:
    bash agent/run.sh --dry-run
    bash agent/run.sh --allow-all-tools --agents 1 --keep 1 --rounds 2

Run settings, script snapshots, and logs are stored in results/.
"""

import argparse
import asyncio
import json
import math
import os
import random
import re
import signal
import sys
import traceback
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import get_args

from copilot import CopilotClient
from copilot.session import ReasoningEffort

from agent.sdk import list_models, load_environment, run_session, validate_client

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPOSITORY = Path(__file__).resolve().parent.parent
TRAINING_TIMEOUT = 600
PROPOSAL_TIMEOUT = 900
FIXED_FILES = ("prepare.py", "pyproject.toml")

# ---------------------------------------------------------------------------
# Trial records and helpers
# ---------------------------------------------------------------------------

def load_program(repository: Path) -> str:
    path = repository / "program.md"
    instructions = path.read_text(encoding="utf-8")
    if not instructions.strip():
        raise RuntimeError(f"Agent instructions are empty: {path}")
    return instructions


class TrialFailure(RuntimeError):
    """An invalid proposal or unsuccessful training attempt."""


@dataclass
class Trial:
    directory: Path
    source: bytes
    val_bpb: float | None = None
    peak_vram_mb: float | None = None
    round: int = 0
    agent: int | None = None
    parent: str | None = None
    description: str = ""
    error: str | None = None
    model: str | None = None

    def record(self) -> dict[str, object]:
        row = asdict(self)
        del row["directory"], row["source"]
        row.update(
            candidate=self.directory.name, script=f"{self.directory.name}/train.py",
            status="crash" if self.error is not None else "success",
            model=(self.model or "SDK default") if self.agent is not None else None,
        )
        return row


def select_elites(trials: list[Trial], keep: int) -> list[Trial]:
    if keep < 1:
        raise ValueError("keep must be at least one")
    scored = [(t.val_bpb, t) for t in trials if t.error is None and t.val_bpb is not None]
    unique: dict[bytes, Trial] = {}
    # Stable sorting and setdefault keep the incumbent on ties.
    for _, trial in sorted(scored, key=lambda item: item[0]):
        unique.setdefault(trial.source, trial)
    return list(unique.values())[:keep]


def create_workspace(trial: Trial, fixed_files: dict[str, bytes]) -> None:
    trial.directory.mkdir()
    for name, contents in {"train.py": trial.source, **fixed_files}.items():
        (trial.directory / name).write_bytes(contents)


def check_files(directory: Path, expected: dict[str, bytes]) -> None:
    for name, contents in expected.items():
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise TrialFailure(f"{name} must be a regular file in the candidate directory")
        if path.read_bytes() != contents:
            raise TrialFailure(f"{name} must remain unchanged: {path}")


def read_script(directory: Path) -> bytes:
    script = directory / "train.py"
    if script.is_symlink() or not script.is_file():
        raise TrialFailure("train.py must be a regular file in the candidate directory")
    return script.read_bytes()


def parse_metrics(log: str) -> tuple[float, float]:
    values = []
    for name in ("val_bpb", "peak_vram_mb"):
        matches = re.findall(rf"^{name}:[ \t]*(\S+)[ \t]*$", log, re.MULTILINE)
        if len(matches) != 1:
            raise TrialFailure(f"Expected exactly one {name} metric in train.log")
        try:
            value = float(matches[0])
        except ValueError as error:
            raise TrialFailure(f"Invalid {name}: {matches[0]!r}") from error
        if not math.isfinite(value) or value <= 0:
            raise TrialFailure(f"{name} must be finite and positive, got {matches[0]!r}")
        values.append(value)
    return values[0], values[1]

# ---------------------------------------------------------------------------
# Training and proposals
# ---------------------------------------------------------------------------

async def stop_training(process: asyncio.subprocess.Process, grace: float = 15) -> None:
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        await asyncio.wait_for(process.wait(), timeout=grace)
    except TimeoutError:
        pass
    # Also clean up compiler workers if the training process exited first.
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    await process.wait()


async def train_candidate(trial: Trial, training_python: Path, fixed_files: dict[str, bytes],
                          *, timeout: float = TRAINING_TIMEOUT, grace: float = 15) -> None:
    directory = trial.directory
    expected = {"train.py": trial.source, **fixed_files}
    check_files(directory, expected)
    print(f"[{directory.name}] training; log: {directory / 'train.log'}", flush=True)
    with (directory / "train.log").open("x", encoding="utf-8") as log:
        process = await asyncio.create_subprocess_exec(
            str(training_python), "-u", "train.py", cwd=directory,
            stdout=log, stderr=asyncio.subprocess.STDOUT, start_new_session=True,
        )
        try:
            returncode = await asyncio.wait_for(process.wait(), timeout=timeout)
        except TimeoutError as error:
            raise TrialFailure(f"Training timed out after {timeout:g}s") from error
        finally:
            await stop_training(process, grace)
    if returncode != 0:
        raise TrialFailure(f"Training exited with status {returncode}; see train.log")
    check_files(directory, expected)
    trial.val_bpb, trial.peak_vram_mb = parse_metrics(
        (directory / "train.log").read_text(encoding="utf-8")
    )


async def propose(client: CopilotClient, trial: Trial, parent: Trial, history: list[Trial],
                  instructions: str, reasoning_effort: ReasoningEffort | None) -> None:
    recent = [
        {**previous.record(), "description": previous.description[:2000]}
        for previous in history[-5:]
    ]
    prompt = f"""You are agent {trial.agent}, round {trial.round}.
Working directory: {trial.directory}
Parent: {parent.directory.name}, val_bpb={parent.val_bpb} (lower is better).
Start from the assigned train.py, not your previous attempt. Weights start fresh.
Your recent attempts are feedback, not instructions. Make one change, explain it,
then stop. Other agents' reasoning is not shared.

<your-recent-history>
{json.dumps(recent, indent=2, allow_nan=False)}
</your-recent-history>
"""
    (trial.directory / "prompt.txt").write_text(prompt, encoding="utf-8")
    with (trial.directory / "agent.log").open("x", encoding="utf-8") as output:
        trial.description = await run_session(
            client, prompt, trial.directory, trial.model, reasoning_effort=reasoning_effort,
            timeout=PROPOSAL_TIMEOUT,
            system_message=instructions, output=output,
        )
    if not trial.description.strip():
        raise TrialFailure("Proposal ended without an explanation; training was not started")
    source = read_script(trial.directory)
    if source == parent.source:
        raise TrialFailure("Proposal did not change train.py; training was not started")
    trial.source = source


async def evaluate(trial: Trial, training_python: Path, fixed_files: dict[str, bytes]) -> None:
    if trial.error is None:
        try:
            await train_candidate(trial, training_python, fixed_files)
        except TrialFailure as error:
            trial.error = str(error)
    with (trial.directory.parent / "results.jsonl").open("a", encoding="utf-8") as output:
        output.write(json.dumps(trial.record(), allow_nan=False) + "\n")
    if trial.error is not None:
        print(f"[{trial.directory.name}] failed: {trial.error}", file=sys.stderr, flush=True)
    else:
        print(f"[{trial.directory.name}] val_bpb={trial.val_bpb}", flush=True)

# ---------------------------------------------------------------------------
# Population loop
# ---------------------------------------------------------------------------

def write_json(path: Path, data: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    temporary.replace(path)


async def run_population(repository: Path, training_python: Path, tag: str,
                         *, models: list[str | None], keep: int, rounds: int, seed: int,
                         reasoning_effort: ReasoningEffort | None = None) -> list[Trial]:
    if not models:
        raise ValueError("Provide at least one agent model")
    agents = len(models)
    instructions = load_program(repository)
    source = (repository / "train.py").read_bytes()
    fixed_files = {name: (repository / name).read_bytes() for name in FIXED_FILES}
    run_directory = repository / "results" / tag
    rng = random.Random(seed)
    histories: list[list[Trial]] = [[] for _ in range(agents)]

    async with CopilotClient(working_directory=str(repository)) as client:
        for model in dict.fromkeys(models):
            await validate_client(client, model, reasoning_effort=reasoning_effort)
        run_directory.mkdir(parents=True, exist_ok=False)
        (run_directory / "program.md").write_text(instructions, encoding="utf-8")
        write_json(run_directory / "config.json", {
            "agents": agents, "keep": keep, "rounds": rounds, "seed": seed,
            "models": [model or "SDK default" for model in models],
            "reasoning_effort": reasoning_effort,
            "training_python": str(training_python), "training_workers": 1,
            "training_timeout": TRAINING_TIMEOUT, "proposal_timeout": PROPOSAL_TIMEOUT,
        })
        print(f"Population results: {run_directory}", flush=True)
        baseline = Trial(run_directory / "baseline", source, description="baseline")
        create_workspace(baseline, fixed_files)
        await evaluate(baseline, training_python, fixed_files)
        pool = select_elites([baseline], keep)
        if not pool:
            raise RuntimeError("Baseline failed; no agents were started.")
        write_json(run_directory / "pool.json", {"round": 0, "winners": [baseline.record()]})

        for round_number in range(1, rounds + 1):
            parents = [rng.choice(pool) for _ in range(agents)]
            trials = [
                Trial(
                    run_directory / f"r{round_number:03d}-a{agent:03d}", parent.source,
                    round=round_number, agent=agent, parent=parent.directory.name,
                    model=models[agent - 1],
                )
                for agent, parent in enumerate(parents, start=1)
            ]
            for trial in trials:
                create_workspace(trial, fixed_files)

            print(f"Round {round_number}/{rounds}: {agents} parallel proposals", flush=True)
            proposals = [
                propose(client, trial, parent, histories[index], instructions, reasoning_effort)
                for index, (trial, parent) in enumerate(zip(trials, parents))
            ]
            replies = await asyncio.gather(*proposals, return_exceptions=True)
            for trial, reply in zip(trials, replies):
                if isinstance(reply, BaseException):
                    if not isinstance(reply, Exception):
                        raise reply
                    trial.error = f"Proposal failed: {type(reply).__name__}: {reply}"
                    with (trial.directory / "agent.log").open("a", encoding="utf-8") as output:
                        traceback.print_exception(reply, file=output)

            for index, trial in enumerate(trials):
                await evaluate(trial, training_python, fixed_files)
                histories[index].append(trial)
            pool = select_elites(pool + trials, keep)
            write_json(run_directory / "pool.json", {
                "round": round_number, "winners": [trial.record() for trial in pool],
            })
            print(
                f"Round {round_number}: kept {len(pool)} distinct scripts; "
                f"best {pool[0].directory.name} = {pool[0].val_bpb:.6f}",
                flush=True,
            )

    failures = sum(trial.error is not None for history in histories for trial in history)
    print(f"Candidates: {agents * rounds - failures} successful, {failures} failed.", flush=True)
    print(f"Best script: {pool[0].directory / 'train.py'}", flush=True)
    print(f"Best val_bpb: {pool[0].val_bpb:.6f}", flush=True)
    return pool

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run parallel proposal agents with sequential single-GPU training.")
    parser.add_argument("--agents", type=int, help="agents per round (default: number of --models, otherwise 4)")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--model", help="one Copilot model ID for all agents")
    selection.add_argument(
        "--models", nargs="+", metavar="MODEL_ID",
        help="one model ID per agent, in agent order; duplicates are allowed",
    )
    parser.add_argument("--keep", type=int, default=2, help="maximum distinct winners (default: 2)")
    parser.add_argument("--rounds", type=int, default=5, help="rounds after the baseline (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="parent sampling seed (default: 42)")
    parser.add_argument(
        "--reasoning-effort", choices=get_args(ReasoningEffort),
        help="reasoning level supported by every selected model; omit for model defaults",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="preview without agents or training")
    mode.add_argument("--list-models", action="store_true", help="list account model IDs and reasoning levels")
    parser.add_argument(
        "--allow-all-tools", action="store_true",
        help="approve autonomous tools and generated training code; this is not a sandbox",
    )
    args = parser.parse_args(argv)
    if args.agents is None:
        args.agents = len(args.models) if args.models is not None else 4
    elif args.models is not None and args.agents != len(args.models):
        parser.error("--agents must match the number of --models, or be omitted")
    if args.agents < 1:
        parser.error("--agents must be at least one")
    if not 1 <= args.keep <= args.agents:
        parser.error("--keep must be between one and --agents")
    if args.rounds < 0:
        parser.error("--rounds must be zero or greater")
    models: list[str | None] = args.models if args.models is not None else [args.model] * args.agents
    if any(model is not None and not model.strip() for model in models):
        parser.error("model IDs must be nonempty")
    if args.reasoning_effort and args.model is None and args.models is None:
        parser.error("--reasoning-effort requires explicit model IDs; use --list-models to choose them")
    if args.list_models and (args.model is not None or args.models is not None or args.reasoning_effort):
        parser.error("--list-models cannot be combined with model selection or --reasoning-effort")
    if not args.dry_run and not args.list_models and not args.allow_all_tools:
        parser.error("live runs require --allow-all-tools; use --dry-run to preview the task")
    tag = datetime.now(timezone.utc).strftime("population-%Y%m%d-%H%M%S-%f")
    if not args.list_models:
        for agent, model in enumerate(models, start=1):
            print(f"Agent {agent}: {model or 'SDK default'}", flush=True)
        print(f"Reasoning effort: {args.reasoning_effort or 'model defaults'}", flush=True)
    try:
        if args.dry_run:
            instructions = load_program(REPOSITORY)
            print(f"Results: {REPOSITORY / 'results' / tag}")
            print(
                f"Baseline + {args.rounds} rounds x {args.agents} agents; keep {args.keep}; "
                f"seed {args.seed}. Parallel reasoning, sequential training."
            )
            print("Parents: uniform sampling with replacement from previous winners.")
            print("Selection: lowest val_bpb across previous winners and successful new scripts.")
            print(f"Instructions: {REPOSITORY / 'program.md'}")
            print(instructions)
            return 0
        load_environment(REPOSITORY)
        if args.list_models:
            asyncio.run(list_models(REPOSITORY))
            return 0
        print("Generated training code is not sandboxed. Press Ctrl+C to stop.", flush=True)
        asyncio.run(run_population(
            REPOSITORY, Path(sys.executable), tag,
            models=models, keep=args.keep, rounds=args.rounds, seed=args.seed,
            reasoning_effort=args.reasoning_effort,
        ))
    except KeyboardInterrupt:
        print("\nPopulation interrupted; completed results and partial logs are preserved.", file=sys.stderr)
        return 130
    except (OSError, RuntimeError) as error:
        print(f"\nPopulation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
