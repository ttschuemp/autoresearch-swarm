import argparse
import asyncio
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import get_args

from copilot import CopilotClient, ToolSet
from copilot.session import PermissionHandler, ReasoningEffort
from copilot.session_events import (
    AssistantMessageData,
    AssistantMessageDeltaData,
    SessionErrorData,
    SessionEvent,
    ToolExecutionCompleteData,
    ToolExecutionStartData,
)
from dotenv import load_dotenv

REPOSITORY = Path(__file__).resolve().parent.parent
TOOLS = [
    "view",
    "edit",
    "create",
    "glob",
    "grep",
    "bash",
    "read_bash",
    "stop_bash",
    "list_bash",
]
SYSTEM_MESSAGE = """You are autoresearch's standalone research agent.
Follow the supplied protocol and run overrides. Complete only the bounded batch.
Work alone; do not launch subagents or other coding agents.
Use the supplied shared Python interpreter for training.
Report blockers and stop; do not wait for interactive input or invent results.
"""


def load_environment(repository: Path) -> None:
    path = repository / "agent" / ".env"
    if path.exists():
        with path.open(encoding="utf-8") as source:
            load_dotenv(stream=source, override=False)


def build_training_command(training_python: Path, log_path: Path) -> str:
    command = (
        "timeout --signal=TERM --kill-after=15s 600s "
        f"{shlex.quote(str(training_python))} -u train.py"
    )
    # Open the archive before starting training, so redirection errors stop it.
    pipeline = f"{{ {command} 2>&1 | tee run.log; }} > {shlex.quote(str(log_path))}"
    return shlex.join(["bash", "-o", "pipefail", "-o", "noclobber", "-c", pipeline])


def build_prompt(
    repository: Path, training_python: Path, experiments: int, tag: str
) -> str:
    program = (repository / "program.md").read_text(encoding="utf-8")
    log_directory = Path("results") / tag
    baseline_command = build_training_command(training_python, log_directory / "baseline.log")
    candidate_command = build_training_command(training_python, log_directory / "attempt-NNN.log")
    return f"""Repository: {repository}

Follow this protocol, except for the run overrides below:
<repository-program>
{program.rstrip()}
</repository-program>

Run overrides (take precedence over the protocol):

1. Run exactly one fresh baseline of the starting code, then at most {experiments}
   candidate attempts. Crashes, timeouts, and repair retries count as attempts.
   Do not retry the baseline: if it fails, record the failure and stop.
   Stop when the budget is exhausted; ignore LOOP FOREVER and NEVER STOP.
   Compare against this fresh baseline, not historical results.

2. Before edits, inspect Git status. Stop for pre-existing train.py edits,
   staged changes, or missing prepared data, tokenizer, GNU timeout, or a free GPU.
   Do not install anything. Preserve unrelated changes and untracked files.
   Create autoresearch/{tag} from current HEAD, not master, and results/{tag}/.
   Stop if either already exists. Before the baseline, back up existing run.log to
   results/{tag}/previous-run.log; stop if that backup fails.

3. Run training from the repository root, one trial at a time, without detaching.
   Baseline command:
   {baseline_command}
   Candidate command (replace NNN with 001, 002, ...):
   {candidate_command}
   Use only these commands; never uv run or additional training invocations.
   Keep pipefail and noclobber. Use the pipeline exit status; wait for exit.
   Output goes live to run.log and its archive; run.log is replaced for each invocation.
   Never overwrite archives or add redirections that flood the tool context.
   Read metrics from that invocation's archive.

4. Only edit and commit train.py for research; preserve prepare.py, its evaluator,
   the fixed training budget, dependencies, program.md, and agent/ application.
   Include a meaningful commit message and this trailer:
   Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>
   Restore rejected/crashed candidates by reverting only your own candidate commit
   to the last accepted code. Never stage all files, amend, reset, clean, stash,
   push, or modify remote repositories. Preserve unrelated work and commit history.

5. Append to results.tsv; never truncate or reinitialize it. Create a header only
   when absent/empty; stop on a different existing header:
   commit\tval_bpb\tmemory_gb\tstatus\tdescription
   Follow the protocol's formatting and keep/discard/crash statuses. Prefix
   descriptions with {tag}; log the baseline and each attempt.
   Do not commit results.tsv or logs. Accept only successful exits with finite
   val_bpb and memory; nonzero exits, missing/invalid metrics, OOM, and timeouts
   are crashes. Use the protocol's crash values, never accept 0.000000 as an improvement.

6. Apply the protocol's metric and simplicity criteria when keeping candidates.
   Summarize the baseline, attempts used, best result, accepted commit,
   log directory, and blockers. Then stop.
"""


async def run_agent(
    prompt: str, repository: Path, model: str | None, experiments: int,
    *, reasoning_effort: ReasoningEffort | None = None, list_models: bool = False,
) -> None:
    async with CopilotClient(working_directory=str(repository)) as client:
        auth = await client.get_auth_status()
        if not auth.isAuthenticated:
            raise RuntimeError(
                "Copilot is not authenticated on this machine. Sign in with the "
                "Copilot CLI, or supply COPILOT_GITHUB_TOKEN in the environment. "
                "VS Code authentication is not automatically available here."
            )

        if list_models or model is not None:
            models = await client.list_models()
            if not models:
                raise RuntimeError("Copilot returned no models for this account.")
            if list_models:
                print("MODEL ID\tREASONING EFFORTS\tDEFAULT EFFORT\tPOLICY")
                for item in models:
                    efforts = ", ".join(
                        effort for effort in item.supported_reasoning_efforts or []
                        if effort in get_args(ReasoningEffort)
                    ) or "-"
                    policy = item.policy.state if item.policy else "-"
                    print(f"{item.id}\t{efforts}\t{item.default_reasoning_effort or '-'}\t{policy}")
                return
            selected = next((item for item in models if item.id == model), None)
            if selected is None:
                raise RuntimeError(f"Unknown model {model!r}. Use --list-models to see model IDs.")
            if selected.policy and selected.policy.state != "enabled":
                raise RuntimeError(f"Model {model!r} is {selected.policy.state} by account policy.")
            if reasoning_effort is not None:
                efforts = selected.supported_reasoning_efforts or []
                if reasoning_effort not in efforts:
                    supported = ", ".join(efforts) or "none advertised"
                    raise RuntimeError(
                        f"Model {model!r} does not advertise reasoning effort "
                        f"{reasoning_effort!r}. Supported: {supported}."
                    )

        async with await client.create_session(
            model=model,
            reasoning_effort=reasoning_effort,
            working_directory=str(repository),
            on_permission_request=PermissionHandler.approve_all,
            available_tools=ToolSet().add_builtin(TOOLS),
            system_message={"mode": "append", "content": SYSTEM_MESSAGE},
            streaming=True,
            skip_custom_instructions=True,
            enable_config_discovery=False,
            enable_file_hooks=False,
        ) as session:
            print(f"Copilot session: {session.session_id}", flush=True)
            streamed: set[str] = set()

            def on_event(event: SessionEvent) -> None:
                match event.data:
                    case AssistantMessageDeltaData() as data if data.delta_content:
                        streamed.add(data.message_id)
                        print(data.delta_content, end="", flush=True)
                    case AssistantMessageData() as data:
                        if data.message_id in streamed:
                            streamed.remove(data.message_id)
                            print(flush=True)
                        elif data.content:
                            print(data.content, flush=True)
                    case ToolExecutionStartData() as data:
                        print(f"\n[tool] {data.tool_name}", flush=True)
                    case ToolExecutionCompleteData() as data if not data.success:
                        detail = data.error.message if data.error else "No error detail provided"
                        print(f"\n[tool failed] {detail}", file=sys.stderr, flush=True)
                    case SessionErrorData() as data:
                        print(f"\n[session error] {data.message}", file=sys.stderr, flush=True)

            session.on(on_event)
            try:
                # The SDK default is 60 seconds, shorter than a single training run.
                response = await session.send_and_wait(
                    prompt, timeout=900 * (experiments + 1)
                )
                if response is None:
                    raise RuntimeError("Copilot ended without an assistant response.")
            except (asyncio.CancelledError, TimeoutError):
                print("\nStopping the Copilot turn...", file=sys.stderr, flush=True)
                await session.abort()
                raise

    print("\nAgent session ended. Review its summary and experiment logs.", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one standalone Copilot research agent."
    )
    parser.add_argument(
        "--experiments", type=int, default=3,
        help="maximum candidate training attempts after the fresh baseline (default: 3)",
    )
    parser.add_argument("--model", help="Copilot model ID; omit to use the SDK default")
    parser.add_argument(
        "--reasoning-effort", choices=get_args(ReasoningEffort),
        help="reasoning level supported by --model; omit for the model default",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run", action="store_true",
        help="print the task without starting Copilot or running experiments",
    )
    mode.add_argument(
        "--list-models", action="store_true",
        help="list account model IDs and reasoning levels, without starting research",
    )
    parser.add_argument(
        "--allow-all-tools", action="store_true",
        help="explicitly approve autonomous file/shell tools; this is not a sandbox",
    )
    args = parser.parse_args(argv)
    if args.experiments < 0:
        parser.error("--experiments must be zero or greater")
    if args.reasoning_effort and not args.model:
        parser.error("--reasoning-effort requires --model; use --list-models to choose one")
    if args.list_models and (args.model or args.reasoning_effort):
        parser.error("--list-models cannot be combined with --model or --reasoning-effort")
    if not args.dry_run and not args.list_models and not args.allow_all_tools:
        parser.error(
            "live runs require --allow-all-tools; use --dry-run to preview the task"
        )

    tag = datetime.now(timezone.utc).strftime("sdk-%Y%m%d-%H%M%S-%f")
    try:
        if not args.dry_run:
            load_environment(REPOSITORY)
        if args.list_models:
            asyncio.run(run_agent("", REPOSITORY, None, 0, list_models=True))
            return 0
        prompt = build_prompt(REPOSITORY, Path(sys.executable), args.experiments, tag)
        if args.dry_run:
            print(f"Model: {args.model or 'SDK default'} | reasoning: {args.reasoning_effort or 'model default'}")
            print(SYSTEM_MESSAGE)
            print(prompt)
            return 0
        print(f"Run: {tag} | baseline + up to {args.experiments} candidate attempts")
        print(f"Model: {args.model or 'SDK default'} | reasoning: {args.reasoning_effort or 'model default'}")
        print("Autonomous file and shell access enabled. Press Ctrl-C to interrupt.", flush=True)
        print(f"Live training output: {REPOSITORY / 'run.log'}", flush=True)
        asyncio.run(run_agent(
            prompt, REPOSITORY, args.model, args.experiments,
            reasoning_effort=args.reasoning_effort,
        ))
    except KeyboardInterrupt:
        print("\nInterrupted. Check for remaining training processes before restarting.", file=sys.stderr)
        return 130
    except (OSError, RuntimeError) as error:
        print(f"\nAgent failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
