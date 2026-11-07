"""
Shared Copilot SDK helpers for the population loop.
Loads credentials, checks models, and runs proposal sessions.
"""

import asyncio
from pathlib import Path
from typing import TextIO, get_args

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

# ---------------------------------------------------------------------------
# Environment and model checks
# ---------------------------------------------------------------------------

def load_environment(repository: Path) -> None:
    path = repository / "agent" / ".env"
    if path.exists():
        with path.open(encoding="utf-8") as source:
            load_dotenv(stream=source, override=False)


async def validate_client(client: CopilotClient, model: str | None,
                          *, reasoning_effort: ReasoningEffort | None = None,
                          list_models: bool = False) -> None:
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


async def list_models(repository: Path) -> None:
    async with CopilotClient(working_directory=str(repository)) as client:
        await validate_client(client, None, list_models=True)

# ---------------------------------------------------------------------------
# Proposal sessions
# ---------------------------------------------------------------------------

async def run_session(client: CopilotClient, prompt: str, repository: Path, model: str | None,
                      *, system_message: str, output: TextIO,
                      reasoning_effort: ReasoningEffort | None = None, timeout: float = 900) -> str:
    async with await client.create_session(
        model=model,
        reasoning_effort=reasoning_effort,
        working_directory=str(repository),
        on_permission_request=PermissionHandler.approve_all,
        # GPT models edit files with apply_patch; Claude models use edit.
        available_tools=ToolSet().add_builtin(["view", "edit", "apply_patch"]),
        system_message={"mode": "append", "content": system_message},
        streaming=True,
        skip_custom_instructions=True,
        enable_config_discovery=False,
        enable_file_hooks=False,
    ) as session:
        print(f"Copilot session: {session.session_id}", file=output, flush=True)
        streamed: set[str] = set()

        def on_event(event: SessionEvent) -> None:
            data = event.data
            if isinstance(data, AssistantMessageDeltaData) and data.delta_content:
                streamed.add(data.message_id)
                print(data.delta_content, end="", file=output, flush=True)
            elif isinstance(data, AssistantMessageData):
                if data.message_id in streamed:
                    streamed.remove(data.message_id)
                    print(file=output, flush=True)
                elif data.content:
                    print(data.content, file=output, flush=True)
            elif isinstance(data, ToolExecutionStartData):
                print(f"\n[tool] {data.tool_name}", file=output, flush=True)
            elif isinstance(data, ToolExecutionCompleteData) and not data.success:
                detail = data.error.message if data.error else "No error detail provided"
                print(f"\n[tool failed] {detail}", file=output, flush=True)
            elif isinstance(data, SessionErrorData):
                print(f"\n[session error] {data.message}", file=output, flush=True)

        session.on(on_event)
        try:
            response = await session.send_and_wait(prompt, timeout=timeout)
            if response is None or not isinstance(response.data, AssistantMessageData):
                raise RuntimeError("Copilot ended without an assistant response.")
            return response.data.content
        except (asyncio.CancelledError, TimeoutError):
            print("\nStopping the Copilot turn...", file=output, flush=True)
            await session.abort()
            raise
