"""Shared pinned agent loop; execution and model transport are injected."""
from __future__ import annotations

import json
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

MINI_SWE_AGENT_VERSION = "2.4.5"
COMMAND_TIMEOUT_SECONDS = 120
STEP_LIMIT = 20
WALL_TIME_LIMIT_SECONDS = 15 * 60
MODEL_MAX_TOKENS = 4096

def _task_prompt(task: dict[str, Any]) -> str:
    expected = task["expected_output"]
    inputs = [item["name"] for item in task.get("inputs", [])]
    return (
        f"TASK ID: {task['task_id']}\n\n"
        f"USER REQUEST:\n{task['instructions']}\n\n"
        f"INPUT FILES: {json.dumps(inputs)}\n"
        f"REQUIRED OUTPUT: create {expected['path']} in the workspace "
        f"as a valid {expected['format']} file.\n"
        "Do not merely describe code: run it and leave the artifact at that "
        "exact path."
    )


def _within_workspace(workspace: Path, relative: str) -> Path:
    candidate = (workspace / relative).resolve()
    if Path(relative).is_absolute() or not candidate.is_relative_to(
        workspace.resolve()
    ):
        raise ValueError(f"unsafe task path: {relative!r}")
    return candidate


def run_task(request: dict[str, Any], complete, environment_factory) -> dict[str, Any]:
    from litellm import ModelResponse
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.config import get_config_from_spec
    from minisweagent.models.litellm_model import LitellmModel
    from minisweagent.models.utils.actions_toolcall import BASH_TOOL

    task = request["task"]
    model_name = request["model"]
    started = time.monotonic()

    class GatewayModel(LitellmModel):
        def _query(self, messages: list[dict[str, Any]], **kwargs: Any):
            generation: dict[str, Any] = {
                "temperature": 0.0,
                "max_tokens": MODEL_MAX_TOKENS,
            }
            if self.config.model_name.startswith("Qwen/Qwen2.5-Coder-"):
                generation["tool_choice"] = "required"
            raw = complete(
                {
                    "model": self.config.model_name,
                    "messages": messages,
                    "tools": [BASH_TOOL],
                    "generation": generation,
                }
            )
            return ModelResponse(**raw)

        def _calculate_cost(self, response):
            return {"cost": 0.0}

    with ExitStack() as stack:
        raw_workspace = stack.enter_context(tempfile.TemporaryDirectory(prefix="agent-eval-"))
        workspace = Path(raw_workspace)
        (workspace / "task.json").write_text(json.dumps(task, indent=2) + "\n")
        for name, content in request.get("input_files", {}).items():
            target = _within_workspace(workspace, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)

        baseline = get_config_from_spec("mini")
        model = GatewayModel(
            model_name=model_name,
            model_kwargs={},
            cost_tracking="ignore_errors",
            observation_template=baseline["model"]["observation_template"],
            format_error_template=baseline["model"]["format_error_template"],
        )
        environment = stack.enter_context(environment_factory(
            cwd=str(workspace),
            timeout=COMMAND_TIMEOUT_SECONDS,
            env=baseline["environment"]["env"]
            | {"MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1"},
        ))
        agent = DefaultAgent(
            model,
            environment,
            system_template=baseline["agent"]["system_template"],
            instance_template=baseline["agent"]["instance_template"],
            step_limit=STEP_LIMIT,
            cost_limit=0,
            wall_time_limit_seconds=WALL_TIME_LIMIT_SECONDS,
        )
        outcome = agent.run(_task_prompt(task))

        if hasattr(environment, "stop"):
            environment.stop()

        relative_artifact = Path(task["expected_output"]["path"])
        artifact = _within_workspace(workspace, str(relative_artifact))
        artifact_bytes = artifact.read_bytes() if artifact.is_file() else None
        artifact_error = None
        if artifact_bytes and task["expected_output"]["format"].lower() == "png":
            try:
                from PIL import Image

                with Image.open(artifact) as rendered:
                    rendered.verify()
            except Exception as error:
                artifact_error = f"Invalid PNG: {error}"

        return {
            "task_id": task["task_id"],
            "task": task,
            "model": model_name,
            "scaffold": f"mini-swe-agent=={MINI_SWE_AGENT_VERSION}",
            "outcome": outcome,
            "trajectory": agent.serialize(),
            "artifact_path": str(relative_artifact),
            "artifact_present": artifact_bytes is not None,
            "artifact_error": artifact_error,
            "artifact_bytes": artifact_bytes,
            "elapsed_seconds": time.monotonic() - started,
        }
