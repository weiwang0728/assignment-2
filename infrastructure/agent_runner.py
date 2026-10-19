"""Run the pinned mini-SWE-agent scaffold in a fresh Modal container."""

from __future__ import annotations

import os
from typing import Any

import modal

MINI_SWE_AGENT_VERSION = "2.4.5"
APP_NAME = "hw2-agent-runner"
FUNCTION_NAME = "run_task"
GATEWAY_APP = "hw2-generation-gateway"
GATEWAY_FUNCTION = "complete"

app = modal.App(APP_NAME)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install(
        f"mini-swe-agent=={MINI_SWE_AGENT_VERSION}",
        "matplotlib==3.10.5",
        "numpy==2.3.2",
        "pandas==2.3.1",
        "pillow==11.3.0",
        "scipy==1.16.1",
        "seaborn==0.13.2",
    )
    .add_local_python_source("workflow")
)


@app.function(image=image, timeout=20 * 60, max_containers=20)
def run_task(request: dict[str, Any]) -> dict[str, Any]:
    from contextlib import nullcontext
    from minisweagent.environments.local import LocalEnvironment, _run
    from workflow.agent_runtime import run_task as run_agent

    class CredentialScrubbedEnvironment(LocalEnvironment):
        """Keep infrastructure credentials out of model-written processes."""

        def execute(
            self,
            action: dict[str, Any],
            cwd: str = "",
            *,
            timeout: int | None = None,
        ):
            command = action.get("command", "")
            cwd = cwd or self.config.cwd or os.getcwd()
            child_env = {
                key: value
                for key, value in (os.environ | self.config.env).items()
                if not any(marker in key for marker in ("TOKEN", "SECRET", "API_KEY"))
            }
            try:
                result = _run(command, cwd, child_env, timeout or self.config.timeout)
                output = {
                    "output": result.stdout,
                    "returncode": result.returncode,
                    "exception_info": "",
                }
            except Exception as error:
                raw_output = getattr(error, "output", None)
                if isinstance(raw_output, bytes):
                    raw_output = raw_output.decode("utf-8", errors="replace")
                output = {
                    "output": raw_output or "",
                    "returncode": -1,
                    "exception_info": (
                        f"An error occurred while executing the command: {error}"
                    ),
                    "extra": {
                        "exception_type": type(error).__name__,
                        "exception": str(error),
                    },
                }
            self._check_finished(output)
            return output

    gateway = modal.Function.from_name(GATEWAY_APP, GATEWAY_FUNCTION)
    return run_agent(
        request,
        gateway.remote,
        lambda **kwargs: nullcontext(CredentialScrubbedEnvironment(**kwargs)),
    )
