"""Host-side agent orchestration with commands isolated in a local Docker daemon."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

from dotenv import load_dotenv


def _docker(*args: str, timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", *args], capture_output=True, text=True, timeout=timeout,
        check=True,
    )


def prepare_runner(image: str, models: list[str]):
    """Fail before starting jobs if dependencies, endpoints or Docker are missing."""
    load_dotenv()
    try:
        from importlib.metadata import version
        from minisweagent.environments.local import LocalEnvironment  # noqa: F401
        import requests  # noqa: F401
        if version("mini-swe-agent") != "2.4.5":
            raise RuntimeError("Docker backend requires mini-swe-agent==2.4.5")
    except ImportError as error:
        raise RuntimeError("Install Docker backend dependencies: uv sync --extra docker") from error

    from .generate import AGENT_BY_KEY
    from .generation_client import _endpoint

    # The JSON registry allows a different API key per model. Otherwise use .env.
    if not os.getenv("GENERATION_ENDPOINTS_JSON"):
        registry = {}
        for key in models:
            agent = AGENT_BY_KEY[key]
            base = os.getenv(agent.url_variable)
            key_variable = f"GENERATION_{key.upper()}_API_KEY"
            api_key = os.getenv(key_variable) or os.getenv("GENERATION_API_KEY")
            if not base or not api_key:
                raise ValueError(f"Set {agent.url_variable} and {key_variable} (or GENERATION_API_KEY)")
            registry[agent.model] = {"api_base": base, "api_key": api_key}
        os.environ["GENERATION_ENDPOINTS_JSON"] = json.dumps(registry)
    for key in models:
        endpoint = _endpoint(AGENT_BY_KEY[key].model)
        if not endpoint["api_base"] or not endpoint["api_key"]:
            raise ValueError(f"Incomplete endpoint settings for {key}")

    if not shutil.which("docker"):
        raise RuntimeError("Docker CLI is unavailable; install Docker and start its daemon")
    try:
        _docker("info", "--format", "{{.ServerVersion}}")
        _docker("image", "inspect", image)
    except (subprocess.SubprocessError, OSError) as error:
        raise RuntimeError(
            f"Docker daemon or image {image!r} is unavailable. Build with: "
            "docker build -f infrastructure/Dockerfile.agent -t assignment-agent:local ."
        ) from error

    def run(request: dict[str, Any]) -> dict[str, Any]:
        from .agent_runtime import run_task
        from .generation_client import complete

        return run_task(
            request, complete,
            lambda **kwargs: DockerEnvironment(image=image, **kwargs),
        )

    return run


# Only imported when the optional Docker backend is selected.
from minisweagent.environments.local import LocalEnvironment


class DockerEnvironment(LocalEnvironment):
    """Preserve mini's finish detection while executing every action in Docker."""

    def __init__(self, *, image: str, cwd: str, timeout: int, env: dict[str, str]):
        super().__init__(cwd="/workspace", timeout=timeout, env=env)
        self.image = image
        self.workspace = Path(cwd).resolve()
        self.container = f"hw2-agent-{uuid.uuid4().hex}"
        self.active = False

    def __enter__(self):
        args = [
            "run", "--detach", "--pull", "never", "--name", self.container,
            "--network", "none", "--cpus", "2", "--memory", "4g",
            "--pids-limit", "256", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--user", f"{os.getuid()}:{os.getgid()}",
            "--workdir", "/workspace",
            "--mount", f"type=bind,src={self.workspace},dst=/workspace",
            "--env", "HOME=/tmp",
        ]
        for key, value in self.config.env.items():
            if not any(marker in key.upper() for marker in ("TOKEN", "SECRET", "API_KEY")):
                args.extend(["--env", f"{key}={value}"])
        args.extend([self.image, "sleep", "infinity"])
        try:
            _docker(*args, timeout=60)
            self.active = True
        except BaseException:
            # The daemon may have created the container before the CLI timed out.
            try:
                _docker("rm", "--force", self.container)
            except (OSError, subprocess.SubprocessError):
                pass
            raise
        return self

    def get_template_vars(self, **kwargs):
        # Do not expose host environment variables to agent templates. The runtime
        # is Linux even when the controller runs on macOS with Docker Desktop.
        return self.config.model_dump() | {
            "system": "Linux", "release": "container", "version": "", "machine": "",
        } | kwargs

    def stop(self):
        if self.active:
            _docker("rm", "--force", self.container)
            self.active = False

    def __exit__(self, exc_type, exc, traceback):
        self.stop()

    def execute(self, action: dict[str, Any], cwd: str = "", *, timeout: int | None = None):
        if not self.active:
            raise RuntimeError("Docker task container is no longer running")
        cwd = cwd or "/workspace"
        path = PurePosixPath(cwd)
        if not path.is_absolute() or ".." in path.parts or not path.is_relative_to("/workspace"):
            raise ValueError(f"Command cwd must be inside /workspace: {cwd!r}")
        try:
            result = subprocess.run(
                ["docker", "exec", "--workdir", cwd, self.container,
                 "bash", "-lc", action.get("command", "")],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                errors="replace", timeout=timeout or self.config.timeout,
            )
        except subprocess.TimeoutExpired as error:
            # Killing only docker exec leaves its child running inside the container.
            self.stop()
            raise RuntimeError("Agent command timed out; task container was destroyed") from error
        output = {"output": result.stdout, "returncode": result.returncode, "exception_info": ""}
        self._check_finished(output)
        return output
