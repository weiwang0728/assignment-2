"""Run one released task through the agent baseline using the configured GLM API."""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from dotenv import load_dotenv
from openai import OpenAI

from workflow.agent_runtime import run_task
from workflow.docker_runner import DockerEnvironment


def main():
    load_dotenv(ROOT / ".env")
    base = os.environ["GENERATION_GLM_BASE_URL"].rstrip("/")
    key = os.getenv("GENERATION_GLM_API_KEY") or os.environ["GENERATION_API_KEY"]
    api_model = os.getenv("GENERATION_GLM_MODEL", "glm-4.7-flash")
    source = sorted((ROOT / "artifacts").glob("*/result.json"))[0]
    record = json.loads(source.read_text())
    task = {**record["task"], "task_id": record["task_id"],
            "expected_output": {"path": "figure.png", "format": "png"}}
    client = OpenAI(base_url=base, api_key=key, timeout=300, max_retries=1)
    calls = 0

    def complete(request):
        nonlocal calls
        calls += 1
        print(f"GLM request {calls}: {api_model}", flush=True)
        response = client.chat.completions.create(
            model=api_model, messages=request["messages"], tools=request["tools"],
            **request["generation"],
        )
        return response.model_dump()

    destination = ROOT / "output" / "glm_baseline_smoke"
    destination.mkdir(parents=True, exist_ok=True)
    print(f"Task {task['task_id']} from {source.parent.name}", flush=True)
    result = run_task({
        "model": "zai-org/GLM-4.7-Flash", "task": task,
        "input_files": {item["name"]: (ROOT / item["path"]).read_bytes()
                        for item in task.get("inputs", [])},
    }, complete, lambda **kwargs: DockerEnvironment(image="assignment-agent:local", **kwargs))
    artifact = result.pop("artifact_bytes", None)
    if artifact:
        (destination / "figure.png").write_bytes(artifact)
    result.update({"task": task, "source_run": source.parent.name,
                   "api_model": api_model, "api_calls": calls})
    (destination / "result.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps({k: result.get(k) for k in
                     ("outcome", "artifact_present", "artifact_error", "elapsed_seconds", "api_calls")}), flush=True)
    print(f"Saved to {destination}", flush=True)
    if not result.get("artifact_present"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
