"""Judge the GLM smoke run with the validator configured in .env."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from validator.baseline import validate
from validator.model import MODEL
from validator.prediction import Prediction, PredictionFile
from validator.runner import Run


def main():
    folder = ROOT / "output/glm_baseline_smoke"
    record = json.loads((folder / "result.json").read_text())
    task = record["task"]
    run = Run(
        run_id="glm_baseline_smoke", task_id=task["task_id"],
        instructions=task["instructions"], messages=record["trajectory"]["messages"],
        figure=folder / "figure.png",
        inputs={item["name"]: ROOT / item["path"] for item in task.get("inputs", [])},
    )
    print(f"Validator model: {MODEL}; task: {run.task_id}", flush=True)
    errors = validate(run)
    output = folder / "validator_predictions.json"
    output.write_text(PredictionFile(predictions=[
        Prediction(run_id=run.run_id, errors=errors)
    ]).model_dump_json(indent=2) + "\n")
    print(output.read_text(), flush=True)
    print(f"Saved: {output}", flush=True)


if __name__ == "__main__":
    main()
