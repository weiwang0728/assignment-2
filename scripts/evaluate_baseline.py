"""Evaluate all seed runs with bounded concurrency and resumable checkpoints."""

import argparse
import importlib
import hashlib
import json
import sys
import time
from threading import Event
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from validator.model import MODEL
from validator.prediction import Prediction, PredictionFile
from validator.runner import runs, write_predictions


def fatal_endpoint_error(error):
    status = getattr(error, 'status_code', None)
    return status in {401, 402, 403, 404} or (status == 429 and any(
        phrase in str(error).lower() for phrase in ('insufficient balance', '余额不足', '1113')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--solution", default="validator.baseline")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    module = importlib.import_module(args.solution)
    validate = module.validate
    source_hash = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    if args.out is None:
        args.out = ROOT / ('output/deepseek_baseline_predictions.json' if args.solution == 'validator.baseline' else 'output/improved_predictions.json')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    all_runs = runs()
    predictions = {}
    if args.out.exists():
        predictions = {p.run_id: p for p in PredictionFile.model_validate_json(args.out.read_text()).predictions}
    metadata = args.out.with_suffix(".metadata.json")
    if metadata.exists():
        previous = json.loads(metadata.read_text())
        if previous['model'] != MODEL or previous['solution'] != args.solution:
            raise SystemExit("Model or solution changed: use a separate --out file.")
        if previous.get('source_hash', source_hash) != source_hash:
            raise SystemExit('Validator source changed: use a separate --out file.')
    elif predictions and args.solution != 'validator.baseline':
        raise SystemExit('Existing predictions have no provenance: use a new --out file.')
    metadata.write_text(json.dumps({"model": MODEL, "solution": args.solution,
                                    "source_hash": source_hash,
                                    "total_runs": len(all_runs), "workers": args.workers}, indent=2) + "\n")
    failures = {}
    stopped = Event()

    def invoke(run):
        if stopped.is_set():
            raise RuntimeError('Not attempted: endpoint circuit breaker is open')
        try:
            return validate(run)
        except Exception as error:
            if fatal_endpoint_error(error):
                stopped.set()
            raise

    for attempt in range(3):
        pending = [r for r in all_runs if r.run_id not in predictions]
        if not pending:
            break
        if attempt:
            time.sleep(30)
        print(f"model={MODEL}, round={attempt + 1}, pending={len(pending)}", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(invoke, run): run for run in pending}
            for future in as_completed(futures):
                if future.cancelled():
                    continue
                run = futures[future]
                try:
                    predictions[run.run_id] = Prediction(run_id=run.run_id, errors=future.result())
                except Exception as error:
                    failures[run.run_id] = f"{type(error).__name__}: {error}"
                    print(f"FAILED {run.run_id}: {failures[run.run_id]}", flush=True)
                    if stopped.is_set():
                        for queued in futures:
                            queued.cancel()
                    continue
                failures.pop(run.run_id, None)
                write_predictions(args.out, predictions)
                print(f"{len(predictions)}/{len(all_runs)} {run.run_id}: "
                      f"{[e.family.value for e in predictions[run.run_id].errors]}", flush=True)
        if stopped.is_set():
            for run in all_runs:
                if run.run_id not in predictions:
                    failures.setdefault(run.run_id, 'Not attempted: endpoint circuit breaker is open')
            break
    args.out.with_suffix(".failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    if failures:
        raise SystemExit(f"{len(failures)} runs failed; rerun to resume.")
    print(f"Complete: {len(predictions)} predictions saved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
