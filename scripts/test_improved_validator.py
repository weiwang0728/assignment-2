"""Small integration smoke test for the improved validator, without human labels."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from validator.runner import Run, write_predictions
from validator.prediction import Prediction
from validator.solution import validate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_ids',nargs='*',default=['001','115','213'])
    parser.add_argument('--out',type=Path,default=ROOT/'output/improved_smoke_predictions.json')
    args=parser.parse_args()
    predictions={}
    for ident in args.run_ids:
        run=Run.load(ROOT/'artifacts'/ident,ROOT)
        print(f'Checking {ident}',flush=True)
        errors=validate(run)
        predictions[ident]=Prediction(run_id=ident,errors=errors)
        write_predictions(args.out,predictions)
        print(predictions[ident].model_dump_json(),flush=True)

if __name__=='__main__':
    main()
