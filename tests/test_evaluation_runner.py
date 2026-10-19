import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts import evaluate_baseline as runner


class EndpointError(Exception):
    status_code=402


class EvaluationRunnerTests(unittest.TestCase):
    def test_credit_error_stops_queue_and_preserves_failure_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'predictions.json'
            validate=Mock(side_effect=EndpointError('Insufficient Balance'))
            module=SimpleNamespace(validate=validate,__file__=__file__)
            with patch.object(runner,'runs',return_value=[SimpleNamespace(run_id=str(i)) for i in range(5)]), \
                 patch.object(runner.importlib,'import_module',return_value=module), \
                 patch.object(sys,'argv',['evaluate','--solution','validator.solution','--workers','1','--out',str(output)]):
                with self.assertRaises(SystemExit):
                    runner.main()
            self.assertEqual(validate.call_count,1)
            failures=json.loads(output.with_suffix('.failures.json').read_text())
            self.assertEqual(set(failures),set(map(str,range(5))))
            self.assertFalse(output.exists())

    def test_busy_error_remains_retryable(self):
        error=EndpointError('busy, retry later')
        error.status_code=429
        self.assertFalse(runner.fatal_endpoint_error(error))
        error.args=('余额不足 1113',)
        self.assertTrue(runner.fatal_endpoint_error(error))

if __name__=='__main__':
    unittest.main()
