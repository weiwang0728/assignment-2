import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
from pydantic import ValidationError

from validator import solution as s
from validator.runner import Run


def check(family, status='pass', requirement=None):
    return dict(requirement=requirement or family, family=family, status=status,
                evidence='Verified against the final image and executed code.')


def assessment(**changes):
    result = dict(final_execution_evidence='Final command executed and PNG was decoded.',
                  checks=[check(f) for f in ['wrong_data', 'wrong_chart', 'hard_to_read']],
                  numeric_audits=[])
    result.update(changes)
    return result


def response(value, finish='stop'):
    return {'choices': [{'finish_reason': finish, 'message': {
        'content': value if isinstance(value, str) else json.dumps(value)}}],
        'usage': {'total_tokens': 10}}


class ValidatorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.env = patch.dict(os.environ, {'VALIDATOR_LOG_DIR': str(self.root/'logs')})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.figure = self.root/'figure.png'
        Image.new('RGB', (100, 80), 'white').save(self.figure)
        self.run = Run('synthetic', 'synthetic', 'Draw a chart.', [], self.figure, {})

    def logs(self):
        return [json.loads(line) for file in (self.root/'logs').glob('*.jsonl')
                for line in file.read_text().splitlines()]

    @patch('validator.solution.complete')
    def test_missing_corrupt_and_non_png_are_terminal_without_api(self, api):
        self.figure.unlink()
        self.assertEqual(s.validate(self.run)[0].family.value, 'execution_failure')
        self.figure.write_bytes(b'broken image')
        self.assertEqual(s.validate(self.run)[0].family.value, 'execution_failure')
        Image.new('RGB', (10, 10)).save(self.figure, format='JPEG')
        self.assertEqual(s.validate(self.run)[0].family.value, 'execution_failure')
        api.assert_not_called()

    @patch('validator.solution.complete')
    def test_valid_png_design_errors_are_not_execution_failure(self, api):
        api.return_value = response(assessment(checks=[check('wrong_data'),
            check('wrong_chart', 'fail', 'Need a secondary axis'), check('hard_to_read')]))
        errors = s.validate(self.run)
        self.assertEqual([e.family.value for e in errors], ['wrong_chart'])
        self.assertTrue(any(e['event']=='response' for e in self.logs()))
        self.assertTrue(any(e['event']=='result' for e in self.logs()))
        images = [c for c in api.call_args.args[0][0]['content'] if c['type']=='image_url']
        self.assertEqual(len(images), 5)

    @patch('validator.solution.complete')
    def test_truncated_and_invalid_json_retry_without_silent_pass(self, api):
        api.side_effect = [response('{', 'length'), response('NO'), response(assessment())]
        self.assertEqual(s.validate(self.run), [])
        self.assertEqual(api.call_count, 3)
        self.assertEqual([c.kwargs['max_tokens'] for c in api.call_args_list], [4000,8000,12000])
        self.assertEqual(sum(x['event']=='retry' for x in self.logs()), 2)

    @patch('validator.solution.complete', return_value=response('The'))
    def test_persistent_invalid_response_raises(self, api):
        with self.assertRaises(RuntimeError):
            s.validate(self.run)
        self.assertFalse(any(e['event']=='result' for e in self.logs()))

    def test_schema_rejects_missing_family_invalid_category_extra_fields(self):
        for value in [assessment(checks=[check('wrong_data')]),
                      assessment(checks=[check('execution_failure')]),
                      dict(assessment(), ignored_error=True)]:
            with self.assertRaises(ValidationError):
                s.Assessment.model_validate(value)

    @patch('validator.solution.complete')
    def test_unknown_is_logged_and_does_not_invent_failure(self, api):
        api.return_value = response(assessment(checks=[check(f,'unknown') for f in ['wrong_data','wrong_chart','hard_to_read']]))
        self.assertEqual(s.validate(self.run), [])
        self.assertEqual(self.logs()[-1]['unknown_count'],3)

    def test_safe_calculator_rejects_execution_and_handles_formula_order(self):
        for expression in ["__import__('os').system('touch forbidden')", 'x.__class__', 'x[0]', '2**1000']:
            with self.assertRaises(ValueError):
                s._expression(expression,np.array([1.,2.]))
        np.testing.assert_array_equal(s._expression('(x-4)*(x-6)*(x-8)+90',np.array([4.,6.])),[90,90])
        expected = s.Vector(source='literal',values=[42.1,55.4,61.8,58.2,49.7])
        observed = s.Vector(source='literal',values=[42.1,55.4,61.8,49.7,58.2])
        audit = s.NumericAudit(requirement='Keep original order',expected_basis='Values in task order',observed_basis='Values in final command',expected=expected,observed=observed)
        self.assertEqual(s._calculate([audit],self.run)[0]['status'],'mismatch')

    def test_independent_column_filter_and_median(self):
        path=self.root/'data.csv'
        path.write_text('region,women,men\nNA,10,100\nEU,30,200\nNA,20,300\n')
        run=Run('csv','csv','Use women in NA.',[],self.figure,{'data.csv':path})
        vector=s.Vector(source='column',file='data.csv',column='women',filter_column='region',filter_op='eq',filter_value='NA',operation='median')
        np.testing.assert_array_equal(s._vector(vector,run),[15.])
        wrong=vector.model_copy(update={'column':'men'})
        audit=s.NumericAudit(requirement='Use women',expected_basis='Task specifies women',observed_basis='Final code uses men',expected=vector,observed=wrong)
        self.assertEqual(s._calculate([audit],run)[0]['status'],'mismatch')
        invalid=vector.model_copy(update={'file':'../secret.csv'})
        audit=audit.model_copy(update={'observed':invalid})
        self.assertEqual(s._calculate([audit],run)[0]['status'],'unknown')

    @patch('validator.solution.complete')
    def test_numeric_review_uses_calculation_and_preserves_checklist(self, api):
        audit=dict(requirement='Median',expected_basis='Median from task',observed_basis='Median in executed code',
                   expected=dict(source='literal',values=[1,2,3],operation='median'),
                   observed=dict(source='literal',values=[20]))
        final=[check('wrong_data','fail'),check('wrong_chart'),check('hard_to_read')]
        api.side_effect=[response(assessment(numeric_audits=[audit])),response({'checks':final})]
        self.assertEqual(s.validate(self.run)[0].family.value,'wrong_data')
        self.assertEqual(api.call_count,2)
        logs=self.logs()
        self.assertEqual(next(x for x in logs if x['event']=='calculations')['results'][0]['expected_values'],[2.])

    @patch('validator.solution.complete')
    def test_review_must_not_drop_requirements(self, api):
        audit=dict(requirement='Values',expected_basis='Task literal values',observed_basis='Final code literals',
                   expected=dict(source='literal',values=[1]),observed=dict(source='literal',values=[2]))
        api.side_effect=[response(assessment(numeric_audits=[audit]))]+[response({'checks':[check('wrong_data')]})]*3
        with self.assertRaises(RuntimeError):
            s.validate(self.run)

    def test_contrast_is_diagnostic_and_alpha_composited(self):
        image=Image.new('RGBA',(40,40),(255,255,255,255))
        image.paste((244,244,244,255),(10,10,30,30))
        self.assertGreater(s._contrast(image)['low_contrast_candidate_fraction'],.9)
        image=Image.new('RGBA',(40,40),(0,0,0,0))
        self.assertAlmostEqual(s._contrast(image)['estimated_border_background_luminance'],1.)

    def test_metadata_dimensions_and_missing_dpi_are_not_guessed(self):
        run=Run('metadata','x','Save at 120 dpi with dimensions 1000x1000 pixels.',[],self.figure,{})
        checks=s._metadata_checks(run,s._image(run))
        self.assertEqual([c.status for c in checks],['fail','unknown'])
        Image.new('RGB',(1000,1000)).save(self.figure,dpi=(120,120))
        self.assertEqual([c.status for c in s._metadata_checks(run,s._image(run))],['pass','pass'])

    def test_logs_redact_keys_and_sanitize_filename(self):
        with patch.dict(os.environ,{'TEST_API_KEY':'sensitive-key-value'}):
            run=Run('../../escape','x','x',[],None,{})
            journal=s.Journal(run)
            journal.write('test',response={'content':'sensitive-key-value'})
            self.assertNotIn('sensitive-key-value',journal.path.read_text())
            self.assertEqual(journal.path.parent,self.root/'logs')

    def test_provider_options_do_not_leak_to_other_endpoints(self):
        with patch.dict(os.environ,{'VALIDATOR_BASE_URL':'https://api.deepseek.com','VALIDATOR_THINKING':''}):
            self.assertEqual(s._generation_options(),{})
        with patch.dict(os.environ,{'VALIDATOR_BASE_URL':'https://example.test/v1'},clear=True):
            self.assertEqual(s._generation_options(),{})
        with patch.dict(os.environ,{'VALIDATOR_BASE_URL':'https://api.deepseek.com'},clear=True):
            self.assertEqual(s._generation_options(),{'extra_body':{'thinking':{'type':'disabled'}}})

if __name__=='__main__':
    unittest.main()
