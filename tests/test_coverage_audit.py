"""Ensure static coverage discoveries are never misreported as verification."""
from pathlib import Path
import tempfile
import unittest

from econscope.coverage_audit import audit_coverage


class CoverageAuditTests(unittest.TestCase):
    def test_field_denominator_is_explicit_and_all_rows_are_unverified(self):
        with tempfile.TemporaryDirectory() as root:
            report=audit_coverage(package_dir=Path(root),tests_dir=Path(root))
        self.assertEqual(report['fields_tracked'],30)
        self.assertEqual([r['issue_number'] for r in report['fields']],list(range(11,41)))
        self.assertEqual(report['denominator_definition'],'11_METHOD_WORKSTREAMS_PLUS_19_ECONOMICS_FIELDS')
        self.assertEqual(report['coverage_percentage'],'NOT_CALCULATED_SCIENTIFIC_ACCEPTANCE_REQUIRED')
        self.assertTrue(all(r['field_completion']=='NOT_ASSESSED' for r in report['fields']))
        self.assertTrue(all(r['original_api_execution']=='NOT_RUN_BY_THIS_AUDIT' for r in report['fields']))
        self.assertTrue(all(r['independent_test_execution']=='NOT_RUN_BY_THIS_AUDIT' for r in report['fields']))

    def test_candidate_requires_actual_top_level_function_declaration(self):
        with tempfile.TemporaryDirectory() as root:
            folder=Path(root); (folder/'generalized_linear.py').write_text('def fit(x):\n    return x\n')
            (folder/'test_generalized_linear.py').write_text('pass\n')
            report=audit_coverage(package_dir=folder,tests_dir=folder)
            first=report['fields'][0]
            self.assertEqual(first['source_status'],'FUNCTION_DECLARED_NOT_EXECUTED')
            self.assertEqual(first['test_status'],'TEST_FILE_PRESENT_NOT_EXECUTED')
            self.assertEqual(report['source_declarations_found'],1)
            self.assertEqual(report['test_files_located'],1)
            (folder/'generalized_linear.py').write_text('class F:\n    def fit(self): pass\n')
            first=audit_coverage(package_dir=folder,tests_dir=folder)['fields'][0]
            self.assertEqual(first['source_status'],'FUNCTION_NOT_DECLARED')

    def test_strict_mode_fails_missing_candidate_but_not_untracked_fields(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError,'COVERAGE_TRACKED_CALLABLE_UNAVAILABLE:econometrics'):
                audit_coverage(package_dir=Path(root),tests_dir=Path(root),strict=True)
            report=audit_coverage(package_dir=Path(root),tests_dir=Path(root))
        field15=next(r for r in report['fields'] if r['issue_number']==15)
        self.assertEqual(field15['source_status'],'NO_TRACKED_PUBLIC_EXEMPLAR')
        self.assertIsNone(field15['python_target'])

    def test_malformed_and_symlinked_sources_are_not_followed(self):
        with tempfile.TemporaryDirectory() as root:
            folder=Path(root)
            file=folder/'generalized_linear.py'
            file.write_text('def fit(\n')
            self.assertEqual(audit_coverage(package_dir=folder)['fields'][0]['source_status'],
                             'SOURCE_PARSE_FAILED')
            file.unlink()
            original=folder/'not_public.py';original.write_text('def fit(): pass\n')
            try:
                file.symlink_to(original)
            except (OSError,NotImplementedError):
                self.skipTest('symlinks unavailable on this platform')
            self.assertEqual(audit_coverage(package_dir=folder)['fields'][0]['source_status'],
                             'SOURCE_UNAVAILABLE')


if __name__=='__main__': unittest.main()
