"""Read-only, evidence-conservative audit of tracked v1 economics field exemplars.

Discovering a Python function declaration and a test file is not executing the
function, validating the test, reproducing external sources, or proving an
empirical result. This bounded inventory tracks one *representative candidate*
per issue, not exhaustive support within a field or a completion percentage.
"""
import ast
import hashlib
import json
from pathlib import Path


# Issue-linked, change-controlled v1 scope: 11 cross-cutting methodological
# workstreams and 19 economics fields. A None candidate means *no tracked
# exemplar yet*, not a claim that no implementation exists anywhere.
# (issue_number, field_id, representative_method, Python target, test file)
_FIELDS = (
    (11, 'econometrics', 'generalized linear regression', 'generalized_linear.fit', 'test_generalized_linear.py'),
    (12, 'iv_gmm', 'linear instrumental-variables estimation', 'instrumental_variables.calculate', None),
    (13, 'panel', 'dynamic panel GMM', 'dynamic_panel.calculate', None),
    (14, 'causal_did', 'staggered difference in differences', None, None),
    (15, 'causal_design', 'regression discontinuity and synthetic control', None, None),
    (16, 'time_series', 'vector autoregression', 'vector_autoregression.calculate', 'test_vector_autoregression.py'),
    (17, 'state_space', 'linear Gaussian filtering', 'state_space.filter', 'test_state_space.py'),
    (18, 'structural_macro', 'DSGE equilibrium solution', None, None),
    (19, 'dynamic_economics', 'heterogeneous-agent Bellman model', None, None),
    (20, 'bayesian_flexible', 'Bayesian and semiparametric inference', None, None),
    (21, 'structural_estimation', 'simulated method of moments', None, None),
    (22, 'micro_theory', 'consumer and producer optimization', None, None),
    (23, 'industrial_organization', 'structural demand estimation', None, None),
    (24, 'games_market_design', 'game equilibrium and matching', None, None),
    (25, 'international', 'gravity trade and tariff equilibrium', None, None),
    (26, 'financial', 'asset pricing and portfolio risk', None, None),
    (27, 'monetary_banking', 'monetary transmission and credit', None, None),
    (28, 'public_economics', 'tax and public finance simulation', None, None),
    (29, 'labor_education', 'labor search and wage policy', None, None),
    (30, 'development', 'development economics program evaluation', None, None),
    (31, 'health', 'health economics cost effectiveness', None, None),
    (32, 'environment_energy', 'environment and power market simulation', None, None),
    (33, 'urban_spatial', 'housing and regional economics', None, None),
    (34, 'distribution_demography', 'inequality and mobility analysis', None, None),
    (35, 'input_output_cge', 'input-output impact', 'input_output_impact.calculate', 'test_input_output_impact.py'),
    (36, 'economic_networks', 'minimum-cost supply network', 'network_cost_flow.run', None),
    (37, 'survey_measurement', 'replicate-weight survey variance', 'survey_replicate_variance.calculate', 'test_survey_replicate_variance.py'),
    (38, 'behavioral_experimental', 'intertemporal budget analysis', 'macro_solver.check_consumption', None),
    (39, 'agriculture_resources', 'farm and resource optimization', None, None),
    (40, 'institutions_history', 'historical institutional analysis', None, None),
)


def _safe_file(path):
    """Read bounded regular public-source files without following symlinks."""
    if path.is_symlink() or not path.is_file():
        return None
    if path.stat().st_size > 1_000_000:
        return None
    with path.open('rb') as stream:
        return stream.read(1_000_001)


def audit_coverage(*, package_dir=None, tests_dir=None, strict=False):
    """Audit source declarations; never import or execute the candidate APIs.

    Strict mode raises if an explicitly tracked candidate is absent/malformed.
    It never mistakes a source declaration or present test file for a pass.
    """
    package = Path(package_dir) if package_dir is not None else Path(__file__).parent
    tests = Path(tests_dir) if tests_dir is not None else package.parent / 'tests'
    rows = []
    for number, field, method, candidate, test in _FIELDS:
        code_state='NO_TRACKED_PUBLIC_EXEMPLAR'
        source_sha=None
        if candidate is not None:
            module,func=candidate.rsplit('.',1)
            raw=_safe_file(package / (module+'.py'))
            if raw is None:
                code_state='SOURCE_UNAVAILABLE'
            else:
                source_sha=hashlib.sha256(raw).hexdigest()
                try:
                    syntax=ast.parse(raw,filename=module+'.py')
                except (SyntaxError,UnicodeDecodeError,ValueError):
                    code_state='SOURCE_PARSE_FAILED'
                else:
                    declared=any(isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))
                                 and node.name==func for node in syntax.body)
                    code_state='FUNCTION_DECLARED_NOT_EXECUTED' if declared else 'FUNCTION_NOT_DECLARED'
        test_state='NO_TRACKED_PUBLIC_TEST'
        if test is not None:
            raw_test=_safe_file(tests / test)
            test_state='TEST_FILE_PRESENT_NOT_EXECUTED' if raw_test is not None else 'TEST_FILE_UNAVAILABLE'
        row=dict(issue_number=number,field=field,representative_method=method,
                 python_target=None if candidate is None else 'econscope.'+candidate,
                 source_status=code_state,source_sha256=source_sha,
                 public_test_file=test,test_status=test_state,
                 original_api_execution='NOT_RUN_BY_THIS_AUDIT',
                 independent_test_execution='NOT_RUN_BY_THIS_AUDIT',
                 empirical_validation='NOT_ESTABLISHED_BY_THIS_AUDIT',
                 causal_identification='NOT_ESTABLISHED_BY_THIS_AUDIT',
                 field_completion='NOT_ASSESSED')
        if strict and candidate is not None and code_state!='FUNCTION_DECLARED_NOT_EXECUTED':
            raise ValueError('COVERAGE_TRACKED_CALLABLE_UNAVAILABLE:'+field)
        rows.append(row)
    return dict(schema='societybox-v1-coverage-declaration-audit-v1',
                status='STATIC_DISCOVERY_ONLY_NO_EXECUTION_OR_FIELD_COMPLETION',
                denominator_definition='11_METHOD_WORKSTREAMS_PLUS_19_ECONOMICS_FIELDS',
                fields_tracked=len(rows),
                source_declarations_found=sum(r['source_status']=='FUNCTION_DECLARED_NOT_EXECUTED' for r in rows),
                test_files_located=sum(r['test_status']=='TEST_FILE_PRESENT_NOT_EXECUTED' for r in rows),
                coverage_percentage='NOT_CALCULATED_SCIENTIFIC_ACCEPTANCE_REQUIRED',
                fields=rows)


def main():
    print(json.dumps(audit_coverage(strict=True),sort_keys=True,separators=(',',':')))


if __name__=='__main__':
    main()
