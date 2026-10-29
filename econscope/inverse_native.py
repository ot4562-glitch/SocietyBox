"""Reviewed inverse screening with a real pinned native solve of selected OLS.

The existing exact screen selects on training only. Original HiGHS solves the
equivalent training-MSE QP; it does not pick hypotheses or establish causality.
Finite per-coefficient bounds must contain the unconstrained exact OLS solution.
"""
from fractions import Fraction as F
import csv
import io
import math
from pathlib import Path

from . import highs_native
from .research_evidence import EvidenceLedger, make_evidence, make_obligation, evaluate_coverage
from .workspace import canonical, digest

COEFFICIENT_ABS_TOL = 1e-6
COEFFICIENT_REL_TOL = 1e-7
MSE_ABS_TOL = 1e-6
MSE_REL_TOL = 1e-7


def _source_receipt():
    from .inverse_investigation import _code_receipt
    receipt = _code_receipt()
    for name in ('inverse_native.py', 'highs_native.py', 'highs_native_worker.py'):
        path = Path(__file__).with_name(name)
        if path.is_symlink() or not path.is_file():
            raise ValueError('INVERSE_NATIVE_CODE_UNAVAILABLE')
        receipt[name] = digest(path.read_bytes())
    return receipt


def _label(feature):
    return feature['series'] + '[t-' + str(feature['lag']) + ']'


def validate(parameters):
    """Reuse the full existing rights/sample/release validator, without fitting."""
    from .inverse_screen import _validate_request, _single_candidates
    from .mechanism_dynamic import _validate, MAX_BYTES
    if type(parameters) is not dict or set(parameters) != {'screen', 'coefficient_bounds'}:
        raise ValueError('INVERSE_NATIVE_PARAMETERS_INVALID')
    payload = parameters['screen']
    data, lags = _validate_request(payload)
    singles = _single_candidates(data, lags, payload['search']['include_target_history'])
    if len(singles) > 24:
        raise ValueError('INVERSE_NATIVE_SEARCH_BUDGET')
    anchor = {'candidate_id': 'window_anchor', 'features': [{'series': singles[0]['features'][0]['series'], 'lag': max(lags)}]}
    _validate(dict(data, candidates=singles + [anchor]), candidate_limit=25, input_size_cap=MAX_BYTES+8192)
    bounds = parameters['coefficient_bounds']
    required = {'intercept'} | {_label(candidate['features'][0]) for candidate in singles}
    if type(bounds) is not dict or set(bounds) != required:
        raise ValueError('INVERSE_NATIVE_COEFFICIENT_BOUNDS_REQUIRED')
    for pair in bounds.values():
        if type(pair) is not list or len(pair) != 2:
            raise ValueError('INVERSE_NATIVE_COEFFICIENT_BOUNDS_INVALID')
        low, high = map(highs_native._num, pair)
        if low >= high:
            raise ValueError('INVERSE_NATIVE_COEFFICIENT_BOUNDS_INVALID')


def _program(parameters, screen):
    data = parameters['screen']['study']
    rows = data['observations']
    features = screen['selected_features']
    labels = ['intercept'] + [_label(feature) for feature in features]
    start = screen['common_max_lag']
    end = next(i for i, row in enumerate(rows) if row['day'] == data['training_end_day'])
    design = [[F(1)] + [F(rows[t-f['lag']]['values'][f['series']]) for f in features]
              for t in range(start, end+1)]
    target = [F(rows[t]['values'][data['target']]) for t in range(start, end+1)]
    beta = list(map(F, screen['selected_coefficients_exact']))
    low = [highs_native._num(parameters['coefficient_bounds'][label][0]) for label in labels]
    high = [highs_native._num(parameters['coefficient_bounds'][label][1]) for label in labels]
    program = _make_ols_program(design, target, beta, low, high)
    return program, labels, design, target


def _make_ols_program(design, target, beta, low, high, *, weights=None):
    """Shared reviewed training-MSE QP; never substitutes a bounded optimum."""
    n, p = len(design), len(beta)
    weights = [F(1)]*n if weights is None else weights
    if len(weights) != n or any(type(w) is not F or w <= 0 for w in weights):
        raise ValueError('INVERSE_NATIVE_WEIGHTS_INVALID')
    mass = sum(weights)
    if any(not lower <= coefficient <= upper for lower, coefficient, upper in zip(low, beta, high)):
        raise ValueError('INVERSE_NATIVE_BOUNDS_EXCLUDE_OLS')
    program = dict(kind='QP', sense='minimize',
        objective=[str(-2*sum(w*row[j]*y for row, y, w in zip(design, target, weights))/mass) for j in range(p)],
        constraints=[], row_lower=[], row_upper=[], lower=list(map(str, low)), upper=list(map(str, high)),
        integrality=[0]*p, offset=str(sum(w*y*y for y,w in zip(target,weights))/mass),
        hessian=[[str(2*sum(w*row[i]*row[j] for row,w in zip(design,weights))/mass) for j in range(p)] for i in range(p)])
    # Native adapter's existing numeric/strict-curvature/resource limits apply.
    highs_native.validate(program)
    return program


def _compare(screen, native, design, target):
    actual = native['actual']
    beta_exact = list(map(F, screen['selected_coefficients_exact']))
    beta = actual['solution']
    if (actual['status'] != 'OPTIMAL' or type(beta) is not list or len(beta) != len(beta_exact)
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in beta)):
        raise ValueError('INVERSE_NATIVE_REGRESSION_PARITY_FAILED')
    errors = [abs(float(expected)-v) for expected, v in zip(beta_exact, beta)]
    if any(error > COEFFICIENT_ABS_TOL + COEFFICIENT_REL_TOL*abs(float(expected))
           for error, expected in zip(errors, beta_exact)):
        raise ValueError('INVERSE_NATIVE_REGRESSION_PARITY_FAILED')
    chosen = next(c for c in screen['candidates'] if c['candidate_id'] == screen['selected_candidate_id'])
    exact_mse = F(chosen['training_sse_exact'])/len(design)
    # Re-evaluate the original observations with returned binary64 coefficients,
    # independently of the QP objective and native adapter's KKT/vertex oracle.
    numeric_beta = [F(str(v)) for v in beta]
    direct_mse = sum((y-sum(v*b for v, b in zip(row, numeric_beta)))**2
                     for row, y in zip(design, target))/len(design)
    direct_error = abs(float(direct_mse-exact_mse))
    objective_error = abs(actual['objective']-float(exact_mse))
    limit = MSE_ABS_TOL + MSE_REL_TOL*abs(float(exact_mse))
    if not math.isfinite(objective_error) or direct_error > limit or objective_error > limit:
        raise ValueError('INVERSE_NATIVE_REGRESSION_PARITY_FAILED')
    return dict(status='WITHIN_DECLARED_REGRESSION_TOLERANCE',
        exact_training_mse=str(exact_mse), native_original_data_mse_decimal=str(float(direct_mse)),
        maximum_coefficient_error=max(errors), original_data_mse_error=direct_error,
        native_objective_error=objective_error,
        coefficient_abs_tolerance=COEFFICIENT_ABS_TOL, coefficient_rel_tolerance=COEFFICIENT_REL_TOL,
        mse_abs_tolerance=MSE_ABS_TOL, mse_rel_tolerance=MSE_REL_TOL,
        interpretation='SELECTED_UNCONSTRAINED_OLS_NUMERICAL_PARITY_NOT_CAUSAL_IDENTIFICATION')


def _native_evidence(screen, native, program, comparison, code_receipt):
    original = screen['investigation']
    ledger = EvidenceLedger(original['evidence'])
    calculated = next(rec for rec in original['evidence'] if rec['operation'] == 'EXACT_LAGGED_SCREEN')
    scope = dict(calculated['scope'], native_selected_candidate_id=screen['selected_candidate_id'])
    subject = screen['input_sha256']
    blockers = [] if native else ['NO_FULL_RANK_SELECTED_MODEL']
    obligation = make_obligation(claim_id='selected_regression_native_parity', subject_sha256=subject,
        scope=scope, required_authority='PINNED_UPSTREAM_NUMERIC_EXECUTION', required_proof_kind='SELECTED_OLS_NATIVE_PARITY',
        verifier_id='inverse.native', dependencies=[], unknown_ids=[], blocking_reasons=blockers,
        next_probe='Inspect original native API/runtime and coefficient/data-level errors; external replication remains separate')
    if native:
        producer = dict(tool_id='inverse.native', version='1', code_sha256=digest(canonical(code_receipt)))
        computed = make_evidence(subject_sha256=subject, producer=producer,
            operation='ORIGINAL_HIGHS_SELECTED_REGRESSION', authority='PINNED_UPSTREAM_NUMERIC_EXECUTION',
            epistemic_state='DERIVED', proof_kind='SELECTED_OLS_NATIVE_RESULT', scope=scope,
            coverage=dict(status='PARTIAL', unknown_dimensions=['unselected_native_models', 'causal_identification', 'external_replication']),
            result=dict(native=native, derived_program=program, comparison=comparison),
            parent_evidence_ids=[calculated['evidence_id']],
            limitations=['Only selected OLS is executed by original HiGHS; all candidates have exact Cramer checks',
                         'Same-source local execution is not independent clean installation or empirical identification'])
        checked = make_evidence(subject_sha256=subject, producer=producer,
            operation='NATIVE_SELECTED_OLS_PARITY_CHECK', authority='PINNED_UPSTREAM_NUMERIC_EXECUTION',
            epistemic_state='DERIVED', proof_kind='SELECTED_OLS_NATIVE_PARITY', scope=scope,
            coverage=dict(status='COMPLETE', unknown_dimensions=[]),
            result=dict(obligation_id=obligation['obligation_id'], revision=1, previous_verification_id=None, outcome='PASS'),
            parent_evidence_ids=[computed['evidence_id']], limitations=['Numerical parity for this selected model only'])
        ledger.merge([computed, checked])
    return ledger.records(), obligation, evaluate_coverage([obligation], ledger)


def run(parameters, runtime):
    validate(parameters)
    code_receipt = _source_receipt()
    from .inverse_screen import screen_inverse_hypotheses
    screen = screen_inverse_hypotheses(parameters['screen'])
    native = program = comparison = None
    labels = []
    if screen['selected_candidate_id'] is not None:
        program, labels, design, target = _program(parameters, screen)
        native = highs_native.run(program, runtime)
        comparison = _compare(screen, native, design, target)
    if code_receipt != _source_receipt():
        raise ValueError('INVERSE_NATIVE_CODE_CHANGED_DURING_RUN')
    evidence, obligation, coverage = _native_evidence(screen, native, program, comparison, code_receipt)
    return dict(engine='EXACT_INVERSE_SCREEN_WITH_PINNED_HIGHS_SELECTED_OLS', mode='NO_LLM',
        native_execution='EXECUTED' if native else 'NOT_RUN',
        native_parity='WITHIN_DECLARED_REGRESSION_TOLERANCE' if native else 'NOT_RUN',
        native_blocker=None if native else 'NO_FULL_RANK_SELECTED_MODEL',
        screen=screen, parameter_order=labels, derived_program=program, native=native, comparison=comparison,
        code_files=code_receipt,
        evidence=evidence, native_obligation=obligation, native_coverage=coverage,
        scientific_acceptance='NOT_ESTABLISHED', os_isolation='NOT_ATTESTED',
        assumptions=['Researcher-declared source, units, releases and rights are unverified',
                     'Training-only exploratory hypothesis selection, uncorrected multiple comparisons',
                     'Native QP solves selected unconstrained OLS only when declared bounds contain the exact solution',
                     'No causal, authenticated historical-as-of, future forecast or external replication acceptance'])


def export_artifacts(parameters, result, chart):
    """Pure deterministic exports of a just-computed result, never disk reads."""
    screen = result['screen']
    files = {}
    def csv_blob(header, rows):
        stream = io.StringIO(newline='')
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)
        return stream.getvalue().encode('utf-8')
    def unit_text(value):
        # Units are arbitrary researcher strings; keep spreadsheet text inert.
        return "'"+value if value.lstrip()[:1] in ('=', '+', '-', '@') else value
    data = parameters['screen']['study']
    native_beta = [] if result['native'] is None else result['native']['actual']['solution']
    units = [data['series_units'][data['target']]] + [
        data['series_units'][data['target']]+'/'+data['series_units'][feature['series']]
        for feature in screen['selected_features']]
    coefficient_rows = [[name, exact, str(native), unit_text(unit), 'NOT_ESTABLISHED']
        for name, exact, native, unit in zip(result['parameter_order'], screen['selected_coefficients_exact'] or [], native_beta, units)]
    files['coefficients.csv'] = csv_blob(['parameter', 'value_exact', 'native_value_decimal', 'unit', 'causal_identification'], coefficient_rows)
    files['values.csv'] = csv_blob(['variable', 'value_exact', 'unit', 'source_id'],
        [[label, value, unit_text(chart['unit']), unit_text(chart['source_id'])]
         for label, value in zip(chart['labels'], chart['values_exact'])])
    files['candidates.csv'] = csv_blob(
        ['candidate_id', 'features', 'status', 'training_sse_exact', 'training_score_exact', 'training_degrees_of_freedom', 'selected', 'holdout_used_for_selection'],
        [[c['candidate_id'], ';'.join(_label(f) for f in c['features']), c['status'], c.get('training_sse_exact', ''),
          c.get('training_score_exact', ''), c.get('training_degrees_of_freedom', ''),
          c['candidate_id'] == screen['selected_candidate_id'], False] for c in screen['candidates']])
    holdout = [] if screen['holdout'] is None else screen['holdout']['rows']
    files['holdout.csv'] = csv_blob(['origin_day', 'target_day', 'prediction_exact', 'observed_exact', 'residual_exact', 'historical_release_authentication'],
        [[cell[field] for field in ('origin_day', 'target_day', 'prediction_exact', 'observed_exact', 'residual_exact')]+['NOT_ESTABLISHED'] for cell in holdout])
    files['investigation.json'] = canonical(dict(investigation=screen['investigation'],
        native_evidence=result['evidence'], native_obligation=result['native_obligation'], native_coverage=result['native_coverage']))
    return files
