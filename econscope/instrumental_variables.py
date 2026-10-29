"""Independent exact 2SLS/GMM, classical AR, and pinned HiGHS IV checks.

All intercepts and exogenous instruments are explicit. Positive analytic,
frequency and survey weights are deliberately unsupported. Structural errors,
not second-stage OLS errors, define covariance. No n-by-n projection is built.
Rational elimination is a small-problem reference, not a large-data QR solver.
"""
from fractions import Fraction as F
from decimal import Decimal, localcontext
from math import isqrt
from pathlib import Path
import math
import re

from .execution_control import checkpoint
from .highs_native import _num
from .mechanism_dynamic import _dot, _solve
from .regression_sensitivity import _covariances, _multiply, _bartlett_meat
from .research_evidence import (EvidenceLedger, UnknownRegistry, make_evidence,
                                make_obligation, evaluate_coverage)
from .workspace import canonical, digest
from .inference_distributions import log_f_survival, log_chi_square_survival, f_critical_value

SCHEMA = 'societybox-linear-iv-v1'
_METHODS = {'CLASSICAL': 'CLASSICAL_WLS', 'HC0': 'HC0', 'HC1': 'HC1',
            'CLUSTER_CR0': 'CLUSTER_CR0', 'CLUSTER_CR1': 'CLUSTER_CR1',
            'HAC_BARTLETT': 'HAC_BARTLETT'}
_FIELDS = {'schema', 'outcome', 'exogenous', 'endogenous', 'excluded_instruments',
           'observation_ids', 'model', 'source_declared_sha256', 'rights', 'covariance'}


def _fail(reason):
    raise ValueError('IV_' + reason)


def _text(value):
    return type(value) is str and 0 < len(value) <= 100 and not any(ord(c) < 32 for c in value)


def _snapshot(value, *, limit=131072):
    """Own strict JSON-native input; never silently repair an invalid schema.

    json.dumps/loads changes tuples into lists and nonstring mapping keys into
    strings. That silently widened accepted economic study contracts. Clone only
    actual JSON-native types before validating and hashing the same values.
    """
    def clone(item, depth=0):
        if depth>96: _fail('JSON_INPUT_REQUIRED')
        if type(item) is dict:
            if any(type(key) is not str for key in item): _fail('JSON_INPUT_REQUIRED')
            return {key:clone(child,depth+1) for key,child in item.items()}
        if type(item) is list:
            return [clone(child,depth+1) for child in item]
        if type(item) is float:
            if not math.isfinite(item): _fail('NUMBER_OR_INPUT_INVALID')
            return item
        if item is None or type(item) in (str,int,bool): return item
        _fail('JSON_INPUT_REQUIRED')
    try: owned=clone(value)
    except (RuntimeError,RecursionError): _fail('INPUT_CHANGED_DURING_SNAPSHOT')
    try: raw=canonical(owned)
    except (ValueError,TypeError,OverflowError,RecursionError): _fail('JSON_INPUT_REQUIRED')
    if len(raw)>limit: _fail('INPUT_SIZE_LIMIT')
    return owned


def _validated(q, *, require_order=True):
    if type(q) is not dict or set(q) != _FIELDS or q['schema'] != SCHEMA:
        _fail('SCHEMA_INVALID')
    if q['rights'] != dict(local_processing='PERMITTED', report_export='PERMITTED'):
        _fail('RIGHTS_REQUIRED')
    if type(q['source_declared_sha256']) is not str or not re.fullmatch('[a-f0-9]{64}', q['source_declared_sha256']):
        _fail('SOURCE_INVALID')
    y = q['outcome']
    if type(y) is not list or not 3 <= len(y) <= 256:
        _fail('DIMENSION_LIMIT')
    n = len(y); model = q['model']
    if type(model) is not dict or set(model) != {'outcome', 'exogenous', 'endogenous', 'excluded_instruments', 'units'}:
        _fail('MODEL_INVALID')
    names = []
    for key, minimum, maximum in (('exogenous', 0, 7), ('endogenous', 1, 8), ('excluded_instruments', 1, 12)):
        labels = model[key]
        if (type(labels) is not list or not minimum <= len(labels) <= maximum
                or any(not _text(v) for v in labels)):
            _fail('MODEL_NAMES_INVALID')
        names.extend(labels)
        rows = q[key]
        if (type(rows) is not list or len(rows) != n
                or any(type(row) is not list or len(row) != len(labels) for row in rows)):
            _fail('DIMENSIONS_INVALID')
    if not _text(model['outcome']) or len(set(names + [model['outcome']])) != len(names)+1:
        _fail('MODEL_NAMES_INVALID')
    if (type(model['units']) is not dict or set(model['units']) != set(names + [model['outcome']])
            or any(not _text(v) for v in model['units'].values())):
        _fail('UNITS_REQUIRED')
    e, d, l = map(lambda key: len(model[key]), ('exogenous', 'endogenous', 'excluded_instruments'))
    if e+d > 8 or e+l > 12 or n <= e+l:
        _fail('DIMENSION_LIMIT')
    if require_order and l < d:
        _fail('UNDERIDENTIFIED_ORDER_CONDITION')
    ids = q['observation_ids']
    if type(ids) is not list or len(ids) != n or any(not _text(v) for v in ids) or len(set(ids)) != n:
        _fail('OBSERVATION_IDS_INVALID')
    cov = q['covariance']
    if (type(cov) is not dict
            or type(cov.get('estimator')) is not str or cov['estimator'] not in _METHODS):
        _fail('COVARIANCE_UNSUPPORTED')
    expected = {'estimator', 'clusters'}
    if cov['estimator'] == 'HAC_BARTLETT': expected |= {'bandwidth', 'time_index'}
    if set(cov) != expected: _fail('COVARIANCE_UNSUPPORTED')
    if cov['estimator'] == 'HAC_BARTLETT':
        bandwidth, time_index = cov['bandwidth'], cov['time_index']
        if type(bandwidth) is not int or not 0 <= bandwidth < n:
            _fail('HAC_BANDWIDTH_INVALID')
        # Period coordinates, not observation IDs or authenticated dates. Never
        # silently sort, close gaps, pool panels, or delete missing periods.
        if (type(time_index) is not list or len(time_index) != n
                or any(type(t) is not int for t in time_index)
                or any(b != a+1 for a,b in zip(time_index,time_index[1:]))):
            _fail('HAC_CONTIGUOUS_INCREASING_PERIODS_REQUIRED')
    if cov['estimator'].startswith('CLUSTER_'):
        c = cov['clusters']
        if (type(c) is not list or len(c) != n or any(not _text(v) for v in c)
                or len(set(c)) < 2):
            _fail('CLUSTER_LABELS_INVALID')
    elif cov['clusters'] is not None:
        _fail('UNUSED_CLUSTER_LABELS')
    try:
        if len(canonical(q)) > 131072: _fail('INPUT_SIZE_LIMIT')
        y = list(map(_num, y))
        exog, endog, excluded = ([[ _num(v) for v in row] for row in q[key]]
                                 for key in ('exogenous', 'endogenous', 'excluded_instruments'))
    except (ValueError, OverflowError, TypeError) as error:
        _fail('NUMBER_OR_INPUT_INVALID:' + str(error))
    x = [a+b for a,b in zip(exog, endog)]
    z = [a+b for a,b in zip(exog, excluded)]
    return y, x, z, exog, endog


def validate(q):
    """Validate declarations only; never fit during review."""
    _validated(q)
    return q


def _cross(a, b):
    columns=list(zip(*a))
    if a is b:
        result=[[F(0)]*len(columns) for _ in columns]
        for i,col in enumerate(columns):
            checkpoint()
            for j in range(i,len(columns)):
                result[i][j]=result[j][i]=_dot(col,columns[j])
        return result
    other_columns=list(zip(*b))
    return [[_dot(col, other) for other in other_columns] for col in columns]


def _ols(x, y, reason):
    if not x[0]: return [], list(y), _dot(y, y)
    beta = _solve(_cross(x, x), [_dot(col, y) for col in zip(*x)])
    if beta is None: _fail(reason)
    residuals = [v-_dot(row, beta) for row,v in zip(x,y)]
    return beta, residuals, _dot(residuals, residuals)


def _inverse_small(a, *, failure='UNDERIDENTIFIED_PROJECTED_DESIGN'):
    """Exact pivoted Gauss--Jordan with all identity right sides at once.

    Trusted callers supply bounded square matrices. Repeating elimination for
    every identity column costs O(p**4); one augmented elimination costs O(p**3).
    No pseudoinverse, float conversion or relaxed rank threshold is used.
    """
    n=len(a)
    augmented=[list(map(F,row))+[F(int(i==j)) for j in range(n)] for i,row in enumerate(a)]
    for j in range(n):
        checkpoint()
        pivot=next((i for i in range(j,n) if augmented[i][j]),None)
        if pivot is None: _fail(failure)
        augmented[j],augmented[pivot]=augmented[pivot],augmented[j]
        row=augmented[j]; scale=row[j]
        for column in range(j+1,2*n): row[column]/=scale
        row[j]=F(1)
        nonzero=[(column,row[column]) for column in range(j+1,2*n) if row[column]]
        for i in range(n):
            if i==j or not augmented[i][j]: continue
            scale=augmented[i][j]; augmented[i][j]=F(0)
            for column,value in nonzero: augmented[i][column]-=scale*value
    return [row[n:] for row in augmented]


def _f_stat(restricted_sse, sse, numerator_df, denominator_df):
    if restricted_sse < sse: _fail('NESTED_MODEL_IDENTITY_FAILED')
    if sse == 0:
        return dict(f_statistic_exact=None,
                    statistic_status='INFINITE' if restricted_sse > 0 else 'UNDEFINED_ZERO_VARIANCE')
    return dict(f_statistic_exact=str((restricted_sse-sse)/numerator_df/(sse/denominator_df)),
                statistic_status='FINITE')


def _code_receipt():
    return {name: digest(Path(__file__).with_name(name).read_bytes()) for name in
            ('instrumental_variables.py', 'mechanism_dynamic.py', 'regression_sensitivity.py',
             'highs_native.py', 'highs_native_worker.py', 'inverse_native.py',
             'optimization_policy.py', 'strict_json.py', 'inference_distributions.py',
             'research_evidence.py', 'workspace.py', 'execution_control.py')}


def _assurance(q, result, *, extra_code=None):
    code = _code_receipt()
    if extra_code: code.update(extra_code)
    scope = dict(method=result['method'], model=q['model'], observation_ids=q['observation_ids'],
                 covariance=q['covariance'], source_declared_sha256=q['source_declared_sha256'])
    if 'critical_f_exact' in result: scope['critical_f_exact'] = result['critical_f_exact']
    if 'linear_restrictions' in q: scope['linear_restrictions'] = q['linear_restrictions']
    if 'k_class_options' in q: scope['k_class_options'] = q['k_class_options']
    if 'gmm_iteration_options' in q: scope['gmm_iteration_options'] = q['gmm_iteration_options']
    subject = digest(canonical(q))
    authority = ('LOCAL_EXACT_STATISTIC_AND_BINARY64_REFERENCE_DISTRIBUTION'
                 if 'distribution_arithmetic' in result else
                 'LOCAL_EXACT_AND_PINNED_UPSTREAM_NUMERIC_EXECUTION'
                 if result.get('native_execution') == 'EXECUTED' else 'LOCAL_EXACT_ARITHMETIC')
    evidence = make_evidence(subject_sha256=subject,
        producer=dict(tool_id='econ.iv', version='1', code_sha256=digest(canonical(code))),
        operation=result['method'], authority=authority, epistemic_state='DERIVED',
        proof_kind='DECLARED_LINEAR_IV_RESULT', scope=scope,
        coverage=dict(status='PARTIAL', unknown_dimensions=['instrument_exogeneity', 'exclusion',
                      'weak_identification', 'sampling_design', 'historical_availability', 'external_replication']),
        result=result, parent_evidence_ids=[],
        limitations=['Exact arithmetic verifies the declared calculation, not causal identification.',
                     'No automatic intercept, missing-row deletion, survey weights or asymptotic certification.'])
    ledger = EvidenceLedger([evidence]); registry = UnknownRegistry(ledger)
    unknowns = []; obligations = []
    claims = [
        ('causal_identification', 'Are instrument exogeneity and exclusion justified independently?'),
        ('weak_identification', 'Are instruments sufficiently informative for this specification and inference?'),
        ('sampling_and_vintage', 'Are sampling, units, source and historical availability authenticated?'),
        ('independent_replication', 'Can an independent implementation and environment replicate this result?')]
    if q['covariance']['estimator'] == 'HAC_BARTLETT':
        claims.append(('serial_dependence_and_bandwidth',
                       'Are the declared equally spaced periods, weak dependence and bandwidth asymptotics justified?'))
    if 'k_class_options' in q:
        claims.append(('reduced_form_likelihood_model',
                       'Are the reduced-form likelihood model and strong-identification covariance asymptotics justified?'))
    if result['method']=='CONTINUOUSLY_UPDATED_LINEAR_GMM':
        claims.append(('global_optimality',
                       'Can independent starts or a global argument rule out other CUE minima? Local convergence does not establish this.'))
    if 'critical_f_exact' in result:
        claims.append(('critical_value_calibration',
                       'Is the declared cutoff the intended F quantile under justified reference assumptions?'))
    for claim, question in claims:
        unknown = registry.register(question, scope, 'EXTERNAL_SCIENTIFIC_VALIDATION', 'IV_ASSUMPTION_CHECK')
        unknowns.append(unknown)
        obligations.append(make_obligation(claim_id=claim, subject_sha256=subject, scope=scope,
            required_authority='EXTERNAL_SCIENTIFIC_VALIDATION', required_proof_kind='IV_ASSUMPTION_CHECK',
            verifier_id='econ.iv', dependencies=[], unknown_ids=[unknown['unknown_id']], blocking_reasons=[],
            next_probe=question))
    return dict(input_sha256=subject, code_sha256=code, evidence=[evidence], unknowns=unknowns,
                obligations=obligations, coverage=evaluate_coverage(obligations, ledger, registry))


def calculate(q):
    """2SLS with classical, HC0/1, one-way CR0/1 or Bartlett HAC covariance.

    HAC covariance keys are estimator='HAC_BARTLETT', clusters=None,
    bandwidth (integer, 0..n-1), time_index (n consecutive integer periods in
    row order). Uses structural errors, uncentered scores and no df adjustment,
    prewhitening or automatic bandwidth. Periods are researcher declarations.

    First-stage F is classical excluded-instrument F, not a robust weak-IV
    test or a multi-endogenous joint rank diagnostic. No threshold certifies
    strength. Full-rank fits still retain unresolved identification obligations.
    """
    checkpoint()
    q=_snapshot(q)
    y, x, z, exog, endog = _validated(q)
    n, k, l = len(y), len(x[0]), len(z[0])
    zz = _cross(z, z); zx = _cross(z, x)
    columns = [_solve(zz, list(col)) for col in zip(*zx)]
    if any(c is None for c in columns): _fail('INSTRUMENT_RANK_DEFICIENT')
    xhat = [[_dot(row, col) for col in columns] for row in z]
    bread = _inverse_small(_cross(xhat, xhat))
    projected_rhs = [_dot(col, y) for col in zip(*xhat)]
    beta = [_dot(row, projected_rhs) for row in bread]
    residuals = [target-_dot(row, beta) for row,target in zip(x,y)]
    normal = [_dot(col, residuals) for col in zip(*xhat)]
    if any(normal): _fail('PROJECTED_NORMAL_IDENTITY_FAILED')
    method = _METHODS[q['covariance']['estimator']]
    ids = q['observation_ids']; clusters = q['covariance']['clusters']
    covariance = _covariances(xhat, residuals, [F(1)]*n, bread, ids,
        dict(estimators=[method], bandwidth=q['covariance'].get('bandwidth'),
             clusters=None if clusters is None else dict(zip(ids, clusters))))
    stages = []
    for name, target in zip(q['model']['endogenous'], zip(*endog)):
        checkpoint()
        first_beta, _, sse = _ols(z, target, 'INSTRUMENT_RANK_DEFICIENT')
        _, _, restricted = _ols(exog, target, 'EXOGENOUS_RANK_DEFICIENT')
        stages.append(dict(variable=name, coefficients_exact=list(map(str, first_beta)),
            partial_r_squared_exact=None if restricted == 0 else str(1-sse/restricted),
            numerator_df=l-len(exog[0]), denominator_df=n-l,
            reference_distribution='CLASSICAL_F_ASSUMPTIONS_UNVERIFIED',
            weak_iv_test='NOT_A_ROBUST_OR_JOINT_RANK_TEST',
            **_f_stat(restricted, sse, l-len(exog[0]), n-l)))
    result = dict(schema='societybox-linear-iv-result-v1', method='TWO_STAGE_LEAST_SQUARES',
        status='COMPUTED_ASSUMPTIONS_UNVERIFIED', implementation_status='INDEPENDENT_IMPLEMENTED',
        native_execution='NOT_RUN', native_parity='NOT_RUN', ui_status='NOT_CONNECTED',
        identification='IDENTIFICATION_NOT_ESTABLISHED',
        parameter_order=q['model']['exogenous']+q['model']['endogenous'],
        coefficients_exact=list(map(str, beta)), covariance_estimator=q['covariance']['estimator'],
        covariance_exact=covariance[method]['matrix_exact'], covariance_specification=q['covariance'],
        structural_residuals_exact=list(map(str, residuals)), structural_sse_exact=str(_dot(residuals, residuals)),
        projected_normal_residual_exact=list(map(str, normal)), first_stages=stages,
        observations=n, residual_df=n-k, overidentifying_restrictions=l-k,
        weak_identification='NOT_ESTABLISHED_USE_DESIGN_SPECIFIC_DIAGNOSTICS_AND_AR',
        source_authentication='NOT_ESTABLISHED', historical_as_of='NOT_ESTABLISHED')
    result.update(_assurance(q, result))
    return result


def anderson_rubin(q, null_endogenous_coefficients):
    """Classical joint AR F statistic for an explicit endogenous coefficient null.

    Regress y-D*b0 on all instruments, test excluded instrument coefficients.
    Remains defined at zero first-stage relevance. The finite-sample F reference
    requires iid Gaussian homoskedastic structural errors and valid instruments.
    P-values use the central F reference with a separately labelled binary64
    tail. Robust AR and scalar confidence-set inversion have separate APIs.
    """
    checkpoint()
    q=_snapshot(q)
    y, _, z, exog, endog = _validated(q,require_order=False)
    if q['covariance']['estimator'] != 'CLASSICAL': _fail('AR_CLASSICAL_ONLY')
    if type(null_endogenous_coefficients) is not list or len(null_endogenous_coefficients) != len(endog[0]):
        _fail('AR_NULL_DIMENSIONS_INVALID')
    try: null = list(map(_num, null_endogenous_coefficients))
    except ValueError as error: _fail('AR_NULL_NUMBER_INVALID:' + str(error))
    transformed = [v-_dot(row, null) for v,row in zip(y,endog)]
    _, _, unrestricted = _ols(z, transformed, 'INSTRUMENT_RANK_DEFICIENT')
    _, _, restricted = _ols(exog, transformed, 'EXOGENOUS_RANK_DEFICIENT')
    df1, df2 = len(z[0])-len(exog[0]), len(y)-len(z[0])
    # Bind the tested null as well as the study specification in the receipt.
    bound = dict(q, tested_null_exact=list(map(str, null)))
    result = dict(schema='societybox-ar-result-v1', method='CLASSICAL_ANDERSON_RUBIN_F',
        null_endogenous_coefficients_exact=list(map(str, null)), numerator_df=df1, denominator_df=df2,
        restricted_sse_exact=str(restricted), unrestricted_sse_exact=str(unrestricted),
        reference_distribution='F_CONDITIONAL_ON_IID_GAUSSIAN_HOMOSKEDASTIC_ERRORS',
        identification='IDENTIFICATION_NOT_ESTABLISHED',
        statistic_arithmetic='EXACT_RATIONAL', distribution_arithmetic='BINARY64_NOT_INTERVAL_CERTIFIED',
        native_execution='NOT_RUN', native_parity='NOT_RUN', implementation_status='INDEPENDENT_IMPLEMENTED',
        **_f_stat(restricted, unrestricted, df1, df2))
    result.update(_f_probability(result,df1,df2))
    result.update(_assurance(bound, result))
    return result


def _probability(log_probability):
    value = math.exp(log_probability)
    return dict(p_value=value,log_p_value='-Infinity' if log_probability == -math.inf else log_probability,
                p_value_status='BINARY64_UNDERFLOW_LOG_AVAILABLE' if value == 0 and math.isfinite(log_probability)
                else 'COMPUTED_CONDITIONAL_REFERENCE_DISTRIBUTION')


def _f_probability(statistic,df1,df2):
    if statistic['statistic_status'] == 'UNDEFINED_ZERO_VARIANCE':
        return dict(p_value=None,log_p_value=None,p_value_status='UNDEFINED_ZERO_RESIDUAL_VARIANCE')
    if statistic['statistic_status'] == 'INFINITE':
        return dict(p_value=0.0,log_p_value='-Infinity',p_value_status='REFERENCE_LIMIT_INFINITE_STATISTIC')
    return _probability(log_f_survival(F(statistic['f_statistic_exact']),df1,df2))


def _chi_probability(statistic,degrees):
    try: return _probability(log_chi_square_survival(statistic,degrees))
    except ValueError as error:
        if str(error) != 'INFERENCE_STATISTIC_FLOAT64_RANGE': raise
        # Preserve the exact Wald/J statistic rather than discard a valid fit
        # because an optional log reference probability cannot fit binary64.
        return dict(p_value=None,log_p_value=None,p_value_status='REFERENCE_PROBABILITY_BLOCKED_FLOAT64_RANGE')


def _positive_weight(raw, size):
    if (type(raw) is not list or len(raw) != size
            or any(type(row) is not list or len(row) != size for row in raw)):
        _fail('GMM_WEIGHT_DIMENSIONS_INVALID')
    try: weight = [[_num(v) for v in row] for row in raw]
    except ValueError as error: _fail('GMM_WEIGHT_NUMBER_INVALID:' + str(error))
    if any(weight[i][j] != weight[j][i] for i in range(size) for j in range(size)):
        _fail('GMM_WEIGHT_SYMMETRY_REQUIRED')
    # Exact Schur complements: all LDL pivots must be strictly positive.
    reduced = [list(row) for row in weight]
    for j in range(size):
        if reduced[j][j] <= 0: _fail('GMM_WEIGHT_POSITIVE_DEFINITE_REQUIRED')
        for i in range(j+1, size):
            for k in range(j+1, size):
                reduced[i][k] -= reduced[i][j]*reduced[j][k]/reduced[j][j]
    return weight


def _gmm_fit(x, y, z, weight, *, cross_products=None):
    # Reuse only mathematical cross products of the same owned study snapshot.
    # This is not a source/evidence digest cache.
    zx,zy = (_cross(z,x),[_dot(col,y) for col in zip(*z)]) if cross_products is None else cross_products
    mapping = _multiply([list(row) for row in zip(*zx)], weight)
    bread = _inverse_small(_multiply(mapping, zx))
    influence_map = _multiply(bread, mapping)
    beta = [_dot(row, zy) for row in influence_map]
    residuals = [v-_dot(row, beta) for v,row in zip(y,x)]
    moments = [_dot(col, residuals) for col in zip(*z)]
    normal = [_dot(row, moments) for row in mapping]
    if any(normal): _fail('GMM_NORMAL_IDENTITY_FAILED')
    return beta, residuals, moments, normal, influence_map


def _moment_meat(z, residuals, method, clusters, parameter_count, bandwidth=None):
    n, l = len(z), len(z[0])
    if method == 'CLASSICAL':
        sigma2 = _dot(residuals, residuals)/(n-parameter_count)
        return [[sigma2*v for v in row] for row in _cross(z,z)]
    scores = [[v*e for v in row] for row,e in zip(z,residuals)]
    if method == 'HAC_BARTLETT': return _bartlett_meat(scores,bandwidth)
    factor = F(n, n-parameter_count) if method == 'HC1' else F(1)
    if method.startswith('CLUSTER_'):
        groups = {}
        for label, score in zip(clusters, scores):
            total = groups.setdefault(label, [F(0)]*l)
            for j in range(l): total[j] += score[j]
        scores = list(groups.values())
        if method == 'CLUSTER_CR1': factor = F(len(groups),len(groups)-1)*F(n-1,n-parameter_count)
    return [[factor*v for v in row] for row in _cross(scores,scores)]


def _fit_linear_gmm(x,y,z,initial,method,clusters,steps,bandwidth=None):
    """Shared trusted matrix core; public entry points validate their contracts."""
    k=len(x[0]); weight=initial; fitted=_gmm_fit(x,y,z,weight)
    first_beta=list(fitted[0])
    if steps == 2:
        checkpoint()
        mode=('CLUSTER_CR0' if method.startswith('CLUSTER_') else
              'HAC_BARTLETT' if method == 'HAC_BARTLETT' else 'HC0')
        meat1=_moment_meat(z,fitted[1],mode,clusters,k,bandwidth)
        try: weight=_inverse_small(meat1)
        except ValueError as error: _fail('GMM_MOMENT_COVARIANCE_SINGULAR:' + str(error))
        fitted=_gmm_fit(x,y,z,weight)
    beta,residuals,moments,normal,influence=fitted
    meat=_moment_meat(z,residuals,method,clusters,k,bandwidth)
    covariance=_multiply(_multiply(influence,meat),[list(row) for row in zip(*influence)])
    criterion=_dot(moments,[_dot(row,moments) for row in weight])
    if criterion < 0 or any(covariance[i][i]<0 for i in range(k)):
        _fail('GMM_NONNEGATIVITY_FAILED')
    return beta,residuals,moments,normal,first_beta,weight,covariance,criterion


def linear_gmm(q, *, steps=1, weight_matrix=None):
    """Linear one-step GMM or uncentered iid/cluster/Bartlett-HAC two-step GMM.

    An explicit weight must be symmetric positive definite and uses SUM moment
    normalization. Default initial weight is inverse Z'Z (2SLS). Two-step weight
    is inverse sum(z_i*z_i'*u1_i**2), or inverse sum of outer products of cluster
    moment totals, without small-sample scaling. Covariance is
    the full bread/meat sandwich, including for arbitrary weights. This is not
    CUE, iterated GMM, dynamic-panel GMM or Windmeijer-corrected inference.
    HAC uses calculate's explicit period/bandwidth contract for both the
    first-step moment weight and the final residual sandwich. A fixed positive
    bandwidth is a finite-sample calculation, not a consistency certificate;
    dependent-data J inference requires justified bandwidth asymptotics.
    """
    checkpoint()
    q=_snapshot(q)
    y,x,z,_,_ = _validated(q)
    if type(steps) is not int or steps not in (1,2): _fail('GMM_STEPS_UNSUPPORTED')
    method = q['covariance']['estimator']
    if steps == 2 and method not in ('HC0','HC1','CLUSTER_CR0','CLUSTER_CR1','HAC_BARTLETT'):
        _fail('GMM_TWO_STEP_REQUIRES_ROBUST_COVARIANCE')
    n,k,l = len(y),len(x[0]),len(z[0])
    # Reject singular instruments even when an arbitrary weight could mask them.
    zz_inverse = _inverse_small(_cross(z,z), failure='INSTRUMENT_RANK_DEFICIENT')
    initial = zz_inverse if weight_matrix is None else _positive_weight(weight_matrix,l)
    beta,residuals,moments,normal,first_beta,weight,covariance,criterion = _fit_linear_gmm(
        x,y,z,initial,method,q['covariance']['clusters'],steps,q['covariance'].get('bandwidth'))
    overid = ('NOT_APPLICABLE_EXACTLY_IDENTIFIED' if l == k else
              'ASYMPTOTIC_CHI_SQUARED_ASSUMPTIONS_UNVERIFIED' if steps == 2 else
              'NOT_CALIBRATED_FOR_DECLARED_WEIGHT')
    overid_probability = (_chi_probability(criterion,l-k) if steps == 2 and l>k else
                          dict(p_value=None,log_p_value=None,p_value_status=overid))
    bound = dict(q, gmm_steps=steps, initial_weight_matrix=weight_matrix)
    result = dict(schema='societybox-linear-gmm-result-v1', method='LINEAR_GMM', steps=steps,
        status='COMPUTED_ASSUMPTIONS_UNVERIFIED', implementation_status='INDEPENDENT_IMPLEMENTED',
        native_execution='NOT_RUN', native_parity='NOT_RUN', ui_status='NOT_CONNECTED',
        identification='IDENTIFICATION_NOT_ESTABLISHED',
        parameter_order=q['model']['exogenous']+q['model']['endogenous'],
        coefficients_exact=list(map(str,beta)), covariance_exact=[list(map(str,row)) for row in covariance],
        covariance_estimator=method, covariance_specification=q['covariance'],
        structural_residuals_exact=list(map(str,residuals)),
        structural_sse_exact=str(_dot(residuals,residuals)),
        weighted_normal_residual_exact=list(map(str,normal)), moment_sums_exact=list(map(str,moments)),
        first_step_coefficients_exact=list(map(str,first_beta)),
        weight_matrix_exact=[list(map(str,row)) for row in weight],
        weight_convention='SUM_MOMENTS_NOT_MEAN_MOMENTS', moment_centering='UNCENTERED',
        two_step_weight_mode=('CLUSTER_MOMENT_TOTALS' if method.startswith('CLUSTER_') else
                              'HAC_BARTLETT_MOMENT_SCORES' if method == 'HAC_BARTLETT' else 'IID_MOMENT_SCORES'),
        gmm_criterion_exact=str(criterion), overidentifying_restrictions=l-k,
        overidentification_test=dict(status=overid, degrees_of_freedom=l-k, **overid_probability),
        observations=n, residual_df=n-k, source_authentication='NOT_ESTABLISHED',
        historical_as_of='NOT_ESTABLISHED', small_sample_two_step_correction='NOT_IMPLEMENTED')
    if steps == 2 and l>k: result['distribution_arithmetic']='BINARY64_NOT_INTERVAL_CERTIFIED'
    result.update(_assurance(bound,result))
    return result


def linear_wald(q, restrictions, *, estimator='2SLS', steps=1, weight_matrix=None):
    """Test R*beta=r using a declared 2SLS or linear GMM sandwich.

    restrictions={'matrix': [[...], ...], 'values': [...]} uses the explicit
    exogenous+endogenous parameter order. Require 1..K independent rows, the
    same bounded numeric contract as model inputs, and nonsingular R*V*R'.
    Only estimator='GMM' accepts steps=2 or a supplied initial weight matrix.
    The exact statistic d'*(R*V*R')^-1*d has a binary64 asymptotic chi-square
    reference probability. No finite-sample F or weak-IV robustness is claimed;
    even a full-rank first stage does not establish the inference assumptions.
    Use anderson_rubin[_robust] for the supported weak-identification path.
    """
    checkpoint()
    owned=_snapshot(dict(study=q, restrictions=restrictions, estimator=estimator,
                         steps=steps, weight_matrix=weight_matrix),limit=196608)
    q=owned['study']; restrictions=owned['restrictions']; weight_matrix=owned['weight_matrix']
    _,x,_,_,_=_validated(q)
    if type(estimator) is not str or estimator not in ('2SLS','GMM'):
        _fail('WALD_ESTIMATOR_UNSUPPORTED')
    if type(steps) is not int or steps not in (1,2): _fail('GMM_STEPS_UNSUPPORTED')
    if estimator=='2SLS' and (steps!=1 or weight_matrix is not None):
        _fail('WALD_UNUSED_GMM_OPTIONS')
    k=len(x[0])
    if type(restrictions) is not dict or set(restrictions)!={'matrix','values'}:
        _fail('WALD_RESTRICTIONS_INVALID')
    raw_matrix, raw_values=restrictions['matrix'],restrictions['values']
    if (type(raw_matrix) is not list or not 1<=len(raw_matrix)<=k
            or any(type(row) is not list or len(row)!=k for row in raw_matrix)
            or type(raw_values) is not list or len(raw_values)!=len(raw_matrix)):
        _fail('WALD_RESTRICTIONS_DIMENSIONS_INVALID')
    try:
        matrix=[[ _num(v) for v in row] for row in raw_matrix]
        values=list(map(_num,raw_values))
    except ValueError as error: _fail('WALD_RESTRICTIONS_NUMBER_INVALID:'+str(error))
    transposed=[list(row) for row in zip(*matrix)]
    _inverse_small(_multiply(matrix,transposed),failure='WALD_RESTRICTIONS_RANK_DEFICIENT')
    fitted=calculate(q) if estimator=='2SLS' else linear_gmm(q,steps=steps,weight_matrix=weight_matrix)
    beta=list(map(F,fitted['coefficients_exact']))
    covariance=[list(map(F,row)) for row in fitted['covariance_exact']]
    difference=[_dot(row,beta)-value for row,value in zip(matrix,values)]
    restricted_covariance=_multiply(_multiply(matrix,covariance),transposed)
    precision=_inverse_small(restricted_covariance,failure='WALD_RESTRICTION_COVARIANCE_SINGULAR')
    statistic=_dot(difference,[_dot(row,difference) for row in precision])
    if statistic<0: _fail('WALD_STATISTIC_NEGATIVE')
    result=dict(schema='societybox-linear-iv-wald-result-v1',method='LINEAR_RESTRICTION_WALD',
        status='COMPUTED_ASSUMPTIONS_UNVERIFIED',implementation_status='INDEPENDENT_IMPLEMENTED',
        estimator=estimator,gmm_steps=steps if estimator=='GMM' else None,
        estimated_coefficients_exact=fitted['coefficients_exact'],parameter_order=fitted['parameter_order'],
        estimation_input_sha256=fitted['input_sha256'],linear_restrictions=restrictions,
        restriction_residuals_exact=list(map(str,difference)),
        restriction_covariance_exact=[list(map(str,row)) for row in restricted_covariance],
        covariance_estimator=q['covariance']['estimator'],covariance_specification=q['covariance'],
        chi_squared_statistic_exact=str(statistic),degrees_of_freedom=len(matrix),
        statistic_arithmetic='EXACT_RATIONAL',distribution_arithmetic='BINARY64_NOT_INTERVAL_CERTIFIED',
        reference_distribution='ASYMPTOTIC_CHI_SQUARED_ASSUMPTIONS_UNVERIFIED',
        weak_instrument_robust=False,identification='IDENTIFICATION_NOT_ESTABLISHED',
        native_execution='NOT_RUN',native_parity='NOT_RUN',ui_status='NOT_CONNECTED',
        **_chi_probability(statistic,len(matrix)))
    bound=dict(q,linear_restrictions=restrictions,estimator=estimator,gmm_steps=steps,
               initial_weight_matrix=weight_matrix)
    result.update(_assurance(bound,result))
    return result


def compare_highs(q, coefficient_bounds, runtime):
    """Execute the pinned HiGHS QP API on the equivalent projected-design loss.

    This verifies 2SLS coefficients and projected least-squares MSE, not IV
    covariance or an instrument-validity test. Bounds must contain the exact
    unconstrained solution. No installation, source intake or automatic runtime
    selection occurs. The existing pinned native numeric/resource limits apply.
    """
    from . import highs_native
    from .regression_sensitivity import _native_wls_program, _compare
    if runtime is None: _fail('PINNED_RUNTIME_REQUIRED')
    caller_request=q
    q=_snapshot(q)
    independent = calculate(q)
    y,x,z,_,_ = _validated(q)
    k = len(x[0])
    if k > 4: _fail('NATIVE_DIMENSION_BLOCKED')
    if (type(coefficient_bounds) is not list or len(coefficient_bounds) != k
            or any(type(pair) is not list or len(pair) != 2 for pair in coefficient_bounds)):
        _fail('NATIVE_BOUNDS_INVALID')
    try:
        low = [_num(pair[0]) for pair in coefficient_bounds]
        high = [_num(pair[1]) for pair in coefficient_bounds]
    except ValueError as error: _fail('NATIVE_BOUNDS_INVALID:' + str(error))
    beta = list(map(F,independent['coefficients_exact']))
    if any(lo >= hi or not lo <= b <= hi for lo,hi,b in zip(low,high,beta)):
        _fail('NATIVE_BOUNDS_EXCLUDE_UNCONSTRAINED_IV')
    bound = dict(q, coefficient_bounds=coefficient_bounds)
    declared = digest(canonical(bound)); code = _code_receipt()
    zz,zx = _cross(z,z),_cross(z,x)
    columns = [_solve(zz,list(col)) for col in zip(*zx)]
    xhat = [[_dot(row,col) for col in columns] for row in z]
    projected_errors = [target-_dot(row,beta) for row,target in zip(xhat,y)]
    loss = _dot(projected_errors,projected_errors)/len(y)
    weights = [F(1)]*len(y)
    program, scale = _native_wls_program(xhat,y,beta,low,high,weights)
    native = highs_native.run(program,runtime)
    parity = _compare(beta,xhat,y,weights,native,loss,scale)
    if parity['status'] == 'FAILED': _fail('NATIVE_PARITY_FAILED')
    try:
        unchanged=(digest(canonical(dict(q,coefficient_bounds=coefficient_bounds)))==declared
                   and digest(canonical(caller_request))==digest(canonical(q))
                   and _code_receipt()==code)
    except (ValueError,TypeError,OverflowError,RecursionError):
        unchanged=False
    if not unchanged: _fail('NATIVE_INPUT_OR_CODE_CHANGED_DURING_RUN')
    bound['runtime_pin'] = native['upstream']['runtime_pin']
    result = dict(schema='societybox-iv-highs-parity-v1', method='2SLS_PROJECTED_QP_COMPARISON',
        implementation_status='NUMERICAL_PARITY_VERIFIED', native_execution='EXECUTED',
        upstream_status='ORIGINAL_API_EXECUTED', independent=independent, native=native, parity=parity,
        projected_mse_exact=str(loss), native_covariance='NOT_RUN',
        parity_scope='2SLS_COEFFICIENTS_AND_PROJECTED_QP_OBJECTIVE_ONLY',
        identification='IDENTIFICATION_NOT_ESTABLISHED', ui_status='NOT_CONNECTED')
    result.update(_assurance(bound,result))
    return result


def _decimal_fraction(value):
    return Decimal(value.numerator)/Decimal(value.denominator)


def _endpoint(value):
    with localcontext() as context:
        context.prec = 80
        return dict(exact=str(value), decimal=str(_decimal_fraction(value)))


def _interval(lower, upper):
    return dict(lower=lower, upper=upper, lower_inclusive=lower is not None, upper_inclusive=upper is not None)


def _quadratic_intervals(a,b,c):
    """Solve a*t**2+b*t+c <= 0. Exact topology; algebraic root provenance."""
    if a == 0:
        if b == 0: return [_interval(None,None)] if c <= 0 else []
        root = _endpoint(-c/b)
        return [_interval(None,root)] if b > 0 else [_interval(root,None)]
    delta = b*b-4*a*c
    if delta < 0: return [_interval(None,None)] if a < 0 else []
    if delta == 0:
        root = _endpoint(-b/(2*a))
        return [_interval(root,root)] if a > 0 else [_interval(None,None)]
    sn,sd = isqrt(delta.numerator),isqrt(delta.denominator)
    if sn*sn == delta.numerator and sd*sd == delta.denominator:
        sqrt = F(sn,sd)
        roots = sorted([(-b-sqrt)/(2*a),(-b+sqrt)/(2*a)])
        lower,upper = map(_endpoint,roots)
    else:
        with localcontext() as context:
            context.prec = 80
            da,db,dc = map(_decimal_fraction,(a,b,c))
            sqrt = _decimal_fraction(delta).sqrt()
            # Stable pair avoids catastrophic cancellation in the smaller root.
            q = -(db+(sqrt if b >= 0 else -sqrt))/2
            pair = [q/da,dc/q]
            if (a > 0) != (b >= 0): pair.reverse()
            lower,upper = [dict(exact=None,decimal=str(value),
                polynomial_coefficients_exact=list(map(str,(a,b,c))),root_index=i)
                for i,value in enumerate(pair)]
    return [_interval(lower,upper)] if a > 0 else [_interval(None,lower),_interval(upper,None)]


def invert_anderson_rubin(q, critical_f):
    """Invert scalar endogenous AR F at a researcher-declared positive cutoff.

    The exact acceptance rule is quadratic <= 0 AND unrestricted SSE > 0.
    Interval topology and rational boundaries are exact; irrational boundaries
    have an exact polynomial/root index plus 80-digit decimal approximations.
    A cutoff alone does not establish a nominal confidence level. No automatic
    F quantile, robust AR, CLR, grid truncation or bounded-set assumption.
    """
    checkpoint()
    q=_snapshot(q)
    y,_,z,exog,endog = _validated(q,require_order=False)
    if q['covariance']['estimator'] != 'CLASSICAL': _fail('AR_CLASSICAL_ONLY')
    if len(endog[0]) != 1: _fail('AR_INVERSION_SINGLE_ENDOGENOUS_REQUIRED')
    critical = _ar_critical_number(critical_f)
    d = [row[0] for row in endog]
    polynomials = []
    for design in (exog,z):
        _,ry,_ = _ols(design,y,'INSTRUMENT_RANK_DEFICIENT')
        _,rd,_ = _ols(design,d,'INSTRUMENT_RANK_DEFICIENT')
        polynomials.append([_dot(rd,rd), -2*_dot(ry,rd), _dot(ry,ry)])
    restricted,unrestricted = polynomials
    df1,df2 = len(z[0])-len(exog[0]),len(y)-len(z[0])
    a,b,c = [df2*(r-u)-critical*df1*u for r,u in zip(restricted,unrestricted)]
    zero_variance = all(v == 0 for v in unrestricted)
    intervals = None if zero_variance else _quadratic_intervals(a,b,c)
    excluded = []
    ua,ub,uc = unrestricted
    if ua > 0 and ub*ub == 4*ua*uc:
        point = -ub/(2*ua); excluded.append(str(point))
        # Since both SSEs are nonnegative, an accepted zero-variance point is
        # either a rational boundary or lies in an all-real quadratic set.
        if a*point*point+b*point+c == 0:
            fixed = []
            for interval in intervals:
                lower,upper = interval['lower'],interval['upper']
                at_lower = lower is not None and lower['exact'] == str(point)
                at_upper = upper is not None and upper['exact'] == str(point)
                if at_lower and at_upper: continue
                if lower is None and upper is None:
                    left,right = _interval(None,_endpoint(point)),_interval(_endpoint(point),None)
                    left['upper_inclusive']=False; right['lower_inclusive']=False
                    fixed.extend([left,right]); continue
                if at_lower: interval['lower_inclusive']=False
                if at_upper: interval['upper_inclusive']=False
                fixed.append(interval)
            intervals = fixed
    bound = dict(q,critical_f_exact=str(critical))
    result = dict(schema='societybox-ar-inversion-v1', method='CLASSICAL_AR_CRITICAL_VALUE_INVERSION',
        status='BLOCKED_ZERO_RESIDUAL_VARIANCE' if zero_variance else 'COMPUTED_ASSUMPTIONS_UNVERIFIED',
        implementation_status='INDEPENDENT_IMPLEMENTED', native_execution='NOT_RUN', native_parity='NOT_RUN',
        identification='IDENTIFICATION_NOT_ESTABLISHED', critical_f_exact=str(critical),
        nominal_coverage='NOT_ESTABLISHED_FOR_DECLARED_CRITICAL_VALUE',
        reference_distribution='F_CONDITIONAL_ON_IID_GAUSSIAN_HOMOSKEDASTIC_ERRORS',
        numerator_df=df1,denominator_df=df2, quadratic_coefficients_exact=list(map(str,(a,b,c))),
        unrestricted_sse_coefficients_exact=list(map(str,unrestricted)),
        acceptance_condition='QUADRATIC_LE_ZERO_AND_UNRESTRICTED_SSE_STRICTLY_POSITIVE',
        acceptance_intervals=intervals,excluded_zero_variance_points_exact=excluded,
        endpoint_decimal_precision=80, ui_status='NOT_CONNECTED')
    result.update(_assurance(bound,result))
    return result


def _ar_critical_number(value):
    # Reference cutoffs can exceed the solver's 1e6 coefficient limit; this
    # independent bound applies only to a scalar statistical threshold.
    if type(value) not in (str,int,float): _fail('AR_CRITICAL_VALUE_INVALID')
    if type(value) is int and value.bit_length() > 512: _fail('AR_CRITICAL_VALUE_LIMIT')
    if type(value) is float and not math.isfinite(value): _fail('AR_CRITICAL_VALUE_INVALID')
    if type(value) is str and (len(value)>160 or not re.fullmatch(r'-?\d+(?:\.\d+|/\d+)?(?:[eE][+-]?\d{1,3})?',value)):
        _fail('AR_CRITICAL_VALUE_INVALID')
    try: result = F(str(value))
    except (ValueError,ZeroDivisionError) as error: _fail('AR_CRITICAL_VALUE_INVALID:' + str(error))
    if result <= 0: _fail('AR_CRITICAL_VALUE_POSITIVE_REQUIRED')
    if result > 10**100 or max(result.numerator.bit_length(),result.denominator.bit_length())>512:
        _fail('AR_CRITICAL_VALUE_LIMIT')
    return result


def anderson_rubin_confidence_set(q, *, alpha=.05):
    """Scalar classical AR confidence set calibrated to the central F reference.

    Quantile calibration is numeric; nominal 1-alpha coverage is conditional on
    valid instruments and iid Gaussian homoskedastic errors, not established by
    this computation. Disconnected/unbounded/empty sets remain supported.
    """
    q=_snapshot(q)
    y,_,z,exog,endog = _validated(q,require_order=False)
    if q['covariance']['estimator'] != 'CLASSICAL': _fail('AR_CLASSICAL_ONLY')
    if len(endog[0]) != 1: _fail('AR_INVERSION_SINGLE_ENDOGENOUS_REQUIRED')
    calibration = f_critical_value(alpha,len(z[0])-len(exog[0]),len(y)-len(z[0]))
    raw = invert_anderson_rubin(q,repr(calibration['critical_value']))
    assurance = {'input_sha256','code_sha256','evidence','unknowns','obligations','coverage'}
    result = {key:value for key,value in raw.items() if key not in assurance}
    result.update(method='CLASSICAL_AR_NOMINAL_CONFIDENCE_SET',
        nominal_coverage='CONDITIONAL_ON_STATED_REFERENCE_ASSUMPTIONS_NOT_DESIGN_VERIFIED',
        requested_confidence_level=1-calibration['requested_upper_tail'],quantile_calibration=calibration,
        calibration_scope='ENDPOINTS_SOLVE_RETURNED_BINARY64_CUTOFF_NOT_CERTIFIED_TRUE_QUANTILE',
        distribution_arithmetic='BINARY64_NOT_INTERVAL_CERTIFIED')
    bound = dict(q,significance_level_exact=str(F(str(alpha))))
    result.update(_assurance(bound,result))
    return result


def anderson_rubin_robust(q, null_endogenous_coefficients):
    """Reduced-form HC0/HC1/one-way-cluster/Bartlett-HAC AR Wald test.

    OLS of y-D*b0 on [exogenous,excluded instruments]; the Wald test concerns
    the excluded coefficients. Covariance df is that unrestricted regression's
    n-L, not the structural IV equation's n-K. Chi-square calibration is
    asymptotic (independent rows, sufficiently many independent clusters, or
    justified weak dependence and bandwidth asymptotics for HAC).
    This is neither CLR nor a finite-sample F test nor robust set inversion.
    """
    checkpoint()
    q=_snapshot(q)
    y,_,z,exog,endog = _validated(q,require_order=False)
    if q['covariance']['estimator'] == 'CLASSICAL': _fail('AR_ROBUST_COVARIANCE_REQUIRED')
    if type(null_endogenous_coefficients) is not list or len(null_endogenous_coefficients) != len(endog[0]):
        _fail('AR_NULL_DIMENSIONS_INVALID')
    try: null = list(map(_num,null_endogenous_coefficients))
    except ValueError as error: _fail('AR_NULL_NUMBER_INVALID:' + str(error))
    target = [v-_dot(row,null) for v,row in zip(y,endog)]
    beta,residuals,sse = _ols(z,target,'INSTRUMENT_RANK_DEFICIENT')
    bread = _inverse_small(_cross(z,z),failure='INSTRUMENT_RANK_DEFICIENT')
    method = _METHODS[q['covariance']['estimator']]; ids = q['observation_ids']
    clusters = q['covariance']['clusters']; n,l,e = len(y),len(z[0]),len(exog[0])
    full = _covariances(z,residuals,[F(1)]*n,bread,ids,
        dict(estimators=[method],bandwidth=q['covariance'].get('bandwidth'),
             clusters=None if clusters is None else dict(zip(ids,clusters))))
    covariance = [[F(v) for v in row[e:]] for row in full[method]['matrix_exact'][e:]]
    inverse = _inverse_small(covariance,failure='AR_ROBUST_COVARIANCE_SINGULAR')
    excluded = beta[e:]; statistic = _dot(excluded,[_dot(row,excluded) for row in inverse])
    if statistic < 0: _fail('AR_ROBUST_WALD_NEGATIVE')
    result = dict(schema='societybox-robust-ar-result-v1',method='REDUCED_FORM_ROBUST_AR_WALD',
        status='COMPUTED_ASSUMPTIONS_UNVERIFIED',implementation_status='INDEPENDENT_IMPLEMENTED',
        native_execution='NOT_RUN',native_parity='NOT_RUN',ui_status='NOT_CONNECTED',
        identification='IDENTIFICATION_NOT_ESTABLISHED',covariance_estimator=q['covariance']['estimator'],
        covariance_specification=q['covariance'],
        null_endogenous_coefficients_exact=list(map(str,null)),excluded_parameter_order=q['model']['excluded_instruments'],
        excluded_coefficients_exact=list(map(str,excluded)),covariance_exact=[list(map(str,row)) for row in covariance],
        unrestricted_residual_df=n-l,unrestricted_sse_exact=str(sse),numerator_df=l-e,
        chi_squared_statistic_exact=str(statistic),statistic_arithmetic='EXACT_RATIONAL',
        distribution_arithmetic='BINARY64_NOT_INTERVAL_CERTIFIED',
        reference_distribution='ASYMPTOTIC_CHI_SQUARED_ASSUMPTIONS_UNVERIFIED',
        clusters=None if clusters is None else len(set(clusters)),
        small_cluster_validity='NOT_ESTABLISHED',**_chi_probability(statistic,l-e))
    bound = dict(q,tested_null_exact=list(map(str,null)))
    result.update(_assurance(bound,result))
    return result
