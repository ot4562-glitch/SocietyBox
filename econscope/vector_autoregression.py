"""Independent bounded exact VAR(p), forecasts, recursive IRF and FEVD.

OLS on a declared cutoff; no automatic lag selection or missing-value filling.
Strict Schur stability uses P-C'PC=I and exact positive-definiteness, not a
rounded eigenvalue threshold. Recursive decomposition requires explicit ordering.
Rational LDL factors keep FEVD accounting exact; only IRF square roots use
Decimal approximations. Small-problem reference core, not a large-data solver.
"""
from decimal import (Decimal,localcontext,Context,ROUND_HALF_EVEN,
                     InvalidOperation,DivisionByZero,Overflow)
from fractions import Fraction as F
from pathlib import Path
import re

from . import instrumental_variables as iv
from .execution_control import checkpoint
from .research_evidence import (EvidenceLedger,UnknownRegistry,make_evidence,
                                make_obligation,evaluate_coverage)
from .workspace import canonical,digest


def _fail(reason): raise ValueError('VAR_'+reason)


def _text(value): return iv._text(value) and bool(value.strip())


def _bounded(values):
    for value in values:
        if isinstance(value,list): _bounded(value)
        elif max(value.numerator.bit_length(),value.denominator.bit_length())>8192:
            _fail('EXACT_ARITHMETIC_RESOURCE_LIMIT')
    return values


def _identity(n): return [[F(int(i==j)) for j in range(n)] for i in range(n)]


def _transpose(matrix): return [list(row) for row in zip(*matrix)]


def _matrix_strings(matrix): return [list(map(str,row)) for row in matrix]


def _validated(q):
    keys={'schema','variables','observations','period_unit','estimation_end_period','model','identification','units','source_declared_sha256','rights'}
    if type(q) is not dict or set(q)!=keys or q['schema']!='societybox-vector-autoregression-v1': _fail('SCHEMA_INVALID')
    if q['rights']!=dict(local_processing='PERMITTED',report_export='PERMITTED'): _fail('RIGHTS_REQUIRED')
    if type(q['source_declared_sha256']) is not str or not re.fullmatch('[a-f0-9]{64}',q['source_declared_sha256']): _fail('SOURCE_INVALID')
    names=q['variables']
    if type(names) is not list or not 1<=len(names)<=3 or any(not _text(v) for v in names) or len(set(names))!=len(names):
        _fail('VARIABLES_INVALID')
    if (type(q['units']) is not dict or set(q['units'])!=set(names) or any(not _text(v) for v in q['units'].values())
            or not _text(q['period_unit'])): _fail('UNITS_REQUIRED')
    model=q['model']
    if (type(model) is not dict or set(model)!={'lags','intercept','covariance_divisor'} or type(model['lags']) is not int or
            not 1<=model['lags']<=3 or len(names)*model['lags']>6 or type(model['intercept']) is not bool or
            type(model['covariance_divisor']) is not str or model['covariance_divisor'] not in ('RESIDUAL_DF','ML')):
        _fail('MODEL_INVALID')
    ident=q['identification']
    if ident is not None:
        if (type(ident) is not dict or set(ident)!={'scheme','ordering'} or ident['scheme']!='RECURSIVE_CHOLESKY' or
                type(ident['ordering']) is not list or len(ident['ordering'])!=len(names) or
                any(type(v) is not str for v in ident['ordering']) or set(ident['ordering'])!=set(names)):
            _fail('IDENTIFICATION_INVALID')
    if type(q['observations']) is not list or not 6<=len(q['observations'])<=128: _fail('OBSERVATION_LIMIT')
    table={}
    for row in q['observations']:
        if type(row) is not dict or set(row)!={'period','values'}: _fail('ROW_INVALID')
        period=row['period']
        if type(period) is not int or not 0<=period<=1000000 or period in table: _fail('PERIOD_INVALID_OR_DUPLICATE')
        if type(row['values']) is not dict or set(row['values'])!=set(names): _fail('VARIABLE_VALUES_REQUIRED')
        try: table[period]=[iv._num(row['values'][name]) for name in names]
        except ValueError as error: _fail('NUMBER_INVALID:'+str(error))
    end=q['estimation_end_period']
    if type(end) is not int or end not in table: _fail('CUTOFF_INVALID')
    periods=sorted(t for t in table if t<=end)
    if any(b-a!=1 for a,b in zip(periods,periods[1:])): _fail('CONTIGUOUS_TRAINING_PERIODS_REQUIRED')
    p=model['lags']; k=int(model['intercept'])+len(names)*p
    if len(periods)-p<=k: _fail('INSUFFICIENT_RESIDUAL_DEGREES_OF_FREEDOM')
    if len(canonical(q))>131072: _fail('INPUT_SIZE_LIMIT')
    return periods,[table[t] for t in periods]


def validate(q):
    """Validate declarations/sample only; no fit or dynamics during review."""
    _validated(q)
    return q


def _ldl(matrix):
    n=len(matrix); lower=_identity(n); diagonal=[]
    for j in range(n):
        value=matrix[j][j]-sum((lower[j][k]**2*diagonal[k] for k in range(j)),F(0))
        if value<=0: return None
        diagonal.append(value)
        for i in range(j+1,n):
            lower[i][j]=(matrix[i][j]-sum((lower[i][k]*lower[j][k]*diagonal[k] for k in range(j)),F(0)))/value
    _bounded(lower); _bounded(diagonal)
    return lower,diagonal


def _lyapunov(companion,forcing):
    """Exact symmetric solution of P-C'PC=forcing, or None if singular."""
    size=len(companion)
    pairs=[(i,j) for i in range(size) for j in range(i,size)]; operator=[]
    for a,b in pairs:
        checkpoint()
        operator.append([F(int((a,b)==(i,j)))-companion[i][a]*companion[j][b]
                         -(companion[j][a]*companion[i][b] if i!=j else F(0)) for i,j in pairs])
    solution=iv._solve(operator,[forcing[a][b] for a,b in pairs])
    if solution is None: return None
    _bounded(solution); matrix=[[F(0)]*size for _ in range(size)]
    for (i,j),value in zip(pairs,solution): matrix[i][j]=matrix[j][i]=value
    product=iv._multiply(iv._multiply(_transpose(companion),matrix),companion)
    if any(matrix[i][j]-product[i][j]!=forcing[i][j] for i in range(size) for j in range(size)):
        _fail('LYAPUNOV_IDENTITY_FAILED')
    return matrix


def _stability(lags):
    m=len(lags[0]); size=m*len(lags)
    companion=[[value for block in lags for value in block[i]] for i in range(m)]
    companion += [[F(int(j==i)) for j in range(size)] for i in range(size-m)]
    matrix=_lyapunov(companion,_identity(size))
    common=dict(method='EXACT_DISCRETE_LYAPUNOV_P_MINUS_C_TRANSPOSE_P_C_EQUALS_I',companion_exact=_matrix_strings(companion))
    if matrix is None:
        return dict(common,status='NOT_SCHUR_STABLE',reason='LYAPUNOV_OPERATOR_SINGULAR',lyapunov_solution_exact=None)
    stable=_ldl(matrix) is not None
    return dict(common,status='SCHUR_STABLE' if stable else 'NOT_SCHUR_STABLE',
                reason=None if stable else 'LYAPUNOV_SOLUTION_NOT_POSITIVE_DEFINITE',lyapunov_solution_exact=_matrix_strings(matrix),
                maximum_identity_residual_exact='0')


def _fit(q):
    checkpoint()
    periods,values=_validated(q); m=len(q['variables']); p=q['model']['lags']; constant=int(q['model']['intercept'])
    design=[[F(1)]*constant+[v for lag in range(1,p+1) for v in values[t-lag]] for t in range(p,len(values))]
    targets=values[p:]; gram=iv._cross(design,design); rhs=iv._cross(design,targets)
    coefficients=[iv._solve(gram,list(col)) for col in zip(*rhs)]
    if any(row is None for row in coefficients): _fail('DESIGN_RANK_DEFICIENT')
    _bounded(coefficients)
    residuals=[[target[j]-iv._dot(row,coefficients[j]) for j in range(m)] for row,target in zip(design,targets)]
    normal=iv._cross(design,residuals)
    if any(v for row in normal for v in row): _fail('NORMAL_EQUATIONS_FAILED')
    n,k=len(design),len(design[0]); divisor=n-k if q['model']['covariance_divisor']=='RESIDUAL_DF' else n
    covariance=[[v/divisor for v in row] for row in iv._cross(residuals,residuals)]; _bounded(covariance)
    lags=[[[coefficients[i][constant+lag*m+j] for j in range(m)] for i in range(m)] for lag in range(p)]
    intercept=[row[0] if constant else F(0) for row in coefficients]
    training=dict(start_period=periods[0],end_period=periods[-1],level_observations=len(values),
                  estimation_observations=n,initial_condition_observations=p,parameter_count_per_equation=k)
    return dict(values=values,coefficients=coefficients,lags=lags,intercept=intercept,covariance=covariance,normal=normal,residuals=residuals,
                training=training,df=n-k,divisor=divisor,stability=_stability(lags))


def _fit_result(q,raw):
    return dict(schema='societybox-var-result-v1',method='DECLARED_OLS_VAR',
        status='COMPUTED_ASSUMPTIONS_UNVERIFIED' if raw['stability']['status']=='SCHUR_STABLE' else 'ESTIMATED_NONSTATIONARY',
        implementation_status='INDEPENDENT_IMPLEMENTED',native_execution='NOT_RUN',native_parity='NOT_RUN',ui_status='NOT_CONNECTED',
        variables=list(q['variables']),training=raw['training'],parameter_order=(['constant'] if q['model']['intercept'] else [])+
            ['L'+str(lag)+':'+name for lag in range(1,q['model']['lags']+1) for name in q['variables']],
        equation_coefficients_exact=_matrix_strings(raw['coefficients']),intercept_exact=list(map(str,raw['intercept'])),
        lag_matrices_exact=[_matrix_strings(matrix) for matrix in raw['lags']],residual_covariance_exact=_matrix_strings(raw['covariance']),
        innovation_covariance_positive_definite=_ldl(raw['covariance']) is not None,
        covariance_divisor=raw['divisor'],residual_degrees_of_freedom=raw['df'],normal_equations_exact=_matrix_strings(raw['normal']),
        stability=raw['stability'],structural_causal_identification='NOT_ESTABLISHED',historical_as_of='NOT_ESTABLISHED',
        parameter_uncertainty='NOT_INCLUDED')


def _assurance(q,result):
    code=iv._code_receipt(); code[Path(__file__).name]=digest(Path(__file__).read_bytes()); subject=digest(canonical(q))
    scope=dict(variables=q['variables'],units=q['units'],model=q['model'],training=result['training'],identification=q['identification'])
    if 'horizon' in result: scope['horizon']=result['horizon']
    evidence=make_evidence(subject_sha256=subject,producer=dict(tool_id='econ.var',version='1',code_sha256=digest(canonical(code))),
        operation=result['method'],authority='LOCAL_EXACT_ARITHMETIC_AND_DECLARED_DECIMAL_ROOTS' if 'orthogonalized_impulses' in result else 'LOCAL_EXACT_ARITHMETIC',
        epistemic_state='DERIVED',proof_kind='DECLARED_VAR_CALCULATION',scope=scope,
        coverage=dict(status='PARTIAL',unknown_dimensions=['innovation_specification','recursive_identification','parameter_uncertainty',
                     'historical_information_set','external_replication']),result=result,parent_evidence_ids=[],
        limitations=['A fitted stable recurrence is not evidence of a stable real economy or causal shock identification.',
                     'Forecast covariance conditions on estimated coefficients and declared innovation covariance.',
                     'Only observations through the declared cutoff enter fitting; source release availability is unverified.'])
    ledger=EvidenceLedger([evidence]); registry=UnknownRegistry(ledger); unknowns=[]; obligations=[]
    for claim,question in (
        ('innovation_specification','Are the model lag order, stable parameters and innovation dependence suitable for this use?'),
        ('recursive_identification','Does independent economic evidence justify any declared recursive innovation ordering?'),
        ('parameter_uncertainty','How much estimation, specification and parameter uncertainty is omitted from these conditional results?'),
        ('historical_information_set','Were the declared data and vintages actually available at the estimation origin?'),
        ('external_replication','Can independent software and data provenance replicate and challenge this result?')):
        unknown=registry.register(question,scope,'EXTERNAL_SCIENTIFIC_VALIDATION','VAR_DESIGN_CHECK'); unknowns.append(unknown)
        obligations.append(make_obligation(claim_id=claim,subject_sha256=subject,scope=scope,required_authority='EXTERNAL_SCIENTIFIC_VALIDATION',
            required_proof_kind='VAR_DESIGN_CHECK',verifier_id='econ.var',dependencies=[],unknown_ids=[unknown['unknown_id']],blocking_reasons=[],next_probe=question))
    return dict(input_sha256=subject,code_sha256=code,evidence=[evidence],unknowns=unknowns,obligations=obligations,
                coverage=evaluate_coverage(obligations,ledger,registry))


def calculate(q):
    q=iv._snapshot(q)
    result=_fit_result(q,_fit(q)); result.update(_assurance(q,result)); return result


def _horizon(value,zero=False):
    if type(value) is not int or not (0 if zero else 1)<=value<=64: _fail('HORIZON_INVALID')


def _stable(raw):
    if raw['stability']['status']!='SCHUR_STABLE': _fail('NONSTATIONARY_DYNAMICS_BLOCKED')


def _ma(raw,horizon):
    m=len(raw['intercept']); result=[_identity(m)]
    for h in range(1,horizon+1):
        checkpoint(); matrix=[[F(0)]*m for _ in range(m)]
        for lag in range(1,min(h,len(raw['lags']))+1):
            contribution=iv._multiply(raw['lags'][lag-1],result[h-lag])
            matrix=[[a+b for a,b in zip(row,other)] for row,other in zip(matrix,contribution)]
        result.append(_bounded(matrix))
    return result


def _covariance_path(raw,ma):
    m=len(raw['intercept']); current=[[F(0)]*m for _ in range(m)]; result=[]
    for psi in ma:
        contribution=iv._multiply(iv._multiply(psi,raw['covariance']),_transpose(psi))
        current=_bounded([[a+b for a,b in zip(row,other)] for row,other in zip(current,contribution)])
        result.append(current)
    return result


def forecast(q,horizon):
    q=iv._snapshot(q)
    _horizon(horizon); raw=_fit(q); _stable(raw); history=list(raw['values']); means=[]
    for _ in range(horizon):
        point=list(raw['intercept'])
        for lag,matrix in enumerate(raw['lags'],1):
            point=[a+iv._dot(row,history[-lag]) for a,row in zip(point,matrix)]
        _bounded(point); history.append(point); means.append(list(map(str,point)))
    covariance=_covariance_path(raw,_ma(raw,horizon-1)); origin=q['estimation_end_period']
    result=dict(schema='societybox-var-forecast-v1',method='CONDITIONAL_VAR_FORECAST',horizon=horizon,origin_period=origin,
        periods=list(range(origin+1,origin+horizon+1)),means_exact=means,covariances_exact=[_matrix_strings(v) for v in covariance],
        training=raw['training'],fit=_fit_result(q,raw),implementation_status='INDEPENDENT_IMPLEMENTED',native_execution='NOT_RUN',
        parameter_uncertainty='NOT_INCLUDED',historical_as_of='NOT_ESTABLISHED',structural_causal_identification='NOT_ESTABLISHED')
    result.update(_assurance(dict(q,requested_operation='forecast',horizon=horizon),result)); return result


def _orthogonal_basis(q,raw):
    if q['identification'] is None: _fail('RECURSIVE_IDENTIFICATION_REQUIRED')
    order=list(q['identification']['ordering']); permutation=[q['variables'].index(name) for name in order]
    covariance=[[raw['covariance'][i][j] for j in permutation] for i in permutation]
    factors=_ldl(covariance)
    if factors is None: _fail('INNOVATION_COVARIANCE_POSITIVE_DEFINITE_REQUIRED')
    lower,diagonal=factors
    return [lower[permutation.index(i)] for i in range(len(permutation))],diagonal,order


def impulse_responses(q,horizon,*,orthogonalized=False):
    _horizon(horizon,zero=True)
    if type(orthogonalized) is not bool: _fail('ORTHOGONALIZATION_FLAG_INVALID')
    q=iv._snapshot(q)
    raw=_fit(q); _stable(raw); ma=_ma(raw,horizon)
    result=dict(schema='societybox-var-irf-v1',method='VAR_MOVING_AVERAGE_RESPONSES',horizon=horizon,
        training=raw['training'],response_order=list(q['variables']),shock_order=list(q['variables']),ma_matrices_exact=[_matrix_strings(v) for v in ma],
        implementation_status='INDEPENDENT_IMPLEMENTED',native_execution='NOT_RUN',structural_causal_identification='NOT_ESTABLISHED')
    if orthogonalized:
        lower,diagonal,order=_orthogonal_basis(q,raw); impulses=[]
        with localcontext(Context(prec=80,rounding=ROUND_HALF_EVEN,Emin=-999999,Emax=999999,
                                  traps=[InvalidOperation,DivisionByZero,Overflow])):
            roots=[(Decimal(v.numerator)/Decimal(v.denominator)).sqrt() for v in diagonal]
            for matrix in ma:
                multipliers=iv._multiply(matrix,lower); _bounded(multipliers)
                impulses.append([[dict(multiplier_exact=str(v),innovation_variance_exact=str(diagonal[j]),
                    decimal=str(Decimal(v.numerator)/Decimal(v.denominator)*roots[j])) for j,v in enumerate(row)] for row in multipliers])
        result.update(orthogonalized_impulses=impulses,shock_order=order,decimal_precision=80,decimal_rounding='ROUND_HALF_EVEN',
                      algebraic_form='MULTIPLIER_TIMES_SQRT_INNOVATION_VARIANCE')
    result.update(_assurance(dict(q,requested_operation='irf',horizon=horizon,orthogonalized=orthogonalized),result)); return result


def variance_decomposition(q,horizon):
    q=iv._snapshot(q)
    _horizon(horizon); raw=_fit(q); _stable(raw); lower,diagonal,order=_orthogonal_basis(q,raw)
    ma=_ma(raw,horizon-1); covariance=_covariance_path(raw,ma); m=len(diagonal)
    current=[[F(0)]*m for _ in range(m)]; shares=[]; contributions=[]
    for psi,cov in zip(ma,covariance):
        mapped=iv._multiply(psi,lower)
        current=_bounded([[v+diagonal[j]*mapped[i][j]**2 for j,v in enumerate(row)] for i,row in enumerate(current)])
        if any(sum(row)!=cov[i][i] for i,row in enumerate(current)): _fail('FORECAST_VARIANCE_ACCOUNTING_FAILED')
        if any(cov[i][i]<=0 for i in range(m)): _fail('FORECAST_VARIANCE_ZERO')
        shares.append([[str(v/cov[i][i]) for v in row] for i,row in enumerate(current)])
        contributions.append(_matrix_strings(current))
    result=dict(schema='societybox-var-fevd-v1',method='RECURSIVE_VAR_FORECAST_ERROR_VARIANCE_DECOMPOSITION',horizon=horizon,
        training=raw['training'],response_order=list(q['variables']),shock_order=order,shares_exact=shares,
        variance_contributions_exact=contributions,total_forecast_variance_exact=[[str(row[i]) for i,row in enumerate(v)] for v in covariance],
        implementation_status='INDEPENDENT_IMPLEMENTED',native_execution='NOT_RUN',structural_causal_identification='NOT_ESTABLISHED',
        parameter_uncertainty='NOT_INCLUDED')
    result.update(_assurance(dict(q,requested_operation='fevd',horizon=horizon),result)); return result


def historical_decomposition(q,*,orthogonalized=False):
    """Ex-post innovations accounting using the full declared estimation sample.

    Initial p levels and fitted intercept form the no-new-innovation baseline.
    Shock columns propagate recursively. Recursive components use unscaled LDL
    innovations, so contributions and the accounting identity stay rational.
    These are not real-time forecasts or identified causes of observed history.
    """
    if type(orthogonalized) is not bool: _fail('ORTHOGONALIZATION_FLAG_INVALID')
    q=iv._snapshot(q)
    raw=_fit(q); _stable(raw); m=len(q['variables']); p=q['model']['lags']
    if orthogonalized:
        basis,diagonal,order=_orthogonal_basis(q,raw)
    else:
        basis=_identity(m); diagonal=None; order=list(q['variables'])
    baselines=list(raw['values'][:p]); components=[[[F(0)]*m for _ in range(m)] for _ in range(p)]
    baselines_out=[]; components_out=[]; innovations=[]
    for t,residual in enumerate(raw['residuals'],p):
        checkpoint(); baseline=list(raw['intercept']); matrix=[[F(0)]*m for _ in range(m)]
        for lag,coefficient in enumerate(raw['lags'],1):
            baseline=[v+iv._dot(row,baselines[-lag]) for v,row in zip(baseline,coefficient)]
            propagated=iv._multiply(coefficient,components[-lag])
            matrix=[[v+w for v,w in zip(row,other)] for row,other in zip(matrix,propagated)]
        innovation=iv._solve(basis,residual)
        if innovation is None: _fail('RECURSIVE_INNOVATION_INVERSE_FAILED')
        matrix=_bounded([[v+basis[i][j]*innovation[j] for j,v in enumerate(row)] for i,row in enumerate(matrix)])
        _bounded(baseline); _bounded(innovation)
        if any(v+sum(row,F(0))!=target for v,row,target in zip(baseline,matrix,raw['values'][t])):
            _fail('HISTORICAL_ACCOUNTING_IDENTITY_FAILED')
        baselines.append(baseline); components.append(matrix)
        baselines_out.append(list(map(str,baseline))); components_out.append(_matrix_strings(matrix)); innovations.append(list(map(str,innovation)))
    start=raw['training']['start_period']; end=raw['training']['end_period']
    result=dict(schema='societybox-var-historical-decomposition-v1',method='EX_POST_VAR_INNOVATION_ACCOUNTING',
        training=raw['training'],periods=list(range(start+p,end+1)),initial_condition_periods=list(range(start,start+p)),
        information_scope='EX_POST_FULL_ESTIMATION_SAMPLE',baseline_definition='ACTUAL_INITIAL_LEVELS_AND_FITTED_INTERCEPT_NO_NEW_INNOVATIONS',
        response_order=list(q['variables']),shock_order=order,orthogonalized=orthogonalized,
        baseline_exact=baselines_out,contributions_exact=components_out,innovations_exact=innovations,
        innovation_variance_exact=None if diagonal is None else list(map(str,diagonal)),
        observed_exact=_matrix_strings(raw['values'][p:]),maximum_accounting_residual_exact='0',
        implementation_status='INDEPENDENT_IMPLEMENTED',native_execution='NOT_RUN',
        structural_causal_identification='NOT_ESTABLISHED',historical_as_of='NOT_ESTABLISHED',parameter_uncertainty='NOT_INCLUDED')
    result.update(_assurance(dict(q,requested_operation='historical_decomposition',orthogonalized=orthogonalized),result)); return result


def stationary_moments(q):
    """Conditional stationary mean, long-run multiplier and innovation covariance.

    Moments assume a stable, correctly specified VAR with white innovations and
    the fitted coefficients held fixed. The state covariance solves V-CVC'=Q;
    no truncated infinite sum, stationarity of actual data or causal claim.
    """
    q=iv._snapshot(q)
    raw=_fit(q); _stable(raw); m=len(q['variables']); summed=[[F(0)]*m for _ in range(m)]
    for matrix in raw['lags']: summed=[[a+b for a,b in zip(row,other)] for row,other in zip(summed,matrix)]
    operator=[[F(int(i==j))-v for j,v in enumerate(row)] for i,row in enumerate(summed)]
    multiplier=iv._inverse_small(operator,failure='VAR_LONG_RUN_OPERATOR_SINGULAR'); _bounded(multiplier)
    if iv._multiply(operator,multiplier)!=_identity(m): _fail('LONG_RUN_MULTIPLIER_IDENTITY_FAILED')
    mean=_bounded([iv._dot(row,raw['intercept']) for row in multiplier])
    companion=[[F(v) for v in row] for row in raw['stability']['companion_exact']]; size=len(companion)
    forcing=[[raw['covariance'][i][j] if i<m and j<m else F(0) for j in range(size)] for i in range(size)]
    state=_lyapunov(_transpose(companion),forcing)
    if state is None: _fail('STATIONARY_COVARIANCE_SINGULAR_OPERATOR')
    covariance=[row[:m] for row in state[:m]]
    result=dict(schema='societybox-var-stationary-moments-v1',method='CONDITIONAL_VAR_STATIONARY_MOMENTS',training=raw['training'],
        long_run_multiplier_exact=_matrix_strings(multiplier),conditional_steady_mean_exact=list(map(str,mean)),
        unconditional_covariance_exact=_matrix_strings(covariance),companion_covariance_exact=_matrix_strings(state),
        maximum_covariance_identity_residual_exact='0',implementation_status='INDEPENDENT_IMPLEMENTED',native_execution='NOT_RUN',
        parameter_uncertainty='NOT_INCLUDED',structural_causal_identification='NOT_ESTABLISHED',
        stationary_data_generating_process='NOT_ESTABLISHED')
    result.update(_assurance(dict(q,requested_operation='stationary_moments'),result)); return result
