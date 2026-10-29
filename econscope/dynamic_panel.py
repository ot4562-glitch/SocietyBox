"""Independent Arellano--Bond-style difference GMM on declared panel levels.

First differences remove entity effects. Lagged levels instrument lagged
outcome and declared predetermined/endogenous regressors; strictly exogenous
regressors supply their own differences as standard instruments. Initial
weight is inverse sum(Z_i' D_i D_i' Z_i), not inverse Z'Z. Entity-clustered
one/two-step sandwich inference is asymptotic and lacks Windmeijer correction.
No system equations, interpolation, arbitrary code or upstream execution.
"""
from fractions import Fraction as F
from pathlib import Path
import math
import re

from . import instrumental_variables as iv
from .execution_control import checkpoint
from .research_evidence import (EvidenceLedger,UnknownRegistry,make_evidence,
                                make_obligation,evaluate_coverage)
from .workspace import canonical,digest


def _fail(reason):
    raise ValueError('DYNAMIC_PANEL_'+reason)


def _text(value):
    return iv._text(value) and bool(value.strip())


def _number(value):
    try: return iv._num(value)
    except ValueError as error: _fail('NUMBER_INVALID:'+str(error))


def _validated(q):
    keys={'schema','rows','model','units','period_unit','source_declared_sha256','rights','estimator'}
    if type(q) is not dict or set(q)!=keys or q['schema']!='societybox-difference-gmm-v1': _fail('SCHEMA_INVALID')
    if q['rights']!=dict(local_processing='PERMITTED',report_export='PERMITTED'): _fail('RIGHTS_REQUIRED')
    if type(q['source_declared_sha256']) is not str or not re.fullmatch('[0-9a-f]{64}',q['source_declared_sha256']):
        _fail('SOURCE_INVALID')
    model=q['model']
    if type(model) is not dict or set(model)!={'outcome','regressors','regressor_timing','instrument_lags','instrument_style'}:
        _fail('MODEL_INVALID')
    names=model['regressors']
    if (not _text(model['outcome']) or type(names) is not list or len(names)>7 or
            any(not _text(name) for name in names) or len(set(names+[model['outcome']]))!=len(names)+1):
        _fail('MODEL_NAMES_INVALID')
    if (type(q['units']) is not dict or set(q['units'])!=set(names+[model['outcome']]) or
            any(not _text(unit) for unit in q['units'].values()) or not _text(q['period_unit'])):
        _fail('UNITS_REQUIRED')
    timing=model['regressor_timing']
    if (type(timing) is not dict or set(timing)!=set(names) or
            any(type(v) is not str or v not in ('STRICTLY_EXOGENOUS','PREDETERMINED','ENDOGENOUS') for v in timing.values())):
        _fail('REGRESSOR_TIMING_INVALID')
    lags=model['instrument_lags']; required={model['outcome']}|{name for name in names if timing[name]!='STRICTLY_EXOGENOUS'}
    if type(lags) is not dict or set(lags)!=required: _fail('INSTRUMENT_LAGS_REQUIRED')
    for name,values in lags.items():
        minimum=1 if timing.get(name)=='PREDETERMINED' else 2
        if (type(values) is not list or not 1<=len(values)<=8 or
                any(type(lag) is not int or not minimum<=lag<=16 for lag in values) or values!=sorted(set(values))):
            _fail('INSTRUMENT_LAGS_INVALID')
    style=model['instrument_style']
    if type(style) is not str or style not in ('COLLAPSED','TIME_SPECIFIC'): _fail('INSTRUMENT_STYLE_UNSUPPORTED')
    estimator=q['estimator']
    if (type(estimator) is not dict or set(estimator)!={'steps','covariance'} or type(estimator['steps']) is not int or
            estimator['steps'] not in (1,2) or type(estimator['covariance']) is not str or
            estimator['covariance'] not in ('CLUSTER_CR0','CLUSTER_CR1')):
        _fail('ESTIMATOR_UNSUPPORTED')
    if type(q['rows']) is not list or not 6<=len(q['rows'])<=512: _fail('LEVEL_OBSERVATION_LIMIT')
    if len(canonical(q))>262144: _fail('INPUT_SIZE_LIMIT')
    table={}
    for row in q['rows']:
        if type(row) is not dict or set(row)!={'entity','period','outcome','regressors'}: _fail('ROW_INVALID')
        if not _text(row['entity']) or type(row['period']) is not int or not 0<=row['period']<=1000000:
            _fail('PANEL_INDEX_INVALID')
        key=(row['entity'],row['period'])
        if key in table: _fail('DUPLICATE_ENTITY_PERIOD')
        if type(row['regressors']) is not dict or set(row['regressors'])!=set(names): _fail('REGRESSORS_REQUIRED')
        table[key]={model['outcome']:_number(row['outcome']),**{name:_number(row['regressors'][name]) for name in names}}
    return table


def _transform(q,table,*,check_sample=True,observation_limit=256):
    model=q['model']; outcome=model['outcome']; names=model['regressors']; active=[]; excluded=[]
    for (entity,t),values in sorted(table.items()):
        missing=[period for period in (t-1,t-2) if (entity,period) not in table]
        if missing:
            excluded.append(dict(entity=entity,period=t,reason='REQUIRES_TWO_CONSECUTIVE_LAGS',missing_periods=missing))
            continue
        previous=table[entity,t-1]; before=table[entity,t-2]
        active.append(dict(entity=entity,period=t,y=values[outcome]-previous[outcome],
            x=[previous[outcome]-before[outcome]]+[values[name]-previous[name] for name in names]))
    if not active or len(active)>observation_limit or (check_sample and len(active)<=1+len(names)):
        _fail('DIFFERENCE_SAMPLE_INVALID')
    entities=sorted({row['entity'] for row in active})
    if check_sample and len(entities)<2: _fail('AT_LEAST_TWO_ENTITIES_REQUIRED')
    base=[(name,lag) for name in [outcome]+names if name in model['instrument_lags'] for lag in model['instrument_lags'][name]]
    if model['instrument_style']=='COLLAPSED':
        columns=[dict(variable=name,lag=lag,equation_period=None,kind='LAGGED_LEVEL') for name,lag in base]
    else:
        columns=[dict(variable=name,lag=lag,equation_period=t,kind='LAGGED_LEVEL')
                 for t in sorted({row['period'] for row in active}) for name,lag in base
                 if any(row['period']==t and (row['entity'],t-lag) in table for row in active)]
    columns += [dict(variable=name,lag=None,equation_period=None,kind='STANDARD_DIFFERENCE')
                for name in names if model['regressor_timing'][name]=='STRICTLY_EXOGENOUS']
    if not 1<=len(columns)<=12: _fail('INSTRUMENT_DIMENSION_LIMIT')
    for row in active:
        z=[]; available=[]; entity,t=row['entity'],row['period']
        for column in columns:
            name,lag=column['variable'],column['lag']
            if column['kind']=='STANDARD_DIFFERENCE':
                present=True; value=table[entity,t][name]-table[entity,t-1][name]
            else:
                present=(column['equation_period'] in (None,t) and (entity,t-lag) in table)
                value=table[entity,t-lag][name] if present else F(0)
            z.append(value); available.append(present)
        row.update(z=z,instrument_available=available)
    return active,excluded,columns,entities


def validate(q):
    """Parse and construct the declared sample; never estimate parameters."""
    _transform(q,_validated(q))
    return q


def _initial_moments(rows,*,entity_variance_ratio=F(0)):
    """Exact working error moment covariance via innovation exposures.

    A differenced equation at (entity, t) loads +z on level innovation t
    and -z on innovation t-1. Summing outer products by innovation is
    algebraically identical to diagonal 2*z*z' and adjacent -z*z' terms;
    it also handles gaps and sparse time-specific instrument columns.
    Optional LEVEL rows load +z on innovation t; their entity totals add
    entity_variance_ratio times the corresponding outer products. Shared
    innovation exposures retain difference/level cross terms. The public
    system caller validates the nonnegative ratio; difference-only defaults
    remain Z'DD'Z. Neither D nor an observation-square matrix is constructed.
    """
    width=len(rows[0]['z'])
    exposures={}; entity_totals={}
    for row in rows:
        checkpoint()
        key=(row['entity'],row['period'])
        now=exposures.setdefault(key,[F(0)]*width)
        kind=row.get('equation','DIFFERENCE')
        if kind=='DIFFERENCE': before=exposures.setdefault((row['entity'],row['period']-1),[F(0)]*width)
        elif kind=='LEVEL': before=None
        else: _fail('INITIAL_MOMENT_EQUATION_INVALID')
        total=(entity_totals.setdefault(row['entity'],[F(0)]*width)
               if kind=='LEVEL' and entity_variance_ratio else None)
        for index,value in enumerate(row['z']):
            if value:
                now[index]+=value
                if before is not None: before[index]-=value
                if total is not None: total[index]+=value
    result=[[F(0)]*width for _ in range(width)]
    for group,scale in ((exposures,F(1)),(entity_totals,entity_variance_ratio)):
        if not scale: continue
        for vector in group.values():
            checkpoint()
            nonzero=[(index,value) for index,value in enumerate(vector) if value]
            for index,(i,value) in enumerate(nonzero):
                for j,other in nonzero[index:]:
                    contribution=value*other
                    if scale!=1: contribution*=scale
                    result[i][j]+=contribution
                    if j!=i:result[j][i]+=contribution
    return result


def _code_receipt():
    code=iv._code_receipt(); code[Path(__file__).name]=digest(Path(__file__).read_bytes())
    return code


def _assurance(q,result,*,extra_code=None):
    code=_code_receipt()
    if extra_code: code.update(extra_code)
    subject=digest(canonical(q)); scope=dict(model=q['model'],units=q['units'],period_unit=q['period_unit'],
                                           estimator=q['estimator'],sample=result['sample'])
    if 'panel_inference_options' in q: scope['panel_inference_options']=q['panel_inference_options']
    variance_note=('Windmeijer is a finite-sample variance approximation, not a calibration or identification certificate.'
                   if result.get('variance_correction')=='WINDMEIJER_2005_FIRST_STEP_PLUGIN' else
                   'Uncorrected two-step sandwich is not Windmeijer variance; descriptive residual products are not AR tests.')
    evidence=make_evidence(subject_sha256=subject,
        producer=dict(tool_id='econ.dynamic_panel',version='1',code_sha256=digest(canonical(code))),
        operation=result['method'],authority=('LOCAL_EXACT_AND_PINNED_UPSTREAM_NUMERIC_EXECUTION'
            if result.get('native_execution')=='EXECUTED' else 'LOCAL_EXACT_STATISTIC_AND_BINARY64_REFERENCE_DISTRIBUTION'
            if 'distribution_arithmetic' in result else 'LOCAL_EXACT_ARITHMETIC'),
        epistemic_state='DERIVED',proof_kind='DECLARED_DYNAMIC_PANEL_GMM',scope=scope,
        coverage=dict(status='PARTIAL',unknown_dimensions=['sequential_exogeneity','level_serial_correlation',
                     'instrument_strength','independent_entities','finite_sample_variance','source_vintage_replication']),
        result=result,parent_evidence_ids=[],limitations=[
            'Entity effects are differenced out; timing declarations do not prove moment validity.',
            variance_note,
            'Difference equations only in this operation; no automatic instrument selection or original upstream execution.'])
    ledger=EvidenceLedger([evidence]); registry=UnknownRegistry(ledger); unknowns=[]; obligations=[]
    for claim,question in (
        ('sequential_exogeneity','Are the declared regressor timings and lagged-level exclusion moments justified?'),
        ('level_serial_correlation','Do level-error serial correlation and formal AR(2) diagnostics support these lag instruments?'),
        ('instrument_strength','Are lag instruments informative without harmful proliferation or weak identification?'),
        ('independent_entities','Are entity clusters independent and sufficiently numerous for the reference asymptotics?'),
        ('finite_sample_variance','Are small-sample and two-step variance biases independently assessed?'),
        ('source_vintage_replication','Are source rights, units, historical availability and independent replication verified?')):
        unknown=registry.register(question,scope,'EXTERNAL_SCIENTIFIC_VALIDATION','PANEL_MOMENT_ASSUMPTION_CHECK')
        unknowns.append(unknown)
        obligations.append(make_obligation(claim_id=claim,subject_sha256=subject,scope=scope,
            required_authority='EXTERNAL_SCIENTIFIC_VALIDATION',required_proof_kind='PANEL_MOMENT_ASSUMPTION_CHECK',
            verifier_id='econ.dynamic_panel',dependencies=[],unknown_ids=[unknown['unknown_id']],blocking_reasons=[],next_probe=question))
    return dict(input_sha256=subject,code_sha256=code,evidence=[evidence],unknowns=unknowns,
                obligations=obligations,coverage=evaluate_coverage(obligations,ledger,registry))


def calculate(q):
    """Estimate one-lag entity-effect panel using explicit difference moments."""
    checkpoint()
    q=iv._snapshot(q,limit=262144)
    table=_validated(q); rows,excluded,columns,entities=_transform(q,table)
    y=[row['y'] for row in rows]; x=[row['x'] for row in rows]; z=[row['z'] for row in rows]
    labels=[row['entity'] for row in rows]; n,k,l=len(y),len(x[0]),len(z[0])
    if l<k: _fail('UNDERIDENTIFIED_ORDER_CONDITION')
    moment1=_initial_moments(rows)
    initial=iv._inverse_small(moment1,failure='PANEL_INITIAL_WEIGHT_SINGULAR')
    steps=q['estimator']['steps']; method=q['estimator']['covariance']
    beta,residuals,moments,normal,first_beta,weight,covariance,criterion=iv._fit_linear_gmm(x,y,z,initial,method,labels,steps)
    overid=('NOT_APPLICABLE_EXACTLY_IDENTIFIED' if l==k else
            'ASYMPTOTIC_CHI_SQUARED_ASSUMPTIONS_UNVERIFIED' if steps==2 else 'NOT_CALIBRATED_ONE_STEP_WEIGHT')
    probability=iv._chi_probability(criterion,l-k) if steps==2 and l>k else dict(p_value=None,log_p_value=None,p_value_status=overid)
    residual_map={(row['entity'],row['period']):e for row,e in zip(rows,residuals)}
    products=[]
    for lag in (1,2):
        pairs=[(e,residual_map[(entity,t-lag)]) for (entity,t),e in residual_map.items() if (entity,t-lag) in residual_map]
        products.append(dict(lag=lag,pairs=len(pairs),product_sum_exact=str(sum((a*b for a,b in pairs),F(0))),
                             scope='DESCRIPTIVE_NOT_AN_ARELLANO_BOND_M_TEST'))
    transformed=[dict(entity=row['entity'],period=row['period'],delta_outcome_exact=str(row['y']),
        design_exact=list(map(str,row['x'])),instruments_exact=list(map(str,row['z'])),
        instrument_available=row['instrument_available'],residual_exact=str(e)) for row,e in zip(rows,residuals)]
    result=dict(schema='societybox-difference-gmm-result-v1',method='ARELLANO_BOND_STYLE_DIFFERENCE_GMM',
        status='COMPUTED_ASSUMPTIONS_UNVERIFIED',implementation_status='INDEPENDENT_IMPLEMENTED',
        native_execution='NOT_RUN',native_parity='NOT_RUN',ui_status='NOT_CONNECTED',
        identification='IDENTIFICATION_NOT_ESTABLISHED',source_authentication='NOT_ESTABLISHED',historical_as_of='NOT_ESTABLISHED',
        parameter_order=['L1:'+q['model']['outcome']]+q['model']['regressors'],coefficients_exact=list(map(str,beta)),
        covariance_exact=[list(map(str,row)) for row in covariance],covariance_estimator=method,steps=steps,
        first_step_coefficients_exact=list(map(str,first_beta)),initial_moment_covariance_exact=[list(map(str,row)) for row in moment1],
        weight_matrix_exact=[list(map(str,row)) for row in weight],weight_convention='SUM_ENTITY_MOMENTS_UNCENTERED',
        moment_sums_exact=list(map(str,moments)),weighted_normal_residual_exact=list(map(str,normal)),
        gmm_criterion_exact=str(criterion),overidentification_test=dict(status=overid,degrees_of_freedom=l-k,**probability),
        sample=dict(level_observations=len(table),difference_observations=n,entities=len(entities),
                    entities_in_input=len({entity for entity,_ in table}),instrument_count=l,parameter_count=k,residual_df=n-k),
        instrument_order=columns,instrument_count_to_entity_ratio_exact=str(F(l,len(entities))),
        warnings=['INSTRUMENT_COUNT_AT_LEAST_ENTITY_COUNT'] if l>=len(entities) else [],
        transformed_rows=transformed,excluded_rows=excluded,residual_lag_products=products,
        arellano_bond_m_test='NOT_RUN',small_sample_two_step_correction='NOT_APPLIED',
        system_gmm='NOT_REQUESTED_USE_SYSTEM_GMM_API')
    if steps==2 and l>k: result['distribution_arithmetic']='BINARY64_NOT_INTERVAL_CERTIFIED'
    result.update(_assurance(q,result))
    return result


def compare_highs(q,coefficient_bounds,runtime):
    """Pinned QP oracle conditional on the independently constructed GMM weight.

    Does not independently build panel instruments, estimate two-step weights,
    validate moments or compute native panel covariance. Bounds must contain the
    unconstrained solution; existing native QP numeric/dimension limits apply.
    """
    from . import highs_native
    if runtime is None: _fail('PINNED_RUNTIME_REQUIRED')
    caller_request=q
    q=iv._snapshot(q,limit=262144)
    bound=dict(q,native_coefficient_bounds=coefficient_bounds)
    declared=digest(canonical(bound)); code=_code_receipt()
    independent=calculate(q); k=len(independent['coefficients_exact'])
    if k>4: _fail('NATIVE_DIMENSION_BLOCKED')
    if (type(coefficient_bounds) is not list or len(coefficient_bounds)!=k or
            any(type(pair) is not list or len(pair)!=2 for pair in coefficient_bounds)):
        _fail('NATIVE_BOUNDS_INVALID')
    low=[_number(pair[0]) for pair in coefficient_bounds]; high=[_number(pair[1]) for pair in coefficient_bounds]
    beta=list(map(F,independent['coefficients_exact']))
    if any(lo>=hi or not lo<=b<=hi for lo,hi,b in zip(low,high,beta)): _fail('NATIVE_BOUNDS_EXCLUDE_ESTIMATE')
    rows=independent['transformed_rows']
    x=[list(map(F,row['design_exact'])) for row in rows]
    z=[list(map(F,row['instruments_exact'])) for row in rows]; y=[F(row['delta_outcome_exact']) for row in rows]
    weight=[list(map(F,row)) for row in independent['weight_matrix_exact']]
    zx=iv._cross(z,x); zy=[iv._dot(col,y) for col in zip(*z)]
    mapping=iv._multiply([list(row) for row in zip(*zx)],weight)
    gram=iv._multiply(mapping,zx); rhs=[iv._dot(row,zy) for row in mapping]
    program=dict(kind='QP',sense='minimize',objective=list(map(str,[-2*v for v in rhs])),
        constraints=[],row_lower=[],row_upper=[],lower=list(map(str,low)),upper=list(map(str,high)),
        integrality=[0]*k,hessian=[[str(2*v) for v in row] for row in gram],
        offset=str(iv._dot(zy,[iv._dot(row,zy) for row in weight])))
    native=highs_native.run(program,runtime); actual=native['actual']
    if actual['status']!='OPTIMAL' or len(actual['solution'])!=k: _fail('NATIVE_OPTIMUM_REQUIRED')
    errors=[abs(float(b)-v) for b,v in zip(beta,actual['solution'])]
    objective_error=abs(float(F(independent['gmm_criterion_exact']))-actual['objective'])
    if (any(not math.isfinite(error) or error>highs_native.ABS_TOL+highs_native.REL_TOL*abs(float(b))
            for error,b in zip(errors,beta)) or not math.isfinite(objective_error) or
            objective_error>highs_native.ABS_TOL+highs_native.REL_TOL*abs(float(F(independent['gmm_criterion_exact'])))):
        _fail('NATIVE_PARITY_FAILED')
    try:
        unchanged=(digest(canonical(dict(q,native_coefficient_bounds=coefficient_bounds)))==declared
                   and digest(canonical(caller_request))==digest(canonical(q))
                   and _code_receipt()==code)
    except (ValueError,TypeError,OverflowError,RecursionError):
        unchanged=False
    if not unchanged: _fail('INPUT_OR_CODE_CHANGED_DURING_RUN')
    bound['native_runtime_pin']=native['upstream']['runtime_pin']
    result=dict(schema='societybox-panel-gmm-highs-parity-v1',method='DIFFERENCE_GMM_FIXED_WEIGHT_QP_COMPARISON',
        implementation_status='NUMERICAL_PARITY_VERIFIED',native_execution='EXECUTED',upstream_status='ORIGINAL_API_EXECUTED',
        identification='IDENTIFICATION_NOT_ESTABLISHED',sample=independent['sample'],independent=independent,native=native,
        native_panel_covariance='NOT_RUN',parity_scope='DECLARED_FIXED_WEIGHT_GMM_COEFFICIENTS_AND_OBJECTIVE_ONLY',
        parity=dict(maximum_coefficient_error=max(errors),objective_error=objective_error,
                    absolute_tolerance=highs_native.ABS_TOL,relative_tolerance=highs_native.REL_TOL))
    result.update(_assurance(bound,result))
    return result
