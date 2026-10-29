"""Exact bounded difference-and-level dynamic-panel GMM.

Level moments require additional difference/entity-effect orthogonality and
initial-condition restrictions. Declarations and large p-values do not prove
them. One-step working weights include difference/level cross covariance;
two-step weights aggregate both equations in each entity. No Windmeijer or
formal Arellano--Bond serial-correlation test is implemented here.
"""
from fractions import Fraction as F
from pathlib import Path

from . import dynamic_panel as difference
from . import instrumental_variables as iv
from .execution_control import checkpoint
from .research_evidence import EvidenceLedger,UnknownRegistry,make_evidence,make_obligation,evaluate_coverage
from .workspace import canonical,digest


def _fail(reason): raise ValueError('SYSTEM_GMM_'+reason)


def _validated(q):
    fields={'schema','rows','model','units','period_unit','source_declared_sha256','rights','estimator'}
    if type(q) is not dict or set(q)!=fields or q['schema']!='societybox-system-gmm-v1': _fail('SCHEMA_INVALID')
    base={'outcome','regressors','regressor_timing','instrument_lags','instrument_style'}
    extra={'level_instrument_lags','include_intercept','level_moment_assumptions'}
    model=q['model']
    if type(model) is not dict or set(model)!=base|extra: _fail('MODEL_INVALID')
    if type(model['include_intercept']) is not bool: _fail('INTERCEPT_DECLARATION_INVALID')
    if model['level_moment_assumptions']!='DECLARED_NOT_VERIFIED': _fail('LEVEL_MOMENT_DECLARATION_REQUIRED')
    estimator=q['estimator']
    if type(estimator) is not dict or set(estimator)!={'steps','covariance','initial_entity_variance_ratio'}:
        _fail('ESTIMATOR_INVALID')
    ratio=difference._number(estimator['initial_entity_variance_ratio'])
    if ratio<0: _fail('NONNEGATIVE_WORKING_ENTITY_VARIANCE_REQUIRED')
    legacy=dict(q,schema='societybox-difference-gmm-v1',model={key:model[key] for key in base},
                estimator={key:estimator[key] for key in ('steps','covariance')})
    table=difference._validated(legacy)
    names=[model['outcome']]+model['regressors']; lags=model['level_instrument_lags']
    if (type(lags) is not dict or not set(lags)<=set(names)
            or (not lags and not model['include_intercept'])):
        _fail('LEVEL_INSTRUMENT_LAGS_REQUIRED')
    for name,values in lags.items():
        minimum=1 if name==model['outcome'] or model['regressor_timing'].get(name)=='ENDOGENOUS' else 0
        if (type(values) is not list or not 1<=len(values)<=8 or
                any(type(v) is not int or not minimum<=v<=16 for v in values) or values!=sorted(set(values))):
            _fail('LEVEL_INSTRUMENT_LAGS_INVALID')
    k=1+len(model['regressors'])+int(model['include_intercept'])
    order=['L1:'+model['outcome']]+model['regressors']+(['CONST:LEVEL'] if model['include_intercept'] else [])
    if k>8 or len(set(order))!=k: _fail('PARAMETER_DIMENSION_OR_NAMES_INVALID')
    if len(canonical(q))>262144: _fail('INPUT_SIZE_LIMIT')
    return table,ratio,legacy


def _transform(q,table,legacy):
    diff,excluded,dcols,_=difference._transform(legacy,table,check_sample=False,observation_limit=512)
    model=q['model']; names=model['regressors']; outcome=model['outcome']; intercept=model['include_intercept']
    excluded=[dict(row,equation='DIFFERENCE') for row in excluded]
    levels=[]
    for (entity,t),values in sorted(table.items()):
        checkpoint()
        if (entity,t-1) not in table:
            excluded.append(dict(entity=entity,period=t,equation='LEVEL',reason='REQUIRES_LAGGED_OUTCOME'))
            continue
        levels.append(dict(entity=entity,period=t,equation='LEVEL',y=values[outcome],
            x=[table[entity,t-1][outcome]]+[values[name] for name in names]+([F(1)] if intercept else [])))
    base=[(name,lag) for name in [outcome]+names if name in model['level_instrument_lags']
          for lag in model['level_instrument_lags'][name]]
    def present(row,name,lag):
        return (row['entity'],row['period']-lag) in table and (row['entity'],row['period']-lag-1) in table
    if model['instrument_style']=='COLLAPSED':
        lcols=[dict(variable=name,lag=lag,equation_period=None,kind='LAGGED_DIFFERENCE',equation='LEVEL') for name,lag in base]
    else:
        lcols=[dict(variable=name,lag=lag,equation_period=t,kind='LAGGED_DIFFERENCE',equation='LEVEL')
               for t in sorted({row['period'] for row in levels}) for name,lag in base
               if any(row['period']==t and present(row,name,lag) for row in levels)]
    if intercept: lcols.append(dict(variable='CONST:LEVEL',lag=None,equation_period=None,kind='STANDARD_LEVEL_CONSTANT',equation='LEVEL'))
    columns=[dict(col,equation='DIFFERENCE') for col in dcols]+lcols
    if not 1<=len(columns)<=12: _fail('INSTRUMENT_DIMENSION_LIMIT')
    active=[]
    for row in diff:
        active.append(dict(row,equation='DIFFERENCE',x=row['x']+([F(0)] if intercept else []),
                           z=row['z']+[F(0)]*len(lcols),
                           instrument_available=row['instrument_available']+[False]*len(lcols)))
    for row in levels:
        z=[]; available=[]; entity,t=row['entity'],row['period']
        for col in lcols:
            if col['kind']=='STANDARD_LEVEL_CONSTANT': value,exists=F(1),True
            else:
                name,lag=col['variable'],col['lag']
                exists=col['equation_period'] in (None,t) and present(row,name,lag)
                value=table[entity,t-lag][name]-table[entity,t-lag-1][name] if exists else F(0)
            z.append(value); available.append(exists)
        if not any(available):
            excluded.append(dict(entity=entity,period=t,equation='LEVEL',reason='NO_LEVEL_INSTRUMENT_AVAILABLE'))
        else: active.append(dict(row,z=[F(0)]*len(dcols)+z,instrument_available=[False]*len(dcols)+available))
    n,k=len(active),1+len(names)+int(intercept)
    if n>512 or n<=k or not any(row['equation']=='LEVEL' for row in active): _fail('STACKED_SAMPLE_INVALID')
    entities=sorted({row['entity'] for row in active})
    if len(entities)<2: _fail('AT_LEAST_TWO_ENTITIES_REQUIRED')
    return active,excluded,columns,entities,len(dcols)


def validate(q):
    """Construct and validate declared samples/instruments, without fitting."""
    owned=iv._snapshot(q,limit=262144); table,_,legacy=_validated(owned)
    _transform(owned,table,legacy)
    return q


def _code_receipt():
    return dict(difference._code_receipt(),**{Path(__file__).name:digest(Path(__file__).read_bytes())})


def _assurance(q,result,*,extra_code=None):
    code=_code_receipt(); subject=digest(canonical(q))
    if extra_code: code.update(extra_code)
    scope=dict(method=result['method'],model=q['model'],estimator=q['estimator'],sample=result['sample'],
               units=q['units'],period_unit=q['period_unit'])
    if 'panel_inference_options' in q: scope['panel_inference_options']=q['panel_inference_options']
    variance_note=('Windmeijer remains a finite-sample approximation; a nonrejected test does not establish moment validity.'
                   if result.get('variance_correction')=='WINDMEIJER_2005_FIRST_STEP_PLUGIN' else
                   'Uncorrected two-step covariance is not Windmeijer; a nonrejected C test does not establish moment validity.')
    evidence=make_evidence(subject_sha256=subject,producer=dict(tool_id='econ.system_gmm',version='1',code_sha256=digest(canonical(code))),
        operation=result['method'],authority='LOCAL_EXACT_STATISTIC_AND_BINARY64_REFERENCE_DISTRIBUTION'
            if 'distribution_arithmetic' in result else 'LOCAL_EXACT_ARITHMETIC',
        epistemic_state='DERIVED',proof_kind='DECLARED_SYSTEM_PANEL_GMM',scope=scope,
        coverage=dict(status='PARTIAL',unknown_dimensions=['additional_level_moment_validity','sequential_exogeneity',
            'level_serial_correlation','independent_entities','weak_instruments','finite_sample_variance','source_vintage_replication']),
        result=result,parent_evidence_ids=[],limitations=[
            'Mean-stationarity and difference/entity-effect orthogonality are declared, not established.',
            'The working entity/innovation variance ratio is not an estimate or authenticated sampling design.',
            variance_note,
            'No automatic lag selection or original upstream execution; inference variants are explicitly declared.'])
    ledger=EvidenceLedger([evidence]); registry=UnknownRegistry(ledger); unknowns=[]; obligations=[]
    for claim,question in (
        ('additional_level_moment_validity','Are the initial conditions and E[declared differences * entity effects] = 0 justified for every level instrument?'),
        ('sequential_exogeneity','Are all timing, excluded-lag and level-constant moment conditions justified?'),
        ('level_serial_correlation','Are the innovation serial-correlation restrictions and formal panel serial diagnostics supported?'),
        ('independent_entities','Are entity clusters independent and sufficiently numerous?'),
        ('weak_instruments','Are difference and level instruments informative without harmful proliferation?'),
        ('finite_sample_variance','Have finite-sample and uncorrected two-step variance biases been assessed?'),
        ('source_vintage_replication','Are rights, units, historical availability and independent replication authenticated?')):
        unknown=registry.register(question,scope,'EXTERNAL_SCIENTIFIC_VALIDATION','SYSTEM_PANEL_ASSUMPTION_CHECK')
        unknowns.append(unknown)
        obligations.append(make_obligation(claim_id=claim,subject_sha256=subject,scope=scope,
            required_authority='EXTERNAL_SCIENTIFIC_VALIDATION',required_proof_kind='SYSTEM_PANEL_ASSUMPTION_CHECK',
            verifier_id='econ.system_gmm',dependencies=[],unknown_ids=[unknown['unknown_id']],blocking_reasons=[],next_probe=question))
    return dict(input_sha256=subject,code_sha256=code,evidence=[evidence],unknowns=unknowns,obligations=obligations,
                coverage=evaluate_coverage(obligations,ledger,registry))


def _run(q):
    code=_code_receipt(); table,ratio,legacy=_validated(q)
    rows,excluded,columns,entities,ndcols=_transform(q,table,legacy)
    x,y,z=[row['x'] for row in rows],[row['y'] for row in rows],[row['z'] for row in rows]
    n,k,l=len(rows),len(x[0]),len(z[0]); steps=q['estimator']['steps']; method=q['estimator']['covariance']
    if l<k: _fail('UNDERIDENTIFIED_ORDER_CONDITION')
    initial_moment=difference._initial_moments(rows,entity_variance_ratio=ratio)
    initial=iv._inverse_small(initial_moment,failure='SYSTEM_INITIAL_WEIGHT_SINGULAR')
    beta,residuals,moments,normal,first,weight,covariance,criterion=iv._fit_linear_gmm(
        x,y,z,initial,method,[row['entity'] for row in rows],steps)
    overid=('NOT_APPLICABLE_EXACTLY_IDENTIFIED' if l==k else 'ASYMPTOTIC_CHI_SQUARED_ASSUMPTIONS_UNVERIFIED'
            if steps==2 else 'NOT_CALIBRATED_ONE_STEP_WORKING_WEIGHT')
    probability=iv._chi_probability(criterion,l-k) if steps==2 and l>k else dict(p_value=None,log_p_value=None,p_value_status=overid)
    result=dict(schema='societybox-system-gmm-result-v1',method='DIFFERENCE_AND_LEVEL_SYSTEM_GMM',
        status='COMPUTED_ASSUMPTIONS_UNVERIFIED',implementation_status='INDEPENDENT_IMPLEMENTED',native_execution='NOT_RUN',
        native_parity='NOT_RUN',ui_status='NOT_CONNECTED',identification='IDENTIFICATION_NOT_ESTABLISHED',
        level_moment_validity_established=False,source_authentication='NOT_ESTABLISHED',historical_as_of='NOT_ESTABLISHED',
        parameter_order=['L1:'+q['model']['outcome']]+q['model']['regressors']+(['CONST:LEVEL'] if q['model']['include_intercept'] else []),
        coefficients_exact=list(map(str,beta)),covariance_exact=[list(map(str,row)) for row in covariance],
        covariance_estimator=method,steps=steps,first_step_coefficients_exact=list(map(str,first)),
        initial_moment_covariance_exact=[list(map(str,row)) for row in initial_moment],
        initial_error_covariance='JOINT_IID_INNOVATIONS_PLUS_DECLARED_ENTITY_VARIANCE_INCLUDING_DIFFERENCE_LEVEL_CROSS_TERMS',
        initial_entity_variance_ratio_exact=str(ratio),weight_matrix_exact=[list(map(str,row)) for row in weight],
        weight_convention='SUM_JOINT_ENTITY_MOMENTS_UNCENTERED',
        covariance_correction='CR1_G_OVER_G_MINUS_1_TIMES_STACKED_N_MINUS_1_OVER_N_MINUS_K' if method=='CLUSTER_CR1' else 'CR0_UNADJUSTED',
        moment_sums_exact=list(map(str,moments)),weighted_normal_residual_exact=list(map(str,normal)),gmm_criterion_exact=str(criterion),
        overidentification_test=dict(status=overid,degrees_of_freedom=l-k,**probability),
        sample=dict(level_observations=len(table),stacked_observations=n,entities=len(entities),instrument_count=l,parameter_count=k,
            difference_observations=sum(row['equation']=='DIFFERENCE' for row in rows),
            level_observations_used=sum(row['equation']=='LEVEL' for row in rows),difference_instrument_count=ndcols,
            level_instrument_count=l-ndcols,residual_df=n-k),instrument_order=columns,
        instrument_count_to_entity_ratio_exact=str(F(l,len(entities))),warnings=['INSTRUMENT_COUNT_AT_LEAST_ENTITY_COUNT'] if l>=len(entities) else [],
        transformed_rows=[dict(entity=row['entity'],period=row['period'],equation=row['equation'],outcome_exact=str(row['y']),
            design_exact=list(map(str,row['x'])),instruments_exact=list(map(str,row['z'])),
            instrument_available=row['instrument_available'],residual_exact=str(u)) for row,u in zip(rows,residuals)],excluded_rows=excluded,
        arellano_bond_m_test='NOT_RUN',small_sample_two_step_correction='NOT_APPLIED')
    if steps==2 and l>k: result['distribution_arithmetic']='BINARY64_NOT_INTERVAL_CERTIFIED'
    assurance=_assurance(q,result)
    if assurance['code_sha256']!=code: _fail('CODE_CHANGED_DURING_RUN')
    result.update(assurance)
    return result,rows,code


def calculate(q):
    """One-/two-step system GMM with explicit difference/level instruments.

    schema='societybox-system-gmm-v1' extends the difference model with
    level_instrument_lags (an explicit subset of model variables), boolean include_intercept, and explicit
    level_moment_assumptions='DECLARED_NOT_VERIFIED'. Level differences lag
    >=1 for y/endogenous x, >=0 for predetermined/strictly-exogenous x.
    Each differenced instrument needs both consecutive source periods.
    Estimator additionally declares nonnegative initial_entity_variance_ratio.
    Intercept is zero in differences, one in levels, with a level constant
    instrument. Bounds: 512 input/stacked rows, 8 coefficients, 12 instruments.
    """
    checkpoint(); owned=iv._snapshot(q,limit=262144)
    return _run(owned)[0]


def test_level_moments(q):
    """Common-covariance nested C test for added level moment restrictions.

    Requires two-step system GMM and an identified difference submodel. Uses
    the SAME full first-step entity-moment covariance for both criteria: the
    reduced weight is inverse(S_DD), not the DD block of inverse(S). Refit
    the difference slopes, removing the unidentifiable level intercept.
    df=(L_full-K_full)-(L_diff-K_diff). The exact difference is nonnegative;
    conditional asymptotic chi-square calibration does not prove level moments.
    """
    checkpoint(); owned=iv._snapshot(q,limit=262144); _validated(owned)
    if owned['estimator']['steps']!=2: _fail('C_TEST_REQUIRES_TWO_STEP')
    full,rows,code=_run(owned); ld=full['sample']['difference_instrument_count']; k=1+len(owned['model']['regressors'])
    if ld<k: _fail('C_TEST_DIFFERENCE_SUBMODEL_UNDERIDENTIFIED')
    covariance=iv._inverse_small([list(map(F,row)) for row in full['weight_matrix_exact']])
    sub=[[v for v in row[:ld]] for row in covariance[:ld]]
    weight=iv._inverse_small(sub,failure='C_TEST_REDUCED_WEIGHT_SINGULAR')
    diff=[row for row in rows if row['equation']=='DIFFERENCE']
    x,y,z=[row['x'][:k] for row in diff],[row['y'] for row in diff],[row['z'][:ld] for row in diff]
    try: beta,residuals,moments,normal,_=iv._gmm_fit(x,y,z,weight)
    except ValueError as error:
        if str(error)=='IV_UNDERIDENTIFIED_PROJECTED_DESIGN': _fail('C_TEST_DIFFERENCE_SUBMODEL_UNDERIDENTIFIED')
        raise
    restricted=iv._dot(moments,[iv._dot(row,moments) for row in weight]); joint=F(full['gmm_criterion_exact'])
    statistic=joint-restricted; df=full['sample']['instrument_count']-full['sample']['parameter_count']-(ld-k)
    if statistic<0: _fail('C_TEST_NESTED_CRITERION_IDENTITY_FAILED')
    if df<=0: _fail('C_TEST_NO_ADDITIONAL_RESTRICTIONS')
    result=dict(schema='societybox-system-gmm-level-moment-test-v1',method='COMMON_WEIGHT_NESTED_LEVEL_MOMENT_C_TEST',
        status='COMPUTED_ASSUMPTIONS_UNVERIFIED',implementation_status='INDEPENDENT_IMPLEMENTED',native_execution='NOT_RUN',
        native_parity='NOT_RUN',ui_status='NOT_CONNECTED',identification='IDENTIFICATION_NOT_ESTABLISHED',level_moment_validity_established=False,
        full_gmm_criterion_exact=str(joint),restricted_gmm_criterion_exact=str(restricted),chi_squared_statistic_exact=str(statistic),
        degrees_of_freedom=df,statistic_arithmetic='EXACT_RATIONAL',distribution_arithmetic='BINARY64_NOT_INTERVAL_CERTIFIED',
        reference_distribution='ASYMPTOTIC_CHI_SQUARED_ASSUMPTIONS_UNVERIFIED',full_parameter_count=full['sample']['parameter_count'],
        restricted_parameter_count=k,restricted_coefficients_exact=list(map(str,beta)),
        restricted_weight_matrix_exact=[list(map(str,row)) for row in weight],
        weight_convention='SAME_FULL_FIRST_STEP_MOMENT_COVARIANCE_PRINCIPAL_SUBMATRIX',sample=full['sample'],
        full_fit_input_sha256=full['input_sha256'],full_coefficients_exact=full['coefficients_exact'],
        **iv._chi_probability(statistic,df))
    assurance=_assurance(owned,result)
    if assurance['code_sha256']!=code: _fail('CODE_CHANGED_DURING_RUN')
    result.update(assurance)
    return result
