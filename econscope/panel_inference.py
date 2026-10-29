"""Exact panel-GMM AR(m) diagnostics and Windmeijer variance correction.

AR(m) implements the robust Arellano--Bond three-term variance with an
explicit coherent residual-covariance plugin basis. Numerator/lag residuals
always use the requested final fit. FIRST_STEP changes the covariance basis,
not that numerator; this is not a claim of DPD/plm/Stata default parity.
Windmeijer uses the first-step moment covariance, not the final-residual
sandwich reported by the uncorrected estimator. No new estimator is wrapped
around external code; both paths reuse the existing panel/GMM core.
"""
from fractions import Fraction as F
from pathlib import Path
import math

from . import dynamic_panel as difference
from . import system_gmm as system
from . import instrumental_variables as iv
from .execution_control import checkpoint
from .workspace import digest


def _fail(reason): raise ValueError('PANEL_INFERENCE_'+reason)


def _extra_code():
    return {name:digest(Path(__file__).with_name(name).read_bytes()) for name in ('system_gmm.py','panel_inference.py')}


def _code_receipt():
    return dict(difference._code_receipt(),**_extra_code())


def _mapping(d,w):
    mapping=iv._multiply([list(row) for row in zip(*d)],w)
    bread=iv._inverse_small(iv._multiply(mapping,d),failure='PANEL_INFERENCE_INFORMATION_SINGULAR')
    return iv._multiply(bread,mapping),bread


def _context(q):
    if type(q) is not dict: _fail('SCHEMA_INVALID')
    if q.get('schema')=='societybox-difference-gmm-v1': fit=difference.calculate(q)
    elif q.get('schema')=='societybox-system-gmm-v1': fit=system.calculate(q)
    else: _fail('SCHEMA_INVALID')
    rows=fit['transformed_rows']; x=[list(map(F,r['design_exact'])) for r in rows]
    z=[list(map(F,r['instruments_exact'])) for r in rows]
    y=[F(r.get('outcome_exact',r.get('delta_outcome_exact'))) for r in rows]
    u=[F(r['residual_exact']) for r in rows]; first=list(map(F,fit['first_step_coefficients_exact']))
    u1=[v-iv._dot(row,first) for row,v in zip(x,y)]
    weight=[list(map(F,row)) for row in fit['weight_matrix_exact']]
    d=iv._cross(z,x); a,bread=_mapping(d,weight)
    entities=sorted({r['entity'] for r in rows}); n,k,g=len(rows),len(x[0]),len(entities)
    factor=F(g,g-1)*F(n-1,n-k) if q['estimator']['covariance']=='CLUSTER_CR1' else F(1)
    return dict(q=q,fit=fit,rows=rows,x=x,y=y,z=z,u=u,u1=u1,d=d,weight=weight,a=a,bread=bread,
                entities=entities,factor=factor)


def _group_moments(ctx,errors):
    width=len(ctx['z'][0]); totals={entity:[F(0)]*width for entity in ctx['entities']}
    for row,z,u in zip(ctx['rows'],ctx['z'],errors):
        checkpoint(); total=totals[row['entity']]
        for j,value in enumerate(z): total[j]+=value*u
    return totals


def _finish(ctx,result,options,code):
    result.update(implementation_status='INDEPENDENT_IMPLEMENTED',native_execution='NOT_RUN',native_parity='NOT_RUN',
        ui_status='NOT_CONNECTED',identification='IDENTIFICATION_NOT_ESTABLISHED',
        source_authentication='NOT_ESTABLISHED',historical_as_of='NOT_ESTABLISHED',
        sample=ctx['fit']['sample'],parameter_order=ctx['fit']['parameter_order'],
        coefficients_exact=ctx['fit']['coefficients_exact'],estimator_steps=ctx['q']['estimator']['steps'],
        fit_input_sha256=ctx['fit']['input_sha256'],options=options)
    if ctx['q']['schema']=='societybox-system-gmm-v1': result['level_moment_validity_established']=False
    if _code_receipt()!=code: _fail('CODE_CHANGED_DURING_RUN')
    bound=dict(ctx['q'],panel_inference_options=dict(method=result['method'],**options))
    assure=system._assurance if ctx['q']['schema']=='societybox-system-gmm-v1' else difference._assurance
    assurance=assure(bound,result,extra_code=_extra_code())
    if assurance['code_sha256']!=code: _fail('CODE_CHANGED_DURING_RUN')
    result.update(assurance)
    return result


def _signed_z(numerator,variance):
    if numerator==0: return 0.0,'BINARY64_REFERENCE_STATISTIC'
    squared=numerator*numerator/variance
    try: number=float(squared)
    except OverflowError: number=math.inf
    if 0<number<math.inf: value=math.sqrt(number)
    else:
        logvalue=(math.log(squared.numerator)-math.log(squared.denominator))/2
        try: value=math.exp(logvalue)
        except OverflowError: return None,'BLOCKED_BINARY64_RANGE'
    if not math.isfinite(value): return None,'BLOCKED_BINARY64_RANGE'
    return (-value if numerator<0 else value),'BINARY64_REFERENCE_STATISTIC' if value else 'BINARY64_UNDERFLOW_EXACT_SQUARED_STATISTIC_AVAILABLE'


def _serial_components(ctx,order,basis,groups):
    rows,x,u=ctx['rows'],ctx['x'],ctx['u']; k=len(x[0])
    index={(r['entity'],r['period']):i for i,r in enumerate(rows) if r.get('equation','DIFFERENCE')=='DIFFERENCE'}
    numerator=F(0); qx=[F(0)]*k; products={entity:F(0) for entity in ctx['entities']}; pair_entities=set(); pairs=0
    for (entity,t),i in index.items():
        checkpoint(); lag=index.get((entity,t-order))
        if lag is None: continue
        pairs+=1; pair_entities.add(entity); lagged=u[lag]
        numerator+=u[i]*lagged; products[entity]+=basis[i]*lagged
        for j,value in enumerate(x[i]): qx[j]+=lagged*value
    c=[iv._dot(qx,list(column)) for column in zip(*ctx['a'])]
    projected={entity:iv._dot(c,g) for entity,g in groups.items()}
    direct=sum((v*v for v in products.values()),F(0))
    cross=-2*sum((products[e]*projected[e] for e in ctx['entities']),F(0))
    parameter=ctx['factor']*sum((v*v for v in projected.values()),F(0))
    variance=direct+cross+parameter
    gram=sum(((products[e]-projected[e])**2 for e in ctx['entities']),F(0))
    gram+=(ctx['factor']-1)*sum((v*v for v in projected.values()),F(0))
    if variance!=gram or variance<0: _fail('SERIAL_VARIANCE_IDENTITY_FAILED')
    return dict(numerator=numerator,variance=variance,terms=[direct,cross,parameter],pairs=pairs,pair_entities=len(pair_entities),
        qx=qx,entity_components=[dict(entity=e,residual_cross_product_exact=str(products[e]),
        parameter_projection_exact=str(projected[e]),adjusted_score_exact=str(products[e]-projected[e])) for e in ctx['entities']])


def arellano_bond(q,*,orders=None,covariance_residuals='FINAL_FIT'):
    """Robust AB zero AR(m) tests on first-differenced residuals.

    Numerator and lagged vector use the requested final one/two-step fit.
    Covariance plugin uses FINAL_FIT or FIRST_STEP residuals coherently in all
    variance/cross-covariance terms. Parameter covariance is the uncorrected
    robust plugin, with the declared CR1 factor only in its parameter term.
    This deliberately identifies the convention rather than claiming native
    software parity. Missing pairs use exact periods, never compressed rows.
    1..16 sorted unique orders in 1..1e6; source panel limits are unchanged.
    Large independent-N/fixed-T, consistent moments and null orthogonality of
    lagged design are assumptions. AR(1) rejection is expected under iid level
    innovations and does not itself imply misspecification. Nonrejection of
    higher orders does not prove moment validity. No Windmeijer-in-AR variant.
    """
    checkpoint(); owned=iv._snapshot(dict(study=q,orders=orders,covariance_residuals=covariance_residuals),limit=327680)
    selected=[1,2] if owned['orders'] is None else owned['orders']; basis_name=owned['covariance_residuals']
    if (type(selected) is not list or not 1<=len(selected)<=16 or
            any(type(v) is not int or not 1<=v<=1000000 for v in selected) or selected!=sorted(set(selected))): _fail('ORDERS_INVALID')
    if type(basis_name) is not str or basis_name not in ('FINAL_FIT','FIRST_STEP'): _fail('COVARIANCE_RESIDUALS_INVALID')
    code=_code_receipt(); ctx=_context(owned['study']); basis=ctx['u'] if basis_name=='FINAL_FIT' else ctx['u1']
    groups=_group_moments(ctx,basis); tests=[]
    for order in selected:
        checkpoint(); parts=_serial_components(ctx,order,basis,groups)
        record=dict(order=order,pairs=parts['pairs'],pair_entities=parts['pair_entities'],numerator_exact=str(parts['numerator']),
            variance_terms_exact=list(map(str,parts['terms'])),variance_exact=str(parts['variance']),
            lagged_design_projection_exact=list(map(str,parts['qx'])),entity_components=parts['entity_components'],
            p_value=None,log_p_value=None,z_statistic=None,chi_squared_statistic_exact=None)
        if not parts['pairs']: status='BLOCKED_NO_RESIDUAL_PAIRS'
        elif parts['pair_entities']<2: status='BLOCKED_INSUFFICIENT_PAIR_ENTITIES'
        elif not parts['variance']: status='BLOCKED_ZERO_ASYMPTOTIC_VARIANCE'
        else:
            statistic=parts['numerator']**2/parts['variance']; record.update(iv._chi_probability(statistic,1))
            z,zstatus=_signed_z(parts['numerator'],parts['variance'])
            record.update(chi_squared_statistic_exact=str(statistic),z_statistic=z,z_status=zstatus)
            status='COMPUTED_ASSUMPTIONS_UNVERIFIED' if record['p_value'] is not None else 'STATISTIC_COMPUTED_REFERENCE_PROBABILITY_BLOCKED'
        record['status']=status; record.setdefault('p_value_status',status); tests.append(record)
    computed=sum(t['chi_squared_statistic_exact'] is not None for t in tests)
    result=dict(schema='societybox-panel-ar-test-v1',method='ARELLANO_BOND_ROBUST_PLUGIN_M_TEST',
        status='COMPUTED_ASSUMPTIONS_UNVERIFIED' if computed==len(tests) else
               'PARTIALLY_COMPUTED_ASSUMPTIONS_UNVERIFIED' if computed else 'NO_CALIBRATED_TESTS',
        tests=tests,lagged_residual_scope='DIFFERENCE_EQUATIONS_ONLY_LEVEL_ENTRIES_ZERO',
        numerator_residuals='REQUESTED_FINAL_FIT',covariance_residuals=basis_name,
        covariance_basis_coefficients_exact=(ctx['fit']['coefficients_exact'] if basis_name=='FINAL_FIT' else
                                            ctx['fit']['first_step_coefficients_exact']),
        parameter_covariance='UNCORRECTED_GMM_PLUGIN_AT_DECLARED_RESIDUAL_BASIS',
        variance_convention='S1_MINUS_2_QX_A_ZHU_PLUS_QX_VBETA_QX_TRANSPOSE',
        arithmetic='EXACT_RATIONAL_STATISTIC_COMPONENTS',reference_distribution='CONDITIONAL_ASYMPTOTIC_STANDARD_NORMAL_TWO_SIDED',
        null_orthogonality='LAGGED_DESIGN_ORTHOGONAL_TO_CURRENT_DIFFERENCED_ERRORS_REQUIRED_NOT_VERIFIED',
        first_order_interpretation='NEGATIVE_AR1_IS_EXPECTED_AFTER_DIFFERENCING_IID_LEVEL_INNOVATIONS_NOT_AUTOMATIC_MISSPECIFICATION',
        original_software_convention_parity='NOT_RUN')
    if any(t['p_value'] is not None for t in tests): result['distribution_arithmetic']='BINARY64_NOT_INTERVAL_CERTIFIED'
    return _finish(ctx,result,dict(declared_orders=owned['orders'],orders=selected,covariance_residuals=basis_name),code)


def _windmeijer_components(ctx):
    x,z=ctx['x'],ctx['z']; k=len(x[0]); l=len(z[0]); groups=_group_moments(ctx,ctx['u1'])
    dgroup={entity:[[F(0)]*k for _ in range(l)] for entity in ctx['entities']}
    for row,xrow,zrow in zip(ctx['rows'],x,z):
        checkpoint(); d=dgroup[row['entity']]
        for i,value in enumerate(zrow):
            if value:
                for j,other in enumerate(xrow): d[i][j]+=value*other
    scores=list(groups.values()); s1=iv._cross(scores,scores)
    w1=iv._inverse_small([list(map(F,row)) for row in ctx['fit']['initial_moment_covariance_exact']])
    a1,_=_mapping(ctx['d'],w1); a2=ctx['a']; b2=ctx['bread']
    efficient=iv._multiply(iv._multiply(a2,s1),[list(row) for row in zip(*a2)])
    if efficient!=b2: _fail('WINDMEIJER_EFFICIENT_WEIGHT_IDENTITY_FAILED')
    v1=iv._multiply(iv._multiply(a1,s1),[list(row) for row in zip(*a1)])
    moments=[iv._dot(list(column),ctx['u']) for column in zip(*z)]
    v=[iv._dot(row,moments) for row in ctx['weight']]
    derivative_sum=[[F(0)]*k for _ in range(l)]
    # -S_j*v = sum_i[D_ij*(g1_i'v) + g1_i*(D_ij'v)]. No l*l*k tensor.
    for entity,g in groups.items():
        checkpoint(); d=dgroup[entity]; gv=iv._dot(g,v)
        dv=[iv._dot(list(column),v) for column in zip(*d)]
        for i in range(l):
            for j in range(k): derivative_sum[i][j]+=d[i][j]*gv+g[i]*dv[j]
    correction=iv._multiply(a2,derivative_sum)
    mapping=iv._multiply(correction,a1)
    mapping=[[v+w for v,w in zip(row,other)] for row,other in zip(a2,mapping)]
    adjusted=[[iv._dot(row,g) for row in mapping] for g in scores]
    variance=iv._cross(adjusted,adjusted)
    return variance,correction,v1,b2,adjusted


def windmeijer(q):
    """First-order Windmeijer two-step finite-sample covariance correction.

    Let S1=sum(g1_i*g1_i'), W2=S1^-1, B2=(D'W2D)^-1,
    A_j=(D'W_jD)^-1 D'W_j and F_j=-A2*(dS1/dbeta1_j)*W2*m2.
    V_WC=B2+F*B2+B2*F'+F*V1*F', V1=A1*S1*A1'. We evaluate the exactly
    equivalent Gram of (A2+F*A1)*g1_i to retain nonnegative variances.
    The declared CR1 scalar multiplies the whole correction. B2 is not the
    final-residual robust sandwich returned by the ordinary fit. Requires a
    two-step difference/system model; output remains an estimated variance,
    not a coverage, valid-moment or weak-identification certificate.
    """
    checkpoint(); owned=iv._snapshot(q,limit=262144)
    if type(owned) is not dict or type(owned.get('estimator')) is not dict: _fail('SCHEMA_INVALID')
    if owned['estimator'].get('steps')!=2: _fail('TWO_STEP_REQUIRED')
    code=_code_receipt(); ctx=_context(owned)
    variance,correction,v1,b2,adjusted=_windmeijer_components(ctx)
    result=dict(schema='societybox-windmeijer-result-v1',method='WINDMEIJER_TWO_STEP_PANEL_GMM_COVARIANCE',
        status='COMPUTED_ASSUMPTIONS_UNVERIFIED',variance_correction='WINDMEIJER_2005_FIRST_STEP_PLUGIN',
        arithmetic='EXACT_RATIONAL_FIRST_ORDER_VARIANCE_CORRECTION',residual_covariance_basis='FIRST_STEP_ENTITY_MOMENTS',
        covariance_exact=[list(map(str,[ctx['factor']*v for v in row])) for row in variance],
        windmeijer_covariance_unscaled_exact=[list(map(str,row)) for row in variance],
        first_step_robust_covariance_unscaled_exact=[list(map(str,row)) for row in v1],
        efficient_two_step_covariance_unscaled_exact=[list(map(str,row)) for row in b2],
        derivative_correction_exact=[list(map(str,row)) for row in correction],
        corrected_entity_scores_exact=[list(map(str,row)) for row in adjusted],
        entity_order=ctx['entities'],uncorrected_reported_covariance_exact=ctx['fit']['covariance_exact'],
        covariance_estimator=ctx['q']['estimator']['covariance'],covariance_scale_exact=str(ctx['factor']),
        efficient_weight_identity='VERIFIED_EXACT',small_sample_two_step_correction='WINDMEIJER_2005_FIRST_STEP_PLUGIN')
    return _finish(ctx,result,dict(variance_correction='WINDMEIJER_2005_FIRST_STEP_PLUGIN'),code)
