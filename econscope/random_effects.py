"""Exact static random-intercept GLS and explicit between-trace moment FGLS.

Model: y_it = x_it beta + a_i + e_it, Cov(u_i)=se I + su 11'. The
unbalanced moment estimator subtracts se*tr(M_between diag(1/T_i)) from
between SSR, not an interchangeable package default or harmonic-T formula.
No dense observation covariance or square-root quasi-demeaning is required.
Requires strict exogeneity, independent entities and the stated variance model;
exact arithmetic cannot validate those scientific assumptions.
"""
from fractions import Fraction as F
from pathlib import Path
import re

from . import instrumental_variables as iv
from .fixed_effects import _orthogonal
from .mechanism_dynamic import _dot
from .execution_control import checkpoint
from .workspace import canonical,digest
from .research_evidence import (make_evidence,EvidenceLedger,UnknownRegistry,
                                make_obligation,evaluate_coverage)

SCHEMA='societybox-random-effects-v1'
EXACT_BITS=12000


def _fail(reason): raise ValueError('RANDOM_EFFECTS_'+reason)


def _validated(q):
    fields={'schema','rows','model','units','period_unit','source_declared_sha256','rights',
            'variance_components','covariance'}
    if type(q) is not dict or set(q)!=fields or q['schema']!=SCHEMA: _fail('SCHEMA_INVALID')
    if q['rights']!=dict(local_processing='PERMITTED',report_export='PERMITTED'): _fail('RIGHTS_REQUIRED')
    if type(q['source_declared_sha256']) is not str or re.fullmatch('[a-f0-9]{64}',q['source_declared_sha256']) is None: _fail('SOURCE_INVALID')
    model=q['model']
    if type(model) is not dict or set(model)!={'outcome','regressors','include_intercept'}: _fail('MODEL_INVALID')
    labels=model['regressors'];intercept=model['include_intercept']
    if (type(labels) is not list or type(intercept) is not bool or not 1<=len(labels)+intercept<=8
            or any(not iv._text(v) for v in labels+[model['outcome']])
            or len(set(labels+[model['outcome'],'__intercept__']))!=len(labels)+2): _fail('MODEL_NAMES_INVALID')
    units=q['units']
    if (type(units) is not dict or set(units)!=set(labels+[model['outcome']])
            or any(not iv._text(v) for v in units.values()) or not iv._text(q['period_unit'])): _fail('UNITS_REQUIRED')
    if type(q['covariance']) is not str or q['covariance'] not in ('MODEL_BASED','ENTITY_CR0','ENTITY_CR1'): _fail('COVARIANCE_INVALID')
    var=q['variance_components']
    if type(var) is not dict or type(var.get('method')) is not str: _fail('VARIANCE_SPEC_INVALID')
    if var['method']=='DECLARED':
        if set(var)!={'method','idiosyncratic','entity'}: _fail('VARIANCE_SPEC_INVALID')
        se,su=_number(var['idiosyncratic']),_number(var['entity'])
        if se<=0 or su<0: _fail('VARIANCE_POSITIVITY_REQUIRED')
    elif var!={'method':'BETWEEN_TRACE_MOMENTS'}: _fail('VARIANCE_SPEC_INVALID')
    rows=q['rows']
    if type(rows) is not list or not 4<=len(rows)<=512 or len(rows)<=len(labels)+intercept: _fail('DIMENSION_LIMIT')
    x=[];y=[];groups={};pairs=set()
    for i,row in enumerate(rows):
        checkpoint()
        if type(row) is not dict or set(row)!={'entity','period','outcome','regressors'}: _fail('ROW_INVALID')
        if not iv._text(row['entity']) or type(row['period']) is not int or abs(row['period'])>10**9: _fail('PANEL_INDEX_INVALID')
        pair=row['entity'],row['period']
        if pair in pairs: _fail('DUPLICATE_ENTITY_PERIOD')
        pairs.add(pair);groups.setdefault(row['entity'],[]).append(i)
        if type(row['regressors']) is not dict or set(row['regressors'])!=set(labels): _fail('ROW_REGRESSORS_INVALID')
        x.append([F(1)]*intercept+[_number(row['regressors'][name]) for name in labels]);y.append(_number(row['outcome']))
    if not 2<=len(groups)<=128: _fail('ENTITY_DIMENSION_LIMIT')
    return x,y,{g:groups[g] for g in sorted(groups)}


def _number(v):
    if type(v) is not str: _fail('EXACT_NUMERIC_STRING_REQUIRED')
    return iv._num(v)


def _exact(v):
    # Keep Python's integer-string safety limit intact. Fractions never become
    # rounded/clipped merely to pass serialization or an evidence budget.
    if max(v.numerator.bit_length(),v.denominator.bit_length())>EXACT_BITS: _fail('EXACT_ARITHMETIC_BIT_LIMIT')
    return str(v)


def validate(q):
    """Validate an owned JSON snapshot only; does not fit or authorize execution."""
    owned=iv._snapshot(q,limit=262144);_validated(owned);return owned


def _projection(y,x):
    basis,norms=_orthogonal(list(map(list,zip(*x))),[F(1)]*len(y))
    ssr=_dot(y,y)-sum(_dot(y,b)**2/norm for b,norm in zip(basis,norms))
    h=[sum((b[i]**2/norm for b,norm in zip(basis,norms)),F(0)) for i in range(len(y))]
    return ssr,len(basis),h


def _moments(x,y,groups):
    k=len(x[0]);n=len(x);g=len(groups);wx=[None]*n;wy=[None]*n;bx=[];by=[];sizes=[]
    for positions in groups.values():
        checkpoint();t=len(positions);xm=[sum(x[i][j] for i in positions)/t for j in range(k)];ym=sum(y[i] for i in positions)/t
        bx.append(xm);by.append(ym);sizes.append(t)
        for i in positions:wx[i]=[v-m for v,m in zip(x[i],xm)];wy[i]=y[i]-ym
    sw,rw,_=_projection(wy,wx);sb,rb,hb=_projection(by,bx);dfw=n-g-rw;dfb=g-rb
    if dfw<=0: _fail('WITHIN_DF_REQUIRED')
    if dfb<=0: _fail('BETWEEN_DF_REQUIRED')
    se=sw/dfw
    if se<=0: _fail('POSITIVE_IDIOSYNCRATIC_VARIANCE_REQUIRED')
    trace=sum(((1-h)/t for h,t in zip(hb,sizes)),F(0));raw=(sb-se*trace)/dfb;su=max(F(0),raw)
    return se,su,dict(method='BETWEEN_TRACE_MOMENTS',idiosyncratic_exact=_exact(se),entity_exact=_exact(su),
        entity_untruncated_exact=_exact(raw),within_ssr_exact=_exact(sw),between_ssr_exact=_exact(sb),
        within_rank=rw,between_rank=rb,within_df=dfw,between_df=dfb,between_noise_trace_exact=_exact(trace),
        boundary='ENTITY_VARIANCE_TRUNCATED_AT_ZERO' if raw<0 else 'ENTITY_VARIANCE_ZERO' if raw==0 else 'INTERIOR',
        original_software_convention_parity='NOT_RUN')


def _precision_apply(columns,groups,se,su):
    """(se I+su 11')^-1 times each entity block, in O(n*p) exact work."""
    result=[None]*len(columns);p=len(columns[0])
    for positions in groups.values():
        checkpoint();rho=su/(se+len(positions)*su);sums=[sum(columns[i][j] for i in positions) for j in range(p)]
        for i in positions:result[i]=[(v-rho*s)/se for v,s in zip(columns[i],sums)]
    return result


def _fit(x,y,groups,se,su,covariance):
    n=len(y);k=len(x[0]);pyx=_precision_apply([[v]+row for v,row in zip(y,x)],groups,se,su)
    # Omega^-1 is symmetric, so X' Omega^-1 X is exactly symmetric.
    # Compute each expensive rational dot product just once.
    gram=[[F(0)]*k for _ in range(k)]
    for i in range(k):
        checkpoint()
        for j in range(i,k):
            value=sum(row[i]*p[j+1] for row,p in zip(x,pyx))
            gram[i][j]=gram[j][i]=value
    rhs=[sum(row[i]*p[0] for row,p in zip(x,pyx)) for i in range(k)]
    bread=iv._inverse_small(gram,failure='RANDOM_EFFECTS_DESIGN_RANK_DEFICIENT');beta=[_dot(row,rhs) for row in bread]
    residual=[v-_dot(row,beta) for row,v in zip(x,y)]
    pe=[p[0]-_dot(p[1:],beta) for p in pyx];scores=[];effects={};conditional=list(residual)
    for entity,positions in groups.items():
        checkpoint();scores.append([sum(x[i][j]*pe[i] for i in positions) for j in range(k)])
        effect=su/(se+len(positions)*su)*sum(residual[i] for i in positions);effects[entity]=_exact(effect)
        for i in positions:conditional[i]-=effect
    normal=[sum(score[j] for score in scores) for j in range(k)]
    if any(normal):_fail('NORMAL_EQUATION_IDENTITY_FAILED')
    factor=F(len(groups),len(groups)-1)*F(n-1,n-k) if covariance=='ENTITY_CR1' else F(1)
    if covariance=='MODEL_BASED':cov=bread
    else:
        # Score Gram avoids two dense bread/meat multiplies and retains PSD.
        adjusted=[[_dot(b,score) for b in bread] for score in scores]
        # A score Gram is symmetric: reuse the shared exact Gram routine.
        score_gram=iv._cross(adjusted,adjusted)
        cov=[[factor*v for v in row] for row in score_gram]
    return dict(coefficients_exact=list(map(_exact,beta)),covariance_exact=[list(map(_exact,row)) for row in cov],
        gls_bread_exact=[list(map(_exact,row)) for row in bread],covariance_scale_exact=_exact(factor),
        marginal_residuals_exact=list(map(_exact,residual)),conditional_residuals_exact=list(map(_exact,conditional)),
        gls_objective_exact=_exact(_dot(residual,pe)),normal_equation_residual_exact=list(map(_exact,normal)),
        entity_effects_exact=effects,entity_scores_exact=[list(map(_exact,row)) for row in scores],
        entity_effect_interpretation='MODEL_CONDITIONAL_LINEAR_PREDICTION_NOT_OBSERVED_EFFECT')


def _code_receipt():
    code=iv._code_receipt()
    for name in ('random_effects.py','fixed_effects.py'):code[name]=digest(Path(__file__).with_name(name).read_bytes())
    return code


def _assurance(q,result,code):
    subject=digest(canonical(q));scope=dict(model=q['model'],units=q['units'],period_unit=q['period_unit'],
        variance_components=q['variance_components'],covariance=q['covariance'],sample=result['sample'])
    # Full exact residual vectors can exceed the ledger's 2 MiB per-record cap.
    # Bind every calculation field without duplicating the numerical payload.
    fields=sorted(result);calculation_sha=digest(canonical(result))
    claims={
        'strict_exogeneity':'Are both entity effects and idiosyncratic errors orthogonal to the entire regressor history?',
        'variance_model':'Are random-intercept and homoskedastic idiosyncratic covariance restrictions appropriate?',
        'independent_entities':'Are entity clusters independent and numerous enough for robust asymptotics?',
        'finite_sample_inference':'Is plug-in variance and entity-effect prediction uncertainty independently assessed?',
        'source_vintage_replication':'Are source data, rights, historical availability and original API replication independently verified?'}
    evidence=make_evidence(subject_sha256=subject,producer=dict(tool_id='econ.random_effects',version='1',code_sha256=digest(canonical(code))),
        operation=result['method'],authority='LOCAL_EXACT_ARITHMETIC',epistemic_state='DERIVED',
        proof_kind='DECLARED_RANDOM_INTERCEPT_GLS',scope=scope,coverage=dict(status='PARTIAL',unknown_dimensions=list(claims)),
        result=dict(calculation_sha256=calculation_sha,calculation_fields=fields,
                    status=result['status'],normal_equations=result['normal_equation_residual_exact']),parent_evidence_ids=[],limitations=[
            'Declared variances are not authenticated; feasible covariance is a plug-in estimate, not calibrated coverage.',
            'Between-trace moments are an explicit estimator, not a claim of any original package default parity.',
            'Robust covariance cannot repair correlated entity effects or endogenous regressors.',
            'Entity effects are linear predictions under the model, not observed effects or causal estimates.'])
    ledger=EvidenceLedger([evidence]);registry=UnknownRegistry(ledger);obligations=[]
    for claim,question in claims.items():
        u=registry.register(question,scope,'EXTERNAL_SCIENTIFIC_VALIDATION','RANDOM_EFFECTS_ASSUMPTION_CHECK')
        obligations.append(make_obligation(claim_id=claim,subject_sha256=subject,scope=scope,
            required_authority='EXTERNAL_SCIENTIFIC_VALIDATION',required_proof_kind='RANDOM_EFFECTS_ASSUMPTION_CHECK',
            verifier_id='econ.random_effects',dependencies=[],unknown_ids=[u['unknown_id']],blocking_reasons=[],next_probe=question))
    return dict(input_sha256=subject,code_sha256=code,calculation_sha256=calculation_sha,calculation_fields=fields,
                evidence=ledger.records(),unknowns=registry.records(),
                obligations=obligations,coverage=evaluate_coverage(obligations,ledger,registry))


def calculate(q):
    """Fit static RE GLS with declared variances or BETWEEN_TRACE_MOMENTS.

    Input: schema societybox-random-effects-v1, rows of entity/period/outcome/
    regressors, model outcome/regressors/include_intercept, units/period_unit,
    source_declared_sha256, permitted rights, variance_components and covariance.
    Data/variance numbers are exact strings. Covariance is MODEL_BASED,
    ENTITY_CR0 or ENTITY_CR1. Limits: rows512, entities128, parameters8,
    numeric output numerator/denominator <=12000 bits (fail, never clip).
    Missing periods are retained; this exchangeable covariance has no lag model.
    """
    checkpoint();owned=iv._snapshot(q,limit=262144);x,y,groups=_validated(owned);code=_code_receipt()
    var=owned['variance_components']
    if var['method']=='DECLARED':
        se,su=_number(var['idiosyncratic']),_number(var['entity'])
        components=dict(method='DECLARED',idiosyncratic_exact=_exact(se),entity_exact=_exact(su),
                        boundary='ENTITY_VARIANCE_ZERO' if su==0 else 'INTERIOR',variance_authentication='NOT_ESTABLISHED')
    else:se,su,components=_moments(x,y,groups)
    result=_fit(x,y,groups,se,su,owned['covariance']);n=len(y);k=len(x[0])
    result.update(schema='societybox-random-effects-result-v1',method='RANDOM_INTERCEPT_GLS',
        implementation_status='INDEPENDENT_IMPLEMENTED',status='COMPUTED_ASSUMPTIONS_UNVERIFIED',
        arithmetic='EXACT_RATIONAL_NO_SQRT_TRANSFORMATION',variance_components=components,
        parameter_order=(['__intercept__'] if owned['model']['include_intercept'] else [])+owned['model']['regressors'],
        entity_order=list(groups),sample=dict(observations=n,entities=len(groups),parameters=k,residual_df=n-k,
            entity_observations={g:len(p) for g,p in groups.items()}),covariance_estimator=owned['covariance'],
        covariance_interpretation='DECLARED_MODEL_COVARIANCE' if var['method']=='DECLARED' and owned['covariance']=='MODEL_BASED'
            else 'ESTIMATED_PLUGIN_NOT_FINITE_SAMPLE_COVERAGE',native_execution='NOT_RUN',native_parity='NOT_RUN',
        causal_identification='NOT_ESTABLISHED',source_authentication='NOT_ESTABLISHED',historical_as_of='NOT_ESTABLISHED',
        exogeneity='DECLARED_NOT_VERIFIED',p_values=None,confidence_intervals=None)
    result.update(_assurance(owned,result,code))
    if code!=_code_receipt():_fail('CODE_CHANGED')
    return result
