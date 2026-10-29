"""Known-parameter, time-varying linear Gaussian state-space reference core.

x[t] = F[t] x[t-1] + c[t] + w[t]; y[t] = H[t] x[t] + d[t] + v[t].
The declared initial prior precedes the first transition. Independent Gaussian
noises, partial missing observations, and positive semidefinite covariances are
explicit. Exact LDL conditioning handles consistent singular innovations; it
never invents a full-dimensional density for a degenerate Gaussian. Joseph
filter updates and generalized RTS smoothing use bounded rational arithmetic.
Only log-density transcendental terms use a fixed Decimal approximation.
"""
from decimal import Decimal,Context,localcontext,ROUND_HALF_EVEN
from fractions import Fraction as F
from pathlib import Path
import json
import math
import re

from . import instrumental_variables as iv
from .execution_control import checkpoint
from .research_evidence import EvidenceLedger,UnknownRegistry,make_evidence,make_obligation,evaluate_coverage
from .workspace import canonical,digest

_CONTEXT=Context(prec=80,rounding=ROUND_HALF_EVEN)
_PI='3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679'
_NOISE='INDEPENDENT_ACROSS_TIME_INITIAL_STATE_AND_MEASUREMENT'
_BLOCK_KEYS={'STATE_DYNAMICS':('transition','state_intercept','process_covariance'),
             'OBSERVATION_DYNAMICS':('measurement','observation_intercept','measurement_covariance')}


def _fail(reason):raise ValueError('STATE_SPACE_'+reason)


def _text(value):return iv._text(value) and bool(value.strip())


def _bounded(values):
    for v in values:
        if isinstance(v,list):_bounded(v)
        elif max(v.numerator.bit_length(),v.denominator.bit_length())>8192:_fail('EXACT_ARITHMETIC_RESOURCE_LIMIT')
    return values


def _identity(n):return [[F(i==j) for j in range(n)] for i in range(n)]


def _transpose(a):return [list(v) for v in zip(*a)]


def _add(a,b,sign=1):return _bounded([[v+sign*w for v,w in zip(row,other)] for row,other in zip(a,b)])


def _multiply(a,b):return _bounded(iv._multiply(a,b))


def _mv(a,b):return _bounded([iv._dot(row,b) for row in a])


def _sandwich(a,b):return _multiply(_multiply(a,b),_transpose(a))


def _factor(matrix):
    """Exact semidefinite LDL; a zero diagonal requires a zero Schur column."""
    n=len(matrix);lower=_identity(n);diagonal=[]
    if any(matrix[i][j]!=matrix[j][i] for i in range(n) for j in range(n)):_fail('COVARIANCE_SYMMETRY_REQUIRED')
    for j in range(n):
        checkpoint()
        value=matrix[j][j]-sum((lower[j][k]**2*diagonal[k] for k in range(j)),F(0))
        _bounded([value])
        if value<0:_fail('COVARIANCE_NOT_POSITIVE_SEMIDEFINITE')
        diagonal.append(value)
        for i in range(j+1,n):
            remainder=matrix[i][j]-sum((lower[i][k]*lower[j][k]*diagonal[k] for k in range(j)),F(0))
            if value==0:
                if remainder!=0:_fail('COVARIANCE_NOT_POSITIVE_SEMIDEFINITE')
            else:lower[i][j]=remainder/value
        _bounded(lower)
    return lower,diagonal


def _solve_factor(factor,rhs,reason):
    lower,diagonal=factor;n=len(diagonal);forward=[]
    for i in range(n):forward.append(rhs[i]-sum((lower[i][j]*forward[j] for j in range(i)),F(0)))
    _bounded(forward)
    if any(d==0 and v!=0 for v,d in zip(forward,diagonal)):_fail(reason)
    scaled=[v/d if d else F(0) for v,d in zip(forward,diagonal)];result=[F(0)]*n
    for i in range(n-1,-1,-1):result[i]=scaled[i]-sum((lower[j][i]*result[j] for j in range(i+1,n)),F(0))
    return _bounded(result)


def _number(v):
    # EM candidates remain re-executable without rounding to IV's 64-bit input
    # limit. Declarations and calculations share the 8192-bit rational budget.
    if type(v) not in (int,float,str):_fail('NUMBER_INVALID')
    if type(v) is str and (len(v)>5000 or not re.fullmatch(r'-?\d+(?:\.\d+|/\d+)?',v,re.ASCII)):_fail('NUMBER_FORMAT_INVALID')
    if type(v) is float and not math.isfinite(v):_fail('NUMBER_INVALID')
    if type(v) is int and v.bit_length()>8192:_fail('NUMBER_INPUT_RESOURCE_LIMIT')
    try:number=F(str(v))
    except (ValueError,ZeroDivisionError):_fail('NUMBER_INVALID')
    if abs(number)>1000000:_fail('NUMBER_MAGNITUDE_LIMIT')
    return _bounded([number])[0]


def _vector(v,n):
    if type(v) is not list or len(v)!=n:_fail('VECTOR_DIMENSIONS_INVALID')
    return list(map(_number,v))


def _matrix(v,n,m):
    if type(v) is not list or len(v)!=n or any(type(row) is not list or len(row)!=m for row in v):_fail('MATRIX_DIMENSIONS_INVALID')
    return [list(map(_number,row)) for row in v]


def _covariance(v,n):
    result=_matrix(v,n,n);_factor(result);return result


def _validated(q):
    keys={'schema','states','observables','period_unit','initial','conditioning_end_period','distribution','noise_assumptions','steps','units','source_declared_sha256','rights'}
    if type(q) is not dict or set(q)!=keys or q['schema']!='societybox-linear-gaussian-state-space-v1':_fail('SCHEMA_INVALID')
    if q['rights']!=dict(local_processing='PERMITTED',report_export='PERMITTED'):_fail('RIGHTS_REQUIRED')
    if type(q['source_declared_sha256']) is not str or not re.fullmatch('[a-f0-9]{64}',q['source_declared_sha256']):_fail('SOURCE_INVALID')
    if q['distribution']!='GAUSSIAN' or q['noise_assumptions']!=_NOISE:_fail('DISTRIBUTION_OR_NOISE_ASSUMPTIONS_UNSUPPORTED')
    for key,limit in (('states',6),('observables',4)):
        v=q[key]
        if type(v) is not list or not 1<=len(v)<=limit or any(not _text(x) for x in v) or len(set(v))!=len(v):_fail('NAMES_INVALID')
    if not _text(q['period_unit']):_fail('PERIOD_UNIT_REQUIRED')
    units=q['units']
    if type(units) is not dict or set(units)!={'states','observables'}:_fail('UNITS_REQUIRED')
    for key in ('states','observables'):
        if type(units[key]) is not dict or set(units[key])!=set(q[key]) or any(not _text(v) for v in units[key].values()):_fail('UNITS_REQUIRED')
    p,m=len(q['states']),len(q['observables']);initial=q['initial']
    if (type(initial) is not dict or set(initial)!={'period','mean','covariance'} or type(initial['period']) is not int
            or not 0<=initial['period']<=1000000):_fail('INITIAL_PRIOR_INVALID')
    mean=_vector(initial['mean'],p);covariance=_covariance(initial['covariance'],p);steps=q['steps']
    if type(steps) is not list or not 1<=len(steps)<=128:_fail('STEP_LIMIT')
    parsed=[]
    fields={'period','transition','state_intercept','process_covariance','measurement','observation_intercept','measurement_covariance','values'}
    for i,s in enumerate(steps,1):
        if type(s) is not dict or set(s)!=fields:_fail('STEP_FIELDS_INVALID')
        if type(s['period']) is not int or s['period']!=initial['period']+i or s['period']>1000000:_fail('CONTIGUOUS_PERIODS_REQUIRED')
        if type(s['values']) is not dict or set(s['values'])!=set(q['observables']):_fail('OBSERVABLE_VALUES_REQUIRED')
        parsed.append(dict(period=s['period'],transition=_matrix(s['transition'],p,p),state_intercept=_vector(s['state_intercept'],p),
            process_covariance=_covariance(s['process_covariance'],p),measurement=_matrix(s['measurement'],m,p),
            observation_intercept=_vector(s['observation_intercept'],m),measurement_covariance=_covariance(s['measurement_covariance'],m),
            values=[None if s['values'][name] is None else _number(s['values'][name]) for name in q['observables']]))
    end=q['conditioning_end_period']
    if type(end) is not int or not initial['period']<end<=parsed[-1]['period']:_fail('CONDITIONING_CUTOFF_INVALID')
    if len(canonical(q))>524288:_fail('INPUT_SIZE_LIMIT')
    return mean,covariance,parsed


def validate(q):
    """Validate declarations and PSD noise matrices, without filtering."""
    _validated(q);return q


def _snapshot(q):
    if type(q) is not dict:_fail('SCHEMA_INVALID')
    try:raw=canonical(q)
    except (TypeError,ValueError,OverflowError,RecursionError):_fail('JSON_INPUT_REQUIRED')
    if len(raw)>524288:_fail('INPUT_SIZE_LIMIT')
    return json.loads(raw)


def _strings(matrix):return [list(map(str,row)) for row in matrix]


def _cell(period,mean,covariance,mode,end,count,future=False):
    return dict(period=period,mean=list(map(str,mean)),covariance=_strings(covariance),information_mode=mode,
                information_end_period=end,conditioned_measurement_count=count,uses_future_measurements=future)


def _decimal(v):return Decimal(v.numerator)/Decimal(v.denominator)


def _log_density(diagonal,quadratic):
    with localcontext(_CONTEXT):
        value=-(len(diagonal)*(2*Decimal(_PI)).ln()+sum((_decimal(v).ln() for v in diagonal),Decimal(0))+_decimal(quadratic))/2
        return str(+value)


def _sum_logs(values):
    if any(v is None for v in values):return None
    with localcontext(_CONTEXT):return str(+sum(map(Decimal,values),Decimal(0)))


def _predict(mean,covariance,step):
    mean=_bounded([a+b for a,b in zip(_mv(step['transition'],mean),step['state_intercept'])])
    covariance=_add(_sandwich(step['transition'],covariance),step['process_covariance']);_factor(covariance)
    return mean,covariance


def _run_filter(q):
    mean,covariance,steps=_validated(q);initial=(list(mean),[list(row) for row in covariance]);p=len(mean)
    priors=[];filtered=[];innovations=[];pattern=[];states=[initial];predicted=[];count=0
    for step in steps:
        period=step['period']
        if period>q['conditioning_end_period']:break
        checkpoint();mean,covariance=_predict(mean,covariance,step);predicted.append((mean,covariance))
        priors.append(_cell(period,mean,covariance,'PRIOR',period-1,count))
        indices=[j for j,v in enumerate(step['values']) if v is not None]
        names=[q['observables'][j] for j in indices];pattern.append(dict(period=period,observables=names));observed=len(indices)
        if observed:
            h=[step['measurement'][j] for j in indices];r=[[step['measurement_covariance'][i][j] for j in indices] for i in indices]
            error=_bounded([step['values'][j]-step['observation_intercept'][j]-iv._dot(step['measurement'][j],mean) for j in indices])
            cross=_multiply(covariance,_transpose(h));innovation=_add(_multiply(h,cross),r);factor=_factor(innovation)
            reason='OBSERVATION_OUTSIDE_DEGENERATE_GAUSSIAN_SUPPORT:'+str(period)
            alpha=_solve_factor(factor,error,reason)
            gain=[_solve_factor(factor,row,'INTERNAL_CONDITIONAL_CROSS_COVARIANCE_RANGE') for row in cross]
            quadratic=iv._dot(error,alpha);_bounded([quadratic]);rank=sum(v>0 for v in factor[1])
            if quadratic<0:_fail('INTERNAL_NEGATIVE_MAHALANOBIS')
            # Exact Joseph form; gain is any consistent symmetric generalized
            # solution, so singular observations give the same conditional law.
            mean=_bounded([v+iv._dot(row,error) for v,row in zip(mean,gain)])
            projection=_add(_identity(p),_multiply(gain,h),-1)
            covariance=_add(_sandwich(projection,covariance),_sandwich(gain,r));_factor(covariance)
            density=_log_density(factor[1],quadratic) if rank==observed else None
            determinant=F(1)
            for v in factor[1]:determinant*=v
            _bounded([determinant])
        else:
            error=[];innovation=[];gain=[[] for _ in range(p)];quadratic=F(0);rank=0;density='0';determinant=F(1)
        count+=observed;states.append((mean,covariance))
        filtered.append(_cell(period,mean,covariance,'FILTERED',period,count))
        innovations.append(dict(period=period,observables=names,residual=list(map(str,error)),covariance=_strings(innovation),
            gain=_strings(gain),rank=rank,dimension=observed,determinant_exact=str(determinant),mahalanobis_exact=str(quadratic),
            log_density=density,likelihood_status='NO_OBSERVATION' if not observed else
            'FULL_DIMENSIONAL_GAUSSIAN_DENSITY' if rank==observed else 'SINGULAR_INNOVATION_NO_FULL_DIMENSIONAL_DENSITY'))
    likelihood=_sum_logs([v['log_density'] for v in innovations])
    result=dict(schema='societybox-state-space-result-v1',method='EXACT_JOSEPH_KALMAN_FILTER',status='CALCULATED',
        implementation_status='INDEPENDENT_IMPLEMENTED',native_execution='NOT_RUN',native_parity='NOT_RUN',ui_status='NOT_CONNECTED',
        state_order=list(q['states']),observable_order=list(q['observables']),
        initial_prior=_cell(q['initial']['period'],*initial,'INITIAL_PRIOR',q['initial']['period'],0),
        priors=priors,filtered=filtered,innovations=innovations,measurement_pattern=pattern,measurement_count=count,
        conditioning_end_period=q['conditioning_end_period'],log_likelihood=likelihood,
        likelihood_status='FULL_DIMENSIONAL_INNOVATION_FACTORIZATION' if likelihood is not None else 'SINGULAR_INNOVATION_NO_FULL_DIMENSIONAL_DENSITY',
        moment_arithmetic='EXACT_RATIONAL',log_likelihood_arithmetic='DECIMAL_80_HALF_EVEN_NOT_INTERVAL_CERTIFIED',
        uncertainty_scope='LATENT_STATE_AND_OBSERVATION_CONDITIONAL_ON_FIXED_DECLARED_PARAMETERS',
        parameter_estimation='NOT_PERFORMED_FIXED_DECLARED_PARAMETERS',parameter_uncertainty='NOT_ASSESSED',historical_information='NOT_ESTABLISHED',
        structural_causal_identification='NOT_ESTABLISHED')
    return result,steps,states,predicted


def _code_receipt():
    code=iv._code_receipt();code['state_space.py']=digest(Path(__file__).read_bytes());return code


def _assurance(q,result,expected_code=None):
    code=_code_receipt()
    if expected_code is not None and code!=expected_code:_fail('CODE_CHANGED_DURING_CALCULATION')
    subject=digest(canonical(q));scope=dict(states=q['states'],observables=q['observables'],units=q['units'],period_unit=q['period_unit'],
        initial_period=q['initial']['period'],conditioning_end_period=q['conditioning_end_period'],distribution=q['distribution'],noise_assumptions=q['noise_assumptions'])
    evidence=make_evidence(subject_sha256=subject,producer=dict(tool_id='econ.state_space',version='1',code_sha256=digest(canonical(code))),
        operation=result['method'],authority='LOCAL_EXACT_RATIONAL_GAUSSIAN_CONDITIONING',epistemic_state='DERIVED',
        proof_kind='DECLARED_STATE_SPACE_RESULT',scope=scope,result=result,parent_evidence_ids=[],
        coverage=dict(status='PARTIAL',unknown_dimensions=['noise_model','parameters','historical_information','causal_identification','independent_replication']),
        limitations=['Conditional moments and EM updates use declared Gaussian assumptions; estimation and identification require separate validation.',
                     'Smoothing is retrospective; filter/forecast cutoffs are mathematical periods, not historical-access proofs.',
                     'A singular innovation has no reported full-dimensional density; diffuse initialization and automatic model selection are unsupported.'])
    ledger=EvidenceLedger([evidence]);registry=UnknownRegistry(ledger);unknowns=[];obligations=[]
    for claim,question in (
        ('noise_model','Are linearity, Gaussian noise, covariance matrices, ignorable missingness and cross-time/state-measurement independence justified?'),
        ('parameters','Are initial distribution, dynamics, loadings, units and known-parameter assumptions independently justified?'),
        ('historical_information','Were observations and declared future dynamics available at the relevant origin with valid source rights?'),
        ('causal_identification','What independent design identifies a causal interpretation of these latent states or forecasts?'),
        ('independent_replication','Do independent calculations, source provenance and real-data diagnostics validate this model?')):
        unknown=registry.register(question,scope,'EXTERNAL_SCIENTIFIC_VALIDATION','STATE_SPACE_DESIGN_CHECK');unknowns.append(unknown)
        obligations.append(make_obligation(claim_id=claim,subject_sha256=subject,scope=scope,required_authority='EXTERNAL_SCIENTIFIC_VALIDATION',
            required_proof_kind='STATE_SPACE_DESIGN_CHECK',verifier_id='econ.state_space',dependencies=[],unknown_ids=[unknown['unknown_id']],blocking_reasons=[],next_probe=question))
    return dict(input_sha256=subject,code_sha256=code,evidence=[evidence],unknowns=unknowns,obligations=obligations,coverage=evaluate_coverage(obligations,ledger,registry))


def filter(q):
    q=_snapshot(q);result,*_=_run_filter(q);result.update(_assurance(q,result));return result


def _run_smoother(q,forward=None):
    if forward is not None and forward[0]!=digest(canonical(q)):_fail('INTERNAL_FORWARD_CACHE_INPUT_MISMATCH')
    result,steps,states,predicted=_run_filter(q) if forward is None else forward[1]
    smoothed=list(states);lag=[None]*(len(states)-1)
    for i in range(len(states)-2,-1,-1):
        checkpoint();mean,covariance=states[i];prior_mean,prior_covariance=predicted[i]
        cross=_multiply(covariance,_transpose(steps[i]['transition']));factor=_factor(prior_covariance)
        gain=[_solve_factor(factor,row,'INTERNAL_SMOOTHING_CROSS_COVARIANCE_RANGE') for row in cross]
        mean=_bounded([v+iv._dot(row,[a-b for a,b in zip(smoothed[i+1][0],prior_mean)]) for v,row in zip(mean,gain)])
        covariance=_add(covariance,_sandwich(gain,_add(smoothed[i+1][1],prior_covariance,-1)));_factor(covariance)
        smoothed[i]=(mean,covariance);lag[i]=_multiply(gain,smoothed[i+1][1])
    end=q['conditioning_end_period'];count=result['measurement_count'];last_observed=max((v['period'] for v in result['measurement_pattern'] if v['observables']),default=q['initial']['period'])
    cells=[_cell(q['initial']['period']+i,*state,'SMOOTHED',end,count,last_observed>q['initial']['period']+i) for i,state in enumerate(smoothed)]
    result.update(method='EXACT_JOSEPH_KALMAN_AND_GENERALIZED_RTS',smoothed_initial=cells[0],smoothed=cells[1:],
        lag_one_covariances=[dict(period=q['initial']['period']+i,next_period=q['initial']['period']+i+1,
            orientation='ROW_EARLIER_STATE_COLUMN_NEXT_STATE',covariance=_strings(value)) for i,value in enumerate(lag)])
    return result,steps,smoothed,lag


def smooth(q):
    q=_snapshot(q);result,*_=_run_smoother(q);result.update(_assurance(q,result));return result


def _outer(a,b):return _bounded([[x*y for y in b] for x in a])


def _augmented_second(mean,covariance):
    second=_add(covariance,_outer(mean,mean))
    return [row+[mean[i]] for i,row in enumerate(second)]+[list(mean)+[F(1)]]


def _em_regression(zz,yz,yy,count,base=None,fixed_columns=(),fixed_noise=None):
    free=[j for j in range(len(zz)) if j not in fixed_columns]
    if not fixed_columns:
        factor=_factor(zz)
        if any(v==0 for v in factor[1]):_fail('EM_UNIDENTIFIED_SECOND_MOMENTS')
        coefficients=[_solve_factor(factor,row,'INTERNAL_EM_NORMAL_EQUATION_RANGE') for row in yz]
    else:
        coefficients=[list(row) for row in base]
        if free:
            factor=_factor([[zz[i][j] for j in free] for i in free])
            if any(v==0 for v in factor[1]):_fail('EM_UNIDENTIFIED_SECOND_MOMENTS')
            for i,row in enumerate(yz):
                rhs=_bounded([row[j]-sum((base[i][k]*zz[k][j] for k in fixed_columns),F(0)) for j in free])
                solved=_solve_factor(factor,rhs,'INTERNAL_EM_CONDITIONAL_NORMAL_EQUATION_RANGE')
                for j,value in zip(free,solved):coefficients[i][j]=value
    residual=_add(yz,_multiply(coefficients,zz),-1)
    if any(row[j]!=0 for row in residual for j in free):_fail('INTERNAL_EM_NORMAL_EQUATION_RESIDUAL')
    if fixed_noise is not None:noise=[list(row) for row in fixed_noise]
    else:
        residual_second=_add(yy,_multiply(coefficients,_transpose(yz)),-1)
        if fixed_columns:
            # Fixed coefficient columns need the full E[(y-Bz)(y-Bz)'].
            # The unconstrained normal-equation shortcut is invalid here.
            residual_second=_add(_add(residual_second,_multiply(yz,_transpose(coefficients)),-1),_sandwich(coefficients,zz))
        noise=_bounded([[v/count for v in row] for row in residual_second])
    rank=sum(v>0 for v in _factor(noise)[1])
    return coefficients,noise,dict(second_predictor_sum=_strings(zz),target_predictor_sum=_strings(yz),
        second_target_sum=_strings(yy),count=count,normal_equations_exact=True,noise_rank=rank,
        normal_equations_scope='FREE_COEFFICIENT_COLUMNS_ONLY',free_coefficient_columns=free,
        fixed_coefficient_columns=list(fixed_columns),noise_estimation='FIXED' if fixed_noise is not None else 'ML_DIVISOR_N')


def _em_controls(blocks,fixed_parameters):
    if (type(blocks) is not list or not blocks or any(type(v) is not str or v not in _BLOCK_KEYS for v in blocks)
            or len(set(blocks))!=len(blocks)):_fail('EM_BLOCKS_INVALID')
    selected=[key for block in blocks for key in _BLOCK_KEYS[block]]
    if fixed_parameters is None:fixed_parameters=[]
    if (type(fixed_parameters) is not list or any(type(v) is not str or v not in selected for v in fixed_parameters)
            or len(set(fixed_parameters))!=len(fixed_parameters)):_fail('EM_FIXED_PARAMETERS_INVALID')
    if all(key in fixed_parameters for key in selected):_fail('EM_NO_FREE_PARAMETERS')
    return list(blocks),list(fixed_parameters)


def _em_free_groups(q,blocks,fixed_parameters,fixed_entries):
    covered={(v['field'],v['row'],v['column']) for v in fixed_entries};p=len(q['states']);m=len(q['observables']);free=[]
    for block in blocks:
        matrix,intercept,noise=_BLOCK_KEYS[block];height=p if block=='STATE_DYNAMICS' else m
        for field in (matrix,intercept,noise):
            if field in fixed_parameters:continue
            coordinates=[(i,j) for i in range(height) for j in range(p)] if field==matrix else [(i,None) for i in range(height)]
            if field==noise or any((field,i,j) not in covered for i,j in coordinates):free.append(field)
    return free


def _em_entry_controls(q,blocks,fixed_parameters,entries):
    if entries is None:return []
    if type(entries) is not list or len(entries)>70:_fail('EM_FIXED_ENTRIES_INVALID')
    try:raw=canonical(entries)
    except (TypeError,ValueError,OverflowError,RecursionError):_fail('EM_FIXED_ENTRIES_INVALID')
    if len(raw)>8192:_fail('EM_FIXED_ENTRIES_RESOURCE_LIMIT')
    entries=json.loads(raw);selected={key for block in blocks for key in _BLOCK_KEYS[block][:2]};seen=set()
    if type(q.get('states')) is not list or type(q.get('observables')) is not list:_fail('SCHEMA_INVALID')
    p,m=len(q['states']),len(q['observables'])
    if not 1<=p<=6 or not 1<=m<=4:_fail('NAMES_INVALID')
    for spec in entries:
        if type(spec) is not dict or set(spec)!={'field','row','column'}:_fail('EM_FIXED_ENTRIES_INVALID')
        field,row,col=spec['field'],spec['row'],spec['column']
        if type(field) is not str or field not in selected or field in fixed_parameters:_fail('EM_FIXED_ENTRIES_FIELD_UNSUPPORTED_OR_GROUP_FIXED')
        height=p if field in ('transition','state_intercept') else m
        if type(row) is not int or not 0<=row<height:_fail('EM_FIXED_ENTRIES_INDEX_INVALID')
        if field in ('state_intercept','observation_intercept'):
            if col is not None:_fail('EM_FIXED_ENTRIES_VECTOR_COLUMN_MUST_BE_NULL')
        elif type(col) is not int or not 0<=col<p:_fail('EM_FIXED_ENTRIES_INDEX_INVALID')
        key=(field,row,col)
        if key in seen:_fail('EM_FIXED_ENTRIES_DUPLICATE')
        seen.add(key)
    if entries and not _em_free_groups(q,blocks,fixed_parameters,entries):_fail('EM_NO_FREE_PARAMETERS')
    return entries


def _restore_em_entries(target,source,entries):
    for dst,src in zip(target['steps'],source['steps']):
        for spec in entries:
            field,row,col=spec['field'],spec['row'],spec['column']
            if col is None:dst[field][row]=src[field][row]
            else:dst[field][row][col]=src[field][row][col]
    return target


def _em_masked_regression(zz,yz,yy,count,base,fixed,old_noise,noise_fixed):
    """One ECM coefficient maximization at old noise, then optional noise ML.

    Different row restrictions require precision-weighted coupled equations.
    Build only the free-coordinate block of precision (Kronecker) Szz; never
    solve independent row equations unless the restriction pattern permits it.
    """
    height,width=len(base),len(zz);free=[(i,j) for i in range(height) for j in range(width) if (i,j) not in fixed]
    fixed_columns=[j for j in range(width) if all((i,j) in fixed for i in range(height))]
    if len(fixed)==height*len(fixed_columns):
        # Uniform row restrictions have identical free predictors: precision
        # cancels from the free normal equations. Reuse one small Szz factor.
        coefficients,noise,_=_em_regression(zz,yz,yy,count,base,fixed_columns,old_noise if noise_fixed else None)
    else:
        coefficients=[list(row) for row in base];noise_factor=_factor(old_noise)
        precision=_transpose([_solve_factor(noise_factor,[F(i==j) for i in range(height)],'INTERNAL_EM_PRECISION_RANGE') for j in range(height)])
        if free:
            normal=_bounded([[precision[i][k]*zz[j][l] for k,l in free] for i,j in free])
            adjusted=_bounded([[yz[i][j]-sum((base[i][l]*zz[l][j] for l in range(width) if (i,l) in fixed),F(0)) for j in range(width)] for i in range(height)])
            weighted=_multiply(precision,adjusted);rhs=[weighted[i][j] for i,j in free]
            factor=_factor(normal)
            if any(v==0 for v in factor[1]):_fail('EM_UNIDENTIFIED_SECOND_MOMENTS')
            solved=_solve_factor(factor,rhs,'INTERNAL_EM_RESTRICTED_NORMAL_EQUATION_RANGE')
            for (i,j),value in zip(free,solved):coefficients[i][j]=value
        gradient=_multiply(precision,_add(yz,_multiply(coefficients,zz),-1))
        if any(gradient[i][j] for i,j in free):_fail('INTERNAL_EM_RESTRICTED_NORMAL_EQUATION_RESIDUAL')
        if noise_fixed:noise=[list(row) for row in old_noise]
        else:
            residual=_add(_add(_add(yy,_multiply(coefficients,_transpose(yz)),-1),_multiply(yz,_transpose(coefficients)),-1),_sandwich(coefficients,zz))
            noise=_bounded([[v/count for v in row] for row in residual])
    rank=sum(v>0 for v in _factor(noise)[1])
    proof=dict(second_predictor_sum=_strings(zz),target_predictor_sum=_strings(yz),second_target_sum=_strings(yy),count=count,
        normal_equations_exact=True,noise_rank=rank,normal_equations_scope='FREE_COEFFICIENT_COORDINATES_PRECISION_WEIGHTED',
        free_coefficient_coordinates=[list(v) for v in free],fixed_coefficient_coordinates=[list(v) for v in sorted(fixed)],
        coefficient_maximization='CONDITIONAL_AT_OLD_NOISE_THEN_OPTIONAL_NOISE_ECM_NOT_JOINT_M_STEP_MAXIMUM',
        coefficient_reference_noise_exact=_strings(old_noise),noise_estimation='FIXED' if noise_fixed else 'ML_DIVISOR_N')
    return coefficients,noise,proof


def _emission_moments(step,mean,covariance):
    """E[y], Cov(y,x), Cov(y) conditional on all observed training values.

    For missing rows, first condition measurement noise on observed rows at the
    same time. Correlated R changes both the effective loading and intercept.
    Integrate the resulting affine Gaussian law over the smoothed state. This
    retains missing-noise variance and its state cross covariance.
    """
    m=len(step['values']);p=len(mean);expected=list(step['values'])
    observed=[i for i,v in enumerate(expected) if v is not None];missing=[i for i,v in enumerate(expected) if v is None]
    yx=[[F(0)]*p for _ in range(m)];yy=[[F(0)]*m for _ in range(m)]
    if not missing:return expected,yx,yy
    h=[step['measurement'][i] for i in missing];d=[step['observation_intercept'][i] for i in missing]
    noise=[[step['measurement_covariance'][i][j] for j in missing] for i in missing]
    if observed:
        factor=_factor([[step['measurement_covariance'][i][j] for j in observed] for i in observed])
        projection=[_solve_factor(factor,[step['measurement_covariance'][i][j] for j in observed],'INTERNAL_EM_MISSING_NOISE_RANGE') for i in missing]
        h=_add(h,_multiply(projection,[step['measurement'][j] for j in observed]),-1)
        adjustment=_mv(projection,[step['values'][j]-step['observation_intercept'][j] for j in observed])
        d=_bounded([a+b for a,b in zip(d,adjustment)])
        noise=_add(noise,_multiply(projection,[[step['measurement_covariance'][i][j] for j in missing] for i in observed]),-1)
    _factor(noise)
    missing_mean=_bounded([a+b for a,b in zip(_mv(h,mean),d)])
    missing_cross=_multiply(h,covariance);missing_covariance=_add(_multiply(missing_cross,_transpose(h)),noise);_factor(missing_covariance)
    for a,i in enumerate(missing):
        expected[i]=missing_mean[a];yx[i]=missing_cross[a]
        for b,j in enumerate(missing):yy[i][j]=missing_covariance[a][b]
    return expected,yx,yy


def _em_proposal(q,blocks,forward=None,fixed_parameters=None,fixed_entries=None):
    """One exact EM/conditional ECM update, fixed prior and shared affine dynamics.

    Unfixed groups within requested blocks update conditionally from one E-step.
    E-step conditioning uses all training observations. This does not iterate to
    convergence or establish parameter identity. Initial Q/R must be positive
    definite. Missing emissions use exact conditional first/second/cross moments.
    Degenerate candidate observed-data laws have no comparable ordinary density.
    """
    q=_snapshot(q);blocks,fixed_parameters=_em_controls(blocks,fixed_parameters)
    fixed_entries=_em_entry_controls(q,blocks,fixed_parameters,fixed_entries)
    _,_,steps=_validated(q);end=q['conditioning_end_period'];training=[v for v in steps if v['period']<=end]
    model_keys=('transition','state_intercept','process_covariance','measurement','observation_intercept','measurement_covariance')
    if any(any(step[k]!=steps[0][k] for k in model_keys) for step in steps):_fail('EM_TIME_INVARIANT_DYNAMICS_REQUIRED')
    if any(v==0 for key in ('process_covariance','measurement_covariance') for v in _factor(steps[0][key])[1]):_fail('EM_POSITIVE_DEFINITE_NOISE_REQUIRED')
    original,_,states,lag=_run_smoother(q,forward);count=len(training);p=len(q['states']);m=len(q['observables'])
    zero=lambda n,k:[[F(0)]*k for _ in range(n)]
    stats={};candidate=_snapshot(q)
    for block in blocks:
        checkpoint();width=p if block=='STATE_DYNAMICS' else m
        zz=zero(p+1,p+1);yz=zero(width,p+1);yy=zero(width,width)
        for i,step in enumerate(training):
            mean,covariance=states[i+1]
            if block=='STATE_DYNAMICS':
                previous,previous_cov=states[i];zz=_add(zz,_augmented_second(previous,previous_cov))
                # lag rows are previous states and columns current states.
                cross=_add(_transpose(lag[i]),_outer(mean,previous))
                yz=_add(yz,[row+[mean[j]] for j,row in enumerate(cross)])
                yy=_add(yy,_add(covariance,_outer(mean,mean)))
            else:
                values,yx_cov,yy_cov=_emission_moments(step,mean,covariance);zz=_add(zz,_augmented_second(mean,covariance))
                target_cross=_add(yx_cov,_outer(values,mean))
                yz=_add(yz,[row+[values[j]] for j,row in enumerate(target_cross)])
                yy=_add(yy,_add(yy_cov,_outer(values,values)))
        matrix_key,intercept_key,noise_key=_BLOCK_KEYS[block]
        fixed_columns=(list(range(p)) if matrix_key in fixed_parameters else [])+([p] if intercept_key in fixed_parameters else [])
        base=[row+[steps[0][intercept_key][j]] for j,row in enumerate(steps[0][matrix_key])]
        selected=[v for v in fixed_entries if v['field'] in (matrix_key,intercept_key)]
        if selected:
            fixed={(i,j) for i in range(width) for j in fixed_columns}
            fixed.update((v['row'],p if v['column'] is None else v['column']) for v in selected)
            coefficients,noise,proof=_em_masked_regression(zz,yz,yy,count,base,fixed,steps[0][noise_key],noise_key in fixed_parameters)
        else:coefficients,noise,proof=_em_regression(zz,yz,yy,count,base,fixed_columns,steps[0][noise_key] if noise_key in fixed_parameters else None)
        stats[block]=proof
        for step in candidate['steps']:
            if matrix_key not in fixed_parameters:step[matrix_key]=_strings([row[:p] for row in coefficients])
            if intercept_key not in fixed_parameters:step[intercept_key]=[str(row[-1]) for row in coefficients]
            if noise_key not in fixed_parameters:step[noise_key]=_strings(noise)
    _restore_em_entries(candidate,q,fixed_entries)
    return q,blocks,original,candidate,stats,count


def em_step(q,blocks,fixed_parameters=None,fixed_entries=None):
    """One exact EM/ECM step with whole groups or individual coefficients fixed."""
    q=_snapshot(q);expected_code=_code_receipt()
    blocks,fixed_parameters=_em_controls(blocks,fixed_parameters)
    fixed_entries=_em_entry_controls(q,blocks,fixed_parameters,fixed_entries)
    q,blocks,original,candidate,stats,count=_em_proposal(q,blocks,fixed_parameters=fixed_parameters,fixed_entries=fixed_entries);m=len(q['observables']);end=q['conditioning_end_period']
    updated,*_=_run_filter(candidate);before,after=original['log_likelihood'],updated['log_likelihood']
    with localcontext(_CONTEXT):change=None if after is None else str(Decimal(after)-Decimal(before))
    comparison=('SINGULAR_CANDIDATE_NO_COMMON_FULL_DIMENSIONAL_DENSITY' if after is None else
                'INCREASED_AT_DECIMAL80_PRECISION' if Decimal(change)>0 else
                'UNCHANGED_AT_DECIMAL80_PRECISION' if Decimal(change)==0 else 'DECREASED_AT_DECIMAL80_PRECISION_REQUIRES_DIAGNOSIS')
    counts={name:sum(step['values'][name] is not None for step in q['steps'] if step['period']<=end) for name in q['observables']}
    result=dict(schema='societybox-state-space-em-step-v1',method='EXACT_GAUSSIAN_ECM_ONE_STEP' if fixed_entries else 'EXACT_GAUSSIAN_EM_ONE_STEP',status='CALCULATED',
        implementation_status='INDEPENDENT_IMPLEMENTED',native_execution='NOT_RUN',native_parity='NOT_RUN',ui_status='NOT_CONNECTED',
        updated_blocks=list(blocks),parameter_estimation='ONE_ECM_STEP_NOT_CONVERGENCE_OR_JOINT_M_STEP_MAXIMUM' if fixed_entries else 'ONE_EM_STEP_NOT_CONVERGENCE_OR_GLOBAL_MAXIMUM',
        fixed_parameters=fixed_parameters,fixed_entries=fixed_entries,free_parameter_groups=_em_free_groups(q,blocks,fixed_parameters,fixed_entries),
        parameter_identification='NOT_ESTABLISHED_LATENT_COORDINATE_EQUIVALENCE',initial_prior_policy='FIXED',
        sufficient_statistics=stats,candidate_request=candidate,original_log_likelihood=before,candidate_log_likelihood=after,
        log_likelihood_change=change,likelihood_comparison_status=comparison,moment_arithmetic='EXACT_RATIONAL',
        likelihood_arithmetic='DECIMAL_80_HALF_EVEN_NOT_INTERVAL_CERTIFIED',conditioning_end_period=end,
        likelihood_scope='OBSERVED_TRAINING_COMPONENTS_MISSING_INTEGRATED_OUT',
        observed_measurement_count=original['measurement_count'],missing_measurement_count=count*m-original['measurement_count'],
        observation_counts=counts,unobserved_observables=[name for name,value in counts.items() if value==0],
        emission_moment_policy='EXACT_CONDITIONAL_FIRST_SECOND_AND_STATE_CROSS_MOMENTS',
        structural_causal_identification='NOT_ESTABLISHED',parameter_uncertainty='NOT_ASSESSED',
        future_observations_used=False,future_observation_scope='AFTER_DECLARED_CONDITIONING_CUTOFF',
        e_step_information='FULL_CONDITIONING_SAMPLE_RETROSPECTIVE_SMOOTHER',parameters_shared_over_declared_steps=True,
        limitations=['Only time-invariant affine dynamics; original Q and R positive definite; observation pattern assumed ignorable.',
                     'Unfixed parameter groups update conditionally from the same original E-step; fixed groups stay declared.',
                     'The initial prior stays fixed; declared future matrices inherit learned shared parameters without future values.',
                     'Likelihood improvement at Decimal precision is diagnostic, not global convergence or interval certification.'])
    result.update(_assurance(dict(q,em_blocks=blocks,fixed_parameters=fixed_parameters,fixed_entries=fixed_entries),result,expected_code));return result


def _quantize(value,digits):
    with localcontext(Context(prec=digits,rounding=ROUND_HALF_EVEN)):return F(str(+_decimal(value)))


def _quantized_covariance(matrix,digits):
    lower,diagonal=_factor(matrix)
    lower=[[_quantize(v,digits) for v in row] for row in lower];diagonal=[_quantize(v,digits) for v in diagonal]
    result=_multiply([[v*diagonal[j] for j,v in enumerate(row)] for row in lower],_transpose(lower))
    _factor(result);return result


def _quantized_request(q,blocks,digits,fixed_parameters=(),fixed_entries=()):
    result=_snapshot(q)
    for block in blocks:
        matrix,intercept,noise=_BLOCK_KEYS[block];model=q['steps'][0]
        if matrix not in fixed_parameters:
            value=_strings([[_quantize(F(v),digits) for v in row] for row in model[matrix]])
            for step in result['steps']:step[matrix]=[list(row) for row in value]
        if intercept not in fixed_parameters:
            value=list(map(str,[_quantize(F(v),digits) for v in model[intercept]]))
            for step in result['steps']:step[intercept]=list(value)
        if noise not in fixed_parameters:
            value=_strings(_quantized_covariance([[F(v) for v in row] for row in model[noise]],digits))
            for step in result['steps']:step[noise]=[list(row) for row in value]
    return _restore_em_entries(result,q,fixed_entries)


def _parameter_distance(old,new,blocks):
    with localcontext(_CONTEXT):
        values=[]
        for block in blocks:
            matrix,intercept,noise=_BLOCK_KEYS[block]
            for key in (matrix,noise):
                for before,after in zip(old['steps'][0][key],new['steps'][0][key]):
                    for a,b in zip(before,after):
                        a,b=_decimal(F(a)),_decimal(F(b));values.append(abs(b-a)/(1+abs(a)))
            for a,b in zip(old['steps'][0][intercept],new['steps'][0][intercept]):
                a,b=_decimal(F(a)),_decimal(F(b));values.append(abs(b-a)/(1+abs(a)))
        return max(values,default=Decimal(0))


def _expected_complete_loglik(stats,q,blocks):
    """Evaluate Q(theta|old theta) from the original exact sufficient statistics."""
    values=[]
    for block in blocks:
        matrix,intercept,noise=_BLOCK_KEYS[block];model=q['steps'][0];data=stats[block]
        b=[[F(v) for v in row]+[F(model[intercept][i])] for i,row in enumerate(model[matrix])]
        zz=[[F(v) for v in row] for row in data['second_predictor_sum']]
        yz=[[F(v) for v in row] for row in data['target_predictor_sum']]
        yy=[[F(v) for v in row] for row in data['second_target_sum']]
        residual=_add(_add(_add(yy,_multiply(b,_transpose(yz)),-1),_multiply(yz,_transpose(b)),-1),_sandwich(b,zz))
        factor=_factor([[F(v) for v in row] for row in model[noise]])
        if any(v==0 for v in factor[1]):return None
        columns=[_solve_factor(factor,list(col),'INTERNAL_FIT_Q_RANGE') for col in zip(*residual)]
        trace=sum((columns[i][i] for i in range(len(columns))),F(0));_bounded([trace])
        if trace<0:_fail('INTERNAL_FIT_NEGATIVE_Q_TRACE')
        with localcontext(_CONTEXT):values.append(str(Decimal(data['count'])*Decimal(_log_density(factor[1],trace/data['count']))))
    return _sum_logs(values)


def _mixed_request(old,target,blocks,alpha,fixed_parameters=(),fixed_entries=()):
    result=_snapshot(old)
    for block in blocks:
        matrix,intercept,noise=_BLOCK_KEYS[block]
        for key in (matrix,noise):
            if key in fixed_parameters:continue
            value=[[str(F(a)+alpha*(F(b)-F(a))) for a,b in zip(row,new)] for row,new in zip(old['steps'][0][key],target['steps'][0][key])]
            for step in result['steps']:step[key]=[list(row) for row in value]
        if intercept not in fixed_parameters:
            value=[str(F(a)+alpha*(F(b)-F(a))) for a,b in zip(old['steps'][0][intercept],target['steps'][0][intercept])]
            for step in result['steps']:step[intercept]=list(value)
    return _restore_em_entries(result,old,fixed_entries)


def fit_em(q,blocks,optimizer=None,fixed_parameters=None,fixed_entries=None):
    """Quantized generalized EM with numerical Q and observed-likelihood ascent.

    Exact E/M sufficient statistics are followed by declared significant-digit
    parameter quantization. Covariance LDL factors preserve definiteness. Step
    halving accepts only nondecreasing Q and observed likelihood at Decimal80.
    Convergence additionally checks the unquantized M-step distance; a rounded
    fixed point alone cannot pass a tighter tolerance. This is a bounded local
    numerical fit, not an exact MLE, identification or global-optimum proof.
    Fixed groups and individual affine coefficients are preserved through
    quantization and mixing. Entry masks use conditional ECM at the old noise;
    individual covariance constraints and general linear restrictions are unsupported.
    """
    source=_snapshot(q);current=_snapshot(source);expected_code=_code_receipt()
    if optimizer is None:optimizer=dict(max_iterations=100,tolerance=1e-7,parameter_precision_digits=12)
    if (type(optimizer) is not dict or set(optimizer)!={'max_iterations','tolerance','parameter_precision_digits'} or
            type(optimizer['max_iterations']) is not int or not 1<=optimizer['max_iterations']<=200 or
            type(optimizer['tolerance']) not in (int,float) or not 1e-12<=optimizer['tolerance']<=1e-4 or
            type(optimizer['parameter_precision_digits']) is not int or not 8<=optimizer['parameter_precision_digits']<=16):_fail('FIT_OPTIMIZER_INVALID')
    optimizer=dict(optimizer);blocks,fixed_parameters=_em_controls(blocks,fixed_parameters)
    fixed_entries=_em_entry_controls(source,blocks,fixed_parameters,fixed_entries)
    tolerance=Decimal(str(optimizer['tolerance']));trace=[];status='ITERATION_LIMIT';reason=None
    initial_ll=None;current_ll=None;last_raw=None;accepted=0;attempts=0;forward=None
    for iteration in range(1,optimizer['max_iterations']+1):
        checkpoint();attempts=iteration
        try:
            _,blocks,original,proposal,stats,_=_em_proposal(current,blocks,forward,fixed_parameters,fixed_entries)
            before=original['log_likelihood'];current_ll=before
            if initial_ll is None:initial_ll=before
            last_raw=_parameter_distance(current,proposal,blocks);q_before=_expected_complete_loglik(stats,current,blocks)
            row=dict(iteration=iteration,log_likelihood_before=before,log_likelihood_after=before,
                expected_complete_loglik_before=q_before,expected_complete_loglik_after=q_before,
                unquantized_m_step_distance_scaled=str(last_raw),parameter_distance_scaled='0',relative_likelihood_improvement='0',
                step_fraction='0',accepted=False)
            if last_raw==0:
                status='CONVERGED_BY_DECLARED_CHANGE_CRITERION';row['decision']='EXACT_EM_MAP_FIXED_POINT';trace.append(row);break
            if any(data['noise_rank']<(len(source['states']) if block=='STATE_DYNAMICS' else len(source['observables'])) for block,data in stats.items()):
                status='M_STEP_COVARIANCE_BOUNDARY';row['decision']=status;trace.append(row);break
            target=_quantized_request(proposal,blocks,optimizer['parameter_precision_digits'],fixed_parameters,fixed_entries)
            if _parameter_distance(current,target,blocks)==0 and last_raw>tolerance:
                status='PARAMETER_QUANTIZATION_LIMIT';row['decision']=status;trace.append(row);break
            alpha=F(1);chosen=None
            for trial in range(16):
                checkpoint();candidate=target if trial==0 else _mixed_request(current,target,blocks,alpha,fixed_parameters,fixed_entries)
                q_after=_expected_complete_loglik(stats,candidate,blocks)
                if q_after is not None and Decimal(q_after)>=Decimal(q_before):
                    evaluated=_run_filter(candidate);after=evaluated[0]['log_likelihood']
                    if after is not None and Decimal(after)>=Decimal(before):chosen=(candidate,after,q_after,evaluated);break
                alpha/=2
            if chosen is None:
                status='LIKELIHOOD_LINE_SEARCH_FAILED';row['decision']=status;trace.append(row);break
            candidate,after,q_after,evaluated=chosen;distance=_parameter_distance(current,candidate,blocks)
            with localcontext(_CONTEXT):improvement=(Decimal(after)-Decimal(before))/(1+abs(Decimal(before)))
            row.update(log_likelihood_after=after,expected_complete_loglik_after=q_after,parameter_distance_scaled=str(distance),
                relative_likelihood_improvement=str(improvement),step_fraction=str(alpha),accepted=distance!=0,decision='ACCEPTED')
            trace.append(row);current=candidate;current_ll=after;accepted+=int(distance!=0);forward=(digest(canonical(current)),evaluated)
            if improvement<=tolerance and distance<=tolerance and last_raw<=tolerance:
                status='CONVERGED_BY_DECLARED_CHANGE_CRITERION';break
        except ValueError as error:
            code=str(error)
            if any(v in code for v in ('RESOURCE_LIMIT','INPUT_SIZE_LIMIT','NUMBER_MAGNITUDE_LIMIT')):status='RESOURCE_LIMIT';reason=code;break
            if 'EM_UNIDENTIFIED_SECOND_MOMENTS' in code:status='M_STEP_UNIDENTIFIED';reason=code;break
            raise
    counts={name:sum(step['values'][name] is not None for step in source['steps'] if step['period']<=source['conditioning_end_period']) for name in source['observables']}
    result=dict(schema='societybox-state-space-em-fit-v1',method='QUANTIZED_GENERALIZED_EM_WITH_DUAL_ASCENT_CHECKS',status=status,reason=reason,
        implementation_status='INDEPENDENT_IMPLEMENTED',native_execution='NOT_RUN',native_parity='NOT_RUN',ui_status='NOT_CONNECTED',
        model_request=current,optimizer=optimizer,updated_blocks=blocks,iterations=attempts,accepted_updates=accepted,iteration_trace=trace,
        fixed_parameters=fixed_parameters,fixed_entries=fixed_entries,free_parameter_groups=_em_free_groups(source,blocks,fixed_parameters,fixed_entries),
        initial_log_likelihood=initial_ll,log_likelihood=current_ll,last_unquantized_m_step_distance_scaled=None if last_raw is None else str(last_raw),
        convergence_criterion='EXACT_EM_MAP_IDENTITY_OR_LIKELIHOOD_ACTUAL_AND_UNQUANTIZED_PARAMETER_CHANGE_THRESHOLDS',
        parameter_quantization='DECLARED_SIGNIFICANT_DIGITS_HALF_EVEN_COEFFICIENTS_AND_LDL_FACTORS_NO_VARIANCE_FLOOR',
        arithmetic='BOUNDED_EXACT_RATIONAL_E_STEP_WITH_DECLARED_PARAMETER_QUANTIZATION_DECIMAL80_DIAGNOSTICS',
        parameter_identification='NOT_ESTABLISHED_LATENT_COORDINATE_EQUIVALENCE',parameter_uncertainty='NOT_ASSESSED',
        likelihood_scope='OBSERVED_TRAINING_COMPONENTS_MISSING_INTEGRATED_OUT',
        observation_counts=counts,unobserved_observables=[name for name,value in counts.items() if value==0],
        code_integrity='ENTRY_EXIT_DIGEST_MATCHED',forward_reuse='CALL_LOCAL_EXACT_INPUT_DIGEST_BOUND',
        future_observations_used=False,future_observation_scope='AFTER_DECLARED_CONDITIONING_CUTOFF',
        structural_causal_identification='NOT_ESTABLISHED',global_optimality='NOT_ESTABLISHED',
        limitations=['Local numerical convergence criteria are not parameter identification or exact/global maximum likelihood certification.',
                     'Fixed initial prior, time-invariant shared dynamics, ignorable missingness and positive-definite original Q/R.',
                     'Quantization, resource, boundary, unidentified and line-search stops retain the last accepted model.',
                     'Both ascent tests use fixed Decimal approximations, not interval arithmetic.'])
    result.update(_assurance(dict(source,em_blocks=blocks,fit_optimizer=optimizer,fixed_parameters=fixed_parameters,fixed_entries=fixed_entries),result,expected_code));return result


def forecast(q,horizon):
    if type(horizon) is not int or not 1<=horizon<=64:_fail('FORECAST_HORIZON_INVALID')
    q=_snapshot(q);result,steps,states,_=_run_filter(q);future=[s for s in steps if s['period']>q['conditioning_end_period']]
    if len(future)<horizon:_fail('DECLARED_FUTURE_DYNAMICS_REQUIRED')
    mean,covariance=states[-1];forecasts=[]
    for step in future[:horizon]:
        checkpoint();mean,covariance=_predict(mean,covariance,step)
        observation_mean=_bounded([a+b for a,b in zip(_mv(step['measurement'],mean),step['observation_intercept'])])
        observation_covariance=_add(_sandwich(step['measurement'],covariance),step['measurement_covariance']);_factor(observation_covariance)
        forecasts.append(dict(period=step['period'],state=_cell(step['period'],mean,covariance,'FORECAST',q['conditioning_end_period'],result['measurement_count']),
            observation_mean=list(map(str,observation_mean)),observation_covariance=_strings(observation_covariance)))
    result.update(method='EXACT_STATE_SPACE_ORIGIN_FORECAST',forecasts=forecasts,horizon=horizon,
        forecast_scope='MARGINAL_STATE_AND_FUTURE_OBSERVATION_DISTRIBUTIONS_FIXED_DECLARED_FUTURE_DYNAMICS',future_observations_used=False)
    result.update(_assurance(dict(q,forecast_horizon=horizon),result));return result
