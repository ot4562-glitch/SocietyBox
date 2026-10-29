"""Bounded exact likelihood score/Hessian for declared Gaussian parameter charts.

Second-order rational forward differentiation follows scalar conditioning of the
small joint (state, current observed emissions) distribution. No finite-difference
step, observation-square covariance or symbolic/user-code execution is used.
Unlisted entries and the initial prior are held fixed. Inverse observed curvature
is a pointwise diagnostic; standard-error references additionally require PD
curvature and a small declared Newton decrement. Regular MLE validity remains an
external assumption, not a consequence of a rank or stationarity calculation.
"""
from decimal import localcontext
from fractions import Fraction as F
from pathlib import Path
import json

from . import state_space as ss
from .inverse_identification import _row_basis,_source_receipt as _rank_receipt
from .execution_control import checkpoint
from .research_evidence import EvidenceLedger,UnknownRegistry,make_evidence,make_obligation,evaluate_coverage
from .workspace import canonical,digest

_PAIRS={n:tuple((i,j) for i in range(n) for j in range(i,n)) for n in range(1,9)}
_INDEX={n:{pair:i for i,pair in enumerate(pairs)} for n,pairs in _PAIRS.items()}
_MATRICES=('transition','process_covariance','measurement','measurement_covariance')
_VECTORS=('state_intercept','observation_intercept')
_COVARIANCES=('process_covariance','measurement_covariance')


def _fail(reason):raise ValueError('STATE_SPACE_INFERENCE_'+reason)


class _Jet:
    __slots__=('v','g','h')
    def __init__(self,value,gradient,hessian):
        self.v=value;self.g=tuple(gradient);self.h=tuple(hessian);ss._bounded([value,list(self.g),list(self.h)])

    @classmethod
    def seed(cls,value,index,n):return cls(value,[F(i==index) for i in range(n)],[F(0)]*len(_PAIRS[n]))

    def __add__(self,other):
        if isinstance(other,_Jet):return _jet(self.v+other.v,[a+b for a,b in zip(self.g,other.g)],[a+b for a,b in zip(self.h,other.h)])
        return self if other==0 else _jet(self.v+other,self.g,self.h)
    __radd__=__add__

    def __neg__(self):return _jet(-self.v,[-v for v in self.g],[-v for v in self.h])
    def __sub__(self,other):return self+-other
    def __rsub__(self,other):return -self+other

    def __mul__(self,other):
        if not isinstance(other,_Jet):
            if other==0:return F(0)
            if other==1:return self
            return _jet(self.v*other,[v*other for v in self.g],[v*other for v in self.h])
        pairs=_PAIRS[len(self.g)]
        return _jet(self.v*other.v,[a*other.v+self.v*b for a,b in zip(self.g,other.g)],
            [a*other.v+self.v*b+self.g[i]*other.g[j]+self.g[j]*other.g[i] for a,b,(i,j) in zip(self.h,other.h,pairs)])
    __rmul__=__mul__

    def inverse(self):
        v=1/self.v;pairs=_PAIRS[len(self.g)]
        return _jet(v,[-g*v*v for g in self.g],[2*self.g[i]*self.g[j]*v*v*v-h*v*v for h,(i,j) in zip(self.h,pairs)])

    def __truediv__(self,other):return self*other.inverse() if isinstance(other,_Jet) else self*(1/F(other))
    def __rtruediv__(self,other):return self.inverse()*other


def _jet(v,g,h):
    if not any(g) and not any(h):ss._bounded([v]);return v
    return _Jet(v,g,h)


def _parts(v,n):return (v.v,v.g,v.h) if isinstance(v,_Jet) else (v,(F(0),)*n,(F(0),)*len(_PAIRS[n]))


def _value(v):return v.v if isinstance(v,_Jet) else v


def _guard(matrix):
    for row in matrix:
        for v in row:
            if not isinstance(v,_Jet):ss._bounded([v])
    return matrix


def _mm(a,b):return _guard([[sum((x*y for x,y in zip(row,col)),F(0)) for col in zip(*b)] for row in a])


def _add(a,b):return _guard([[v+w for v,w in zip(row,other)] for row,other in zip(a,b)])


def _unpack(values,n):return [[values[_INDEX[n][(min(i,j),max(i,j))]] for j in range(n)] for i in range(n)]


def _differentiate(mean,covariance,steps,parameters):
    n=len(parameters);p=len(mean);score=[F(0)]*n;hessian=[F(0)]*len(_PAIRS[n]);predictable=[F(0)]*len(_PAIRS[n]);periods=[];logs=[]
    last=max((i for i,s in enumerate(steps) if any(v is not None for v in s['values'])),default=-1)
    for t,step in enumerate(steps):
        checkpoint();g=[F(0)]*n;h=[F(0)]*len(_PAIRS[n]);pi=[F(0)]*len(_PAIRS[n]);densities=[]
        if t<=last:
            for index,spec in enumerate(parameters):
                field,row,col=spec['field'],spec['row'],spec['column']
                if col is None:step[field][row]=_Jet.seed(step[field][row],index,n)
                else:
                    step[field][row][col]=_Jet.seed(step[field][row][col],index,n)
                    if field in _COVARIANCES:step[field][col][row]=step[field][row][col]
            mean=[v+d for v,d in zip([row[0] for row in _mm(step['transition'],[[v] for v in mean])],step['state_intercept'])]
            _guard([mean])
            covariance=_add(_mm(_mm(step['transition'],covariance),ss._transpose(step['transition'])),step['process_covariance'])
            ss._factor([[_value(v) for v in row] for row in covariance])
            observed=[j for j,v in enumerate(step['values']) if v is not None]
            if observed:
                loading=[step['measurement'][j] for j in observed]
                cross=_mm(covariance,ss._transpose(loading))
                innovation=_add(_mm(loading,cross),[[step['measurement_covariance'][i][j] for j in observed] for i in observed])
                emission=[v+step['observation_intercept'][j] for v,j in zip([row[0] for row in _mm(loading,[[v] for v in mean])],observed)]
                _guard([emission])
                joint_mean=mean+emission
                joint_cov=[row+cross[i] for i,row in enumerate(covariance)]+[row+innovation[i] for i,row in enumerate(ss._transpose(cross))]
                for j in observed:
                    _,mg,_=_parts(joint_mean[p],n)
                    error=step['values'][j]-joint_mean[p];variance=joint_cov[p][p];v,vg,vh=_parts(variance,n)
                    if v<=0:_fail('NONDEGENERATE_INNOVATIONS_REQUIRED')
                    ratio=error*error/variance;rv,rg,rh=_parts(ratio,n)
                    ss._bounded([_value(error),v,rv])
                    for i in range(n):g[i]-=(vg[i]/v+rg[i])/2
                    for k,(i,l) in enumerate(_PAIRS[n]):
                        h[k]-=(vh[k]/v-vg[i]*vg[l]/(v*v)+rh[k])/2
                        pi[k]+=mg[i]*mg[l]/v+vg[i]*vg[l]/(2*v*v)
                    ss._bounded([g,h,pi]);densities.append(ss._log_density([v],rv))
                    remaining=[i for i in range(len(joint_mean)) if i!=p];column=[row[p] for row in joint_cov]
                    gain=[column[i]/variance for i in remaining]
                    joint_mean=[joint_mean[i]+a*error for i,a in zip(remaining,gain)]
                    _guard([joint_mean])
                    joint_cov=_guard([[joint_cov[i][k]-a*column[k] for k in remaining] for i,a in zip(remaining,gain)])
                mean,covariance=joint_mean,joint_cov
                ss._factor([[_value(v) for v in row] for row in covariance])
        density=ss._sum_logs(densities);logs.append(density)
        score=[a+b for a,b in zip(score,g)];hessian=[a+b for a,b in zip(hessian,h)];predictable=[a+b for a,b in zip(predictable,pi)];ss._bounded([score,hessian,predictable])
        periods.append(dict(period=step['period'],observed_count=sum(v is not None for v in step['values']),score_exact=list(map(str,g)),
            hessian_exact=ss._strings(_unpack(h,n)),predictable_information_exact=ss._strings(_unpack(pi,n)),log_density=density))
    return score,_unpack(hessian,n),_unpack(predictable,n),periods,ss._sum_logs(logs)


def _chart(q,steps,parameters):
    if type(parameters) is not list or not 1<=len(parameters)<=8:_fail('PARAMETER_CHART_INVALID')
    try:raw=canonical(parameters)
    except (TypeError,ValueError,RecursionError):_fail('PARAMETER_CHART_INVALID')
    if len(raw)>8192:_fail('PARAMETER_CHART_SIZE_LIMIT')
    parameters=json.loads(raw);seen=set();names=set();values=[]
    p,m=len(q['states']),len(q['observables'])
    for spec in parameters:
        if type(spec) is not dict or set(spec)!={'name','field','row','column'}:_fail('PARAMETER_CHART_INVALID')
        name,field,row,col=spec['name'],spec['field'],spec['row'],spec['column']
        if not ss._text(name) or name in names or type(field) is not str or field not in _MATRICES+_VECTORS:_fail('PARAMETER_CHART_INVALID')
        height=p if field in ('transition','state_intercept','process_covariance') else m
        if type(row) is not int or not 0<=row<height:_fail('PARAMETER_CHART_INVALID')
        if field in _VECTORS:
            if col is not None:_fail('PARAMETER_CHART_VECTOR_COLUMN_MUST_BE_NULL')
        else:
            width=m if field=='measurement_covariance' else p
            if type(col) is not int or not 0<=col<width:_fail('PARAMETER_CHART_INVALID')
        key=(field,min(row,col),max(row,col)) if field in _COVARIANCES else (field,row,col)
        if key in seen:_fail('PARAMETER_CHART_DUPLICATE_COORDINATE')
        seen.add(key);names.add(name)
        nominal=steps[0][field][row] if col is None else steps[0][field][row][col]
        if any((s[field][row] if col is None else s[field][row][col])!=nominal for s in steps):_fail('SHARED_TRAINING_ENTRY_REQUIRED')
        if field in _COVARIANCES and any(v==0 for s in steps for v in ss._factor(s[field])[1]):_fail('COVARIANCE_PARAMETER_INTERIOR_REQUIRES_PD')
        values.append(nominal)
    last=max((i+1 for i,s in enumerate(steps) if any(v is not None for v in s['values'])),default=0)
    if last*(p+m)**3*(len(parameters)+1)**2>5000000:_fail('RESOURCE_WORK_LIMIT')
    return parameters,values


def _code_receipt():
    code=ss._code_receipt();code.update(_rank_receipt());code['state_space_inference.py']=digest(Path(__file__).read_bytes());return code


def _assurance(q,parameters,tolerance,result,expected_code,hypothesis=None):
    code=_code_receipt()
    if code!=expected_code:_fail('CODE_CHANGED_DURING_CALCULATION')
    subject_data=dict(model_request=q,parameters=parameters,stationarity_tolerance=str(tolerance))
    if hypothesis is not None:subject_data['hypothesis']=hypothesis
    subject=digest(canonical(subject_data))
    scope=dict(states=q['states'],observables=q['observables'],units=q['units'],conditioning_end_period=q['conditioning_end_period'],
        parameter_chart=parameters,other_entries='FIXED',initial_prior='FIXED',stationarity_tolerance=str(tolerance))
    if hypothesis is not None:scope['hypothesis_sha256']=digest(canonical(hypothesis))
    dimensions=['parameter_chart','regular_mle','gaussian_noise','source_vintage','causal_identification','independent_replication']
    if hypothesis is not None:dimensions.append('null_regularity')
    evidence=make_evidence(subject_sha256=subject,producer=dict(tool_id='econ.state_space_information',version='1',code_sha256=digest(canonical(code))),
        operation=result['method'],authority='LOCAL_EXACT_RATIONAL_LIKELIHOOD_DIFFERENTIATION',epistemic_state='DERIVED',
        proof_kind='DECLARED_GAUSSIAN_SCORE_CURVATURE' if hypothesis is None else 'DECLARED_GAUSSIAN_LINEAR_RESTRICTION_WALD',scope=scope,result=result,parent_evidence_ids=[],
        coverage=dict(status='PARTIAL',unknown_dimensions=dimensions),
        limitations=['Local curvature rank is not structural or causal identification.',
                     'Small score and PD observed curvature do not prove global optimality or finite-sample inference.',
                     'Unlisted nuisance entries and initial prior are held fixed; no parameter fitting or LR testing is performed.',
                     'A null witness and predictable information check do not establish regularity over the null or justify a chi-square approximation.'])
    ledger=EvidenceLedger([evidence]);registry=UnknownRegistry(ledger);unknowns=[];obligations=[]
    questions=[
        ('parameter_chart','Are all estimated nuisance parameters included, or are unlisted parameters independently known and fixed?'),
        ('regular_mle','Is this a regular interior consistent likelihood estimate with adequate identification, sample size and asymptotic approximation?'),
        ('gaussian_noise','Are Gaussian independent noise, covariance models and an ignorable missingness pattern justified?'),
        ('source_vintage','Were observations and declared parameters available at this origin with verified source rights and units?'),
        ('causal_identification','What separate research design supports any causal interpretation?'),
        ('independent_replication','Have independent numerical calculations and substantive diagnostics validated these derivatives and uncertainty references?')]
    if hypothesis is not None:questions.append(('null_regularity',
        'Is the relevant null regular and identified, with a justified chi-square limit, including unit-root/boundary and weak-identification exclusions? A single declared null witness and realized predictable information do not prove this.'))
    for claim,question in questions:
        unknown=registry.register(question,scope,'EXTERNAL_SCIENTIFIC_VALIDATION','STATE_SPACE_INFORMATION_CHECK');unknowns.append(unknown)
        obligations.append(make_obligation(claim_id=claim,subject_sha256=subject,scope=scope,required_authority='EXTERNAL_SCIENTIFIC_VALIDATION',
            required_proof_kind='STATE_SPACE_INFORMATION_CHECK',verifier_id='econ.state_space_information',dependencies=[],unknown_ids=[unknown['unknown_id']],blocking_reasons=[],next_probe=question))
    return dict(input_sha256=subject,model_request_sha256=digest(canonical(q)),code_sha256=code,evidence=[evidence],unknowns=unknowns,
        obligations=obligations,coverage=evaluate_coverage(obligations,ledger,registry))


def likelihood_information(q,parameters,stationarity_tolerance=1e-6):
    """Aggregate score, observed Hessian/rank, and conditional local SE references.

    Coordinates are common scalar training entries; covariance off-diagonals move
    both symmetric entries. All unlisted model entries and the initial prior are
    fixed. This evaluates a declared point, not an estimator or a finite-sample
    uncertainty guarantee. Singular observed-data likelihoods are unsupported.
    """
    q=ss._snapshot(q);expected_code=_code_receipt();mean,covariance,all_steps=ss._validated(q)
    steps=[s for s in all_steps if s['period']<=q['conditioning_end_period']];parameters,values=_chart(q,steps,parameters)
    tolerance=ss._number(stationarity_tolerance)
    if not F(1,10**12)<=tolerance<=F(1,100):_fail('STATIONARITY_TOLERANCE_INVALID')
    score,hessian,predictable,periods,likelihood=_differentiate(mean,covariance,steps,parameters);information=[[-v for v in row] for row in hessian];n=len(parameters)
    pbasis,_,pnull=_row_basis(predictable);ss._bounded([row for _,row in pbasis]+pnull)
    pfactor=ss._factor(predictable);p_positive=all(v>0 for v in pfactor[1])
    basis,_,nullspace=_row_basis(information);ss._bounded([row for _,row in basis]+nullspace)
    factor=None
    try:factor=ss._factor(information)
    except ValueError as error:
        if 'COVARIANCE_NOT_POSITIVE_SEMIDEFINITE' not in str(error):raise
    positive=factor is not None and all(v>0 for v in factor[1]);inverse=None;decrement=None;standard_errors=None
    status='SINGULAR_OR_NONCONCAVE_CURVATURE'
    if positive:
        columns=[ss._solve_factor(factor,[F(i==j) for i in range(n)],'INTERNAL_INFORMATION_SOLVE_RANGE') for j in range(n)]
        inverse=ss._transpose(columns);decrement=ss.iv._dot(score,ss._mv(inverse,score));ss._bounded([decrement])
        if decrement<0:_fail('INTERNAL_NEGATIVE_NEWTON_DECREMENT')
        status='NOT_STATIONARY_AT_DECLARED_TOLERANCE'
        if decrement<=tolerance*tolerance:
            status='CONDITIONAL_ASYMPTOTIC_REFERENCE_AVAILABLE'
            with localcontext(ss._CONTEXT):standard_errors=[str(+ss._decimal(inverse[i][i]).sqrt()) for i in range(n)]
    result=dict(schema='societybox-state-space-information-v1',method='SECOND_ORDER_EXACT_RATIONAL_INNOVATION_DIFFERENTIATION',status='CALCULATED',
        implementation_status='INDEPENDENT_IMPLEMENTED',native_execution='NOT_RUN',native_parity='NOT_RUN',ui_status='NOT_CONNECTED',
        parameter_order=[v['name'] for v in parameters],parameter_values_exact=list(map(str,values)),score_exact=list(map(str,score)),
        hessian_exact=ss._strings(hessian),observed_information_exact=ss._strings(information),information_rank=len(basis),
        predictable_information_exact=ss._strings(predictable),predictable_information_rank=len(pbasis),
        predictable_information_positive_definite=p_positive,predictable_information_nullspace_exact=ss._strings(pnull),
        predictable_information_scope='CONDITIONAL_ON_ACTUAL_PRIOR_OBSERVATIONS_AND_SCALAR_CONDITIONING_ORDER_NOT_UNCONDITIONAL_FISHER',
        information_nullspace_exact=ss._strings(nullspace),information_positive_definite=positive,
        inverse_information_exact=None if inverse is None else ss._strings(inverse),
        newton_decrement_squared_exact=None if decrement is None else str(decrement),stationarity_tolerance_exact=str(tolerance),
        standard_error_reference_decimal=standard_errors,reference_status=status,period_contributions=periods,log_likelihood=likelihood,
        conditioning_end_period=q['conditioning_end_period'],measurement_count=sum(v['observed_count'] for v in periods),
        moment_derivative_arithmetic='EXACT_RATIONAL',likelihood_and_standard_error_arithmetic='DECIMAL_80_HALF_EVEN_NOT_INTERVAL_CERTIFIED',
        likelihood_normalization='SUM_OVER_OBSERVED_TRAINING_COMPONENTS_NOT_AVERAGE',
        coordinate_convention='COMMON_TRAINING_ENTRIES_SYMMETRIC_COVARIANCE_OFF_DIAGONALS_MOVE_BOTH_ENTRIES',
        conditioning='ALL_UNLISTED_MODEL_ENTRIES_AND_INITIAL_PRIOR_FIXED',
        inverse_scope='POINTWISE_CURVATURE_REQUIRES_REGULAR_MLE_ASSUMPTIONS_FOR_ESTIMATOR_COVARIANCE',
        uncertainty_reference='CONDITIONAL_ASYMPTOTIC_REGULAR_MLE_NOT_FINITE_SAMPLE_OR_CAUSAL',
        parameter_estimation='NOT_PERFORMED',structural_identification='NOT_ESTABLISHED',causal_identification='NOT_ESTABLISHED',
        hypothesis_tests='NOT_PERFORMED_SEPARATE_WALD_API_AVAILABLE')
    result.update(_assurance(q,parameters,tolerance,result,expected_code));return result


def wald_test(q,parameters,restriction,null_witness,stationarity_tolerance=1e-6):
    """Evaluate a linear restriction at a declared, locally stationary point.

    The witness is one point on the null; its interior and predictable-information
    checks do not establish regularity of the whole null or asymptotic validity.
    """
    q=ss._snapshot(q);expected_code=_code_receipt()
    try:
        raw=canonical(dict(parameters=parameters,restriction=restriction,null_witness=null_witness))
        if len(raw)>16384:_fail('WALD_INPUT_SIZE_LIMIT')
        owned=json.loads(raw)
    except (TypeError,ValueError,OverflowError,RecursionError) as error:
        if str(error).startswith('STATE_SPACE_INFERENCE_'):raise
        _fail('WALD_JSON_INPUT_REQUIRED')
    parameters=owned['parameters'];restriction=owned['restriction'];null_witness=owned['null_witness']
    if type(parameters) is not list or not 1<=len(parameters)<=8:_fail('PARAMETER_CHART_INVALID')
    n=len(parameters)
    if (type(restriction) is not dict or set(restriction)!={'matrix','target'}
        or type(restriction['matrix']) is not list or not 1<=len(restriction['matrix'])<=n
        or any(type(row) is not list or len(row)!=n for row in restriction['matrix'])
        or type(restriction['target']) is not list or len(restriction['target'])!=len(restriction['matrix'])
        or type(null_witness) is not list or len(null_witness)!=n):_fail('WALD_RESTRICTION_DIMENSIONS_INVALID')
    matrix=[[ss._number(v) for v in row] for row in restriction['matrix']]
    target=list(map(ss._number,restriction['target']));witness=list(map(ss._number,null_witness));m=len(matrix)
    basis,_,nullspace=_row_basis(matrix);ss._bounded([row for _,row in basis]+nullspace)
    if len(basis)!=m:_fail('WALD_RESTRICTION_RANK_DEFICIENT')
    if ss._mv(matrix,witness)!=target:_fail('WALD_NULL_WITNESS_CONSTRAINT_MISMATCH')
    point=likelihood_information(q,parameters,stationarity_tolerance)
    if point['reference_status']!='CONDITIONAL_ASYMPTOTIC_REFERENCE_AVAILABLE':_fail('WALD_REQUIRES_STATIONARY_POSITIVE_CURVATURE')
    tolerance=ss._number(stationarity_tolerance);theta=list(map(F,point['parameter_values_exact']))
    inverse=[[F(v) for v in row] for row in point['inverse_information_exact']]
    difference=[a-b for a,b in zip(ss._mv(matrix,theta),target)]
    variance=ss._multiply(ss._multiply(matrix,inverse),ss._transpose(matrix));ss._bounded(variance)
    factor=ss._factor(variance);solution=ss._solve_factor(factor,difference,'INTERNAL_WALD_SOLVE_RANGE')
    statistic=ss.iv._dot(difference,solution);ss._bounded([statistic])
    if statistic<0:_fail('INTERNAL_NEGATIVE_WALD_STATISTIC')
    null_q=ss._snapshot(q)
    for step in null_q['steps']:
        if step['period']>q['conditioning_end_period']:continue
        for spec,value in zip(parameters,witness):
            field,row,col=spec['field'],spec['row'],spec['column'];value=str(value)
            if col is None:step[field][row]=value
            else:
                step[field][row][col]=value
                if field in _COVARIANCES:step[field][col][row]=value
    null_info=None;null_error=None
    try:null_info=likelihood_information(null_q,parameters,stationarity_tolerance)
    except ValueError as error:
        if not any(v in str(error) for v in ('COVARIANCE_PARAMETER_INTERIOR_REQUIRES_PD',
            'COVARIANCE_NOT_POSITIVE_SEMIDEFINITE','NONDEGENERATE_INNOVATIONS_REQUIRED')):raise
        null_error=str(error)
    if null_info is None:null_status='BLOCKED_NULL_WITNESS_BOUNDARY_OR_DEGENERATE'
    elif not null_info['predictable_information_positive_definite']:null_status='BLOCKED_NULL_PREDICTABLE_INFORMATION_SINGULAR'
    else:null_status='NULL_CHECK_PASSED_NOT_REGULARITY_PROOF'
    probability=ss.iv._chi_probability(statistic,m) if null_status=='NULL_CHECK_PASSED_NOT_REGULARITY_PROOF' else dict(
        p_value=None,log_p_value=None,p_value_status='BLOCKED_NULL_REGULARITY_CHECK')
    result=dict(schema='societybox-state-space-wald-v1',method='EXACT_GAUSSIAN_LINEAR_RESTRICTION_WALD',status='CALCULATED',
        implementation_status='INDEPENDENT_IMPLEMENTED',native_execution='NOT_RUN',native_parity='NOT_RUN',ui_status='NOT_CONNECTED',
        parameter_order=point['parameter_order'],parameter_values_exact=point['parameter_values_exact'],restriction=restriction,
        null_witness=null_witness,restriction_difference_exact=list(map(str,difference)),restriction_covariance_exact=ss._strings(variance),
        statistic_exact=str(statistic),degrees_of_freedom=m,null_regularity_status=null_status,null_witness_error=null_error,
        null_model_sha256=digest(canonical(null_q)),null_information_input_sha256=None if null_info is None else null_info['input_sha256'],
        null_predictable_information_rank=None if null_info is None else null_info['predictable_information_rank'],
        null_predictable_information_exact=None if null_info is None else null_info['predictable_information_exact'],
        null_information_scope='ONE_DECLARED_NULL_WITNESS_CONDITIONAL_ON_ACTUAL_PRIOR_OBSERVATIONS_AND_SCALAR_ORDER',
        point_information_input_sha256=point['input_sha256'],stationarity_tolerance_exact=str(tolerance),
        conditioning='ALL_UNLISTED_MODEL_ENTRIES_AND_INITIAL_PRIOR_FIXED',parameter_estimation='NOT_PERFORMED',
        reference_distribution='CONDITIONAL_ASYMPTOTIC_CHI_SQUARE_REQUIRES_UNVERIFIED_REGULARITY',
        structural_identification='NOT_ESTABLISHED',causal_identification='NOT_ESTABLISHED',**probability)
    hypothesis=dict(restriction=restriction,null_witness=null_witness)
    result.update(_assurance(q,parameters,tolerance,result,expected_code,hypothesis));return result
