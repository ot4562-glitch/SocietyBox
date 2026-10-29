"""Bounded iterated linear GMM and continuously updated GMM (CUE).

Iterates use binary64 to bound arithmetic growth; the criterion, analytic
derivatives and stopping tests are evaluated as exact rationals at those
points. Convergence is a fixed-point or local stationarity statement, not
an estimator error bound, global optimum, weak-IV or causal certificate.
"""
from collections import Counter
from fractions import Fraction as F
from pathlib import Path
import math

from . import instrumental_variables as iv
from .execution_control import checkpoint
from .instrumental_variables_likelihood import _positive_status
from .regression_sensitivity import _bartlett_apply
from .workspace import digest


def _extra_code():
    return {name:digest(Path(__file__).with_name(name).read_bytes()) for name in
            ('iterative_gmm.py','instrumental_variables_likelihood.py')}


def _prepare(q):
    y,x,z,_,_=iv._validated(q)
    method=q['covariance']['estimator']
    if method=='CLASSICAL': iv._fail('GMM_ROBUST_COVARIANCE_REQUIRED')
    initial=iv._inverse_small(iv._cross(z,z),failure='INSTRUMENT_RANK_DEFICIENT')
    zx=iv._cross(z,x); zy=[iv._dot(col,y) for col in zip(*z)]
    first=iv._gmm_fit(x,y,z,initial,cross_products=(zx,zy))
    return dict(y=y,x=x,z=z,zx=zx,zy=zy,initial_weight=initial,first_beta=first[0],
                covariance=q['covariance'],weight_mode='CLUSTER_CR0' if method.startswith('CLUSTER_') else
                'HAC_BARTLETT' if method=='HAC_BARTLETT' else 'HC0')


def _kernel_apply(values,data):
    method=data['weight_mode']; cov=data['covariance']
    if method=='HAC_BARTLETT': return _bartlett_apply(values,cov['bandwidth'])
    if method=='CLUSTER_CR0':
        totals={}
        for label,v in zip(cov['clusters'],values): totals[label]=totals.get(label,F(0))+v
        return [totals[label] for label in cov['clusters']]
    return list(values)


def _evaluate(data,beta,*,derivatives=False):
    checkpoint()
    x,y,z=data['x'],data['y'],data['z']; n,k=len(y),len(beta); cov=data['covariance']
    errors=[v-iv._dot(row,beta) for row,v in zip(x,y)]
    meat=iv._moment_meat(z,errors,data['weight_mode'],cov['clusters'],k,cov.get('bandwidth'))
    if _positive_status(meat)!='PD': iv._fail('GMM_MOMENT_COVARIANCE_SINGULAR')
    weight=iv._inverse_small(meat,failure='GMM_MOMENT_COVARIANCE_SINGULAR')
    moments=[v-iv._dot(row,beta) for row,v in zip(data['zx'],data['zy'])]
    a=[iv._dot(row,moments) for row in weight]
    criterion=iv._dot(moments,a)
    if criterion<0: iv._fail('GMM_CRITERION_NEGATIVE')
    result=dict(beta=list(beta),errors=errors,meat=meat,weight=weight,moments=moments,a=a,criterion=criterion)
    if not derivatives: return result
    dcols=[list(row) for row in zip(*data['zx'])]
    za=[iv._dot(row,a) for row in z]
    h=[u*v for u,v in zip(errors,za)]; kh=_kernel_apply(h,data)
    bcols=[[-row[j]*v for row,v in zip(x,za)] for j in range(k)]
    kbcols=[_kernel_apply(column,data) for column in bcols]
    sj_a=[]
    for j in range(k):
        checkpoint()
        factors=[-row[j]*v+u*w for row,v,u,w in zip(x,kh,errors,kbcols[j])]
        sj_a.append([iv._dot(list(column),factors) for column in zip(*z)])
    rcols=[[-v-w for v,w in zip(dcol,sj)] for dcol,sj in zip(dcols,sj_a)]
    gradient=[-2*iv._dot(col,a)-iv._dot(a,sj) for col,sj in zip(dcols,sj_a)]
    weighted_r=iv._multiply(rcols,weight)
    hessian=[[2*(iv._dot(weighted_r[i],rcols[j])-iv._dot(bcols[i],kbcols[j]))
              for j in range(k)] for i in range(k)]
    mapping=iv._multiply(dcols,weight); information=iv._multiply(mapping,data['zx'])
    metric=[[2*v for v in row] for row in information]
    if _positive_status(metric)!='PD': iv._fail('GMM_UNDERIDENTIFIED_INFORMATION')
    scaled=max(g*g/(metric[j][j]*(1+criterion)) for j,g in enumerate(gradient))
    result.update(gradient=gradient,hessian=hessian,metric=metric,information=information,
                  scaled_gradient_squared=scaled,hessian_status=_positive_status(hessian))
    return result


def _numbers(raw,k):
    if type(raw) is not list or len(raw)!=k: iv._fail('GMM_STARTING_DIMENSIONS_INVALID')
    try: return list(map(iv._num,raw))
    except ValueError as error: iv._fail('GMM_STARTING_NUMBER_INVALID:'+str(error))


def _binary_point(values):
    try: point=list(map(float,values))
    except OverflowError as error: raise ValueError('IV_GMM_BINARY64_RANGE') from error
    if any(not math.isfinite(v) for v in point): iv._fail('GMM_BINARY64_RANGE')
    return list(map(F,point))


def _options(owned):
    iterations=owned['max_iterations']
    if type(iterations) is not int or not 1<=iterations<=200: iv._fail('GMM_ITERATION_LIMIT_INVALID')
    try: tol=iv._num(owned['tolerance'])
    except ValueError as error: iv._fail('GMM_TOLERANCE_INVALID:'+str(error))
    if not 0<tol<=F(1,100): iv._fail('GMM_TOLERANCE_INVALID')
    return tol


def _predictor_residual(data,beta,direction):
    magnitude=max(abs(iv._dot(row,direction)) for row in data['x'])
    return magnitude/(1+max(abs(iv._dot(row,beta)) for row in data['x']))


def _diagnostic(value):
    try: v=float(value)
    except OverflowError: return None
    return v if math.isfinite(v) else None


def _negative_curvature(hessian):
    """Exact congruence elimination returns v with v'Hv<0, if one exists."""
    k=len(hessian); work=[list(row) for row in hessian]
    basis=[[F(int(i==j)) for i in range(k)] for j in range(k)]
    for j in range(k):
        checkpoint(); pivot=work[j][j]
        if pivot<0: return basis[j]
        if pivot==0:
            for i in range(j+1,k):
                if work[i][j]:
                    scale=-(abs(work[i][i])+1)/(2*work[i][j])
                    return [scale*v+w for v,w in zip(basis[j],basis[i])]
            continue
        for i in range(j+1,k):
            factor=work[i][j]/pivot
            basis[i]=[v-factor*w for v,w in zip(basis[i],basis[j])]
        for i in range(j+1,k):
            for r in range(i,k):
                work[i][r]=work[r][i]=work[i][r]-work[i][j]*work[r][j]/pivot
    return None


def _finish(q,data,state,method,options,initial_code,**diagnostics):
    n,k,l=len(data['y']),len(state['beta']),len(data['z'][0]); cov=q['covariance']
    mapping=iv._multiply([list(row) for row in zip(*data['zx'])],state['weight'])
    bread=iv._inverse_small(iv._multiply(mapping,data['zx']),failure='GMM_UNDERIDENTIFIED_INFORMATION')
    influence=iv._multiply(bread,mapping)
    correction=F(n,n-k) if cov['estimator']=='HC1' else F(1)
    if cov['estimator']=='CLUSTER_CR1':
        groups=len(set(cov['clusters'])); correction=F(groups,groups-1)*F(n-1,n-k)
    meat=[[correction*v for v in row] for row in state['meat']]
    variance=iv._multiply(iv._multiply(influence,meat),[list(row) for row in zip(*influence)])
    normal=[iv._dot(row,state['moments']) for row in mapping]
    evaluation_only=method=='CONTINUOUSLY_UPDATED_GMM_OBJECTIVE'
    calibration=('NOT_CALIBRATED_OBJECTIVE_EVALUATION' if evaluation_only else
                 'ASYMPTOTIC_ASSUMPTIONS_UNVERIFIED' if l>k else 'NOT_APPLICABLE_EXACTLY_IDENTIFIED')
    probability=(iv._chi_probability(state['criterion'],l-k) if l>k and not evaluation_only else
                 dict(p_value=None,log_p_value=None,p_value_status=calibration))
    result=dict(schema='societybox-iterative-gmm-result-v1',method=method,
        status='COMPUTED_ASSUMPTIONS_UNVERIFIED',implementation_status='INDEPENDENT_IMPLEMENTED',
        native_execution='NOT_RUN',native_parity='NOT_RUN',ui_status='NOT_CONNECTED',identification='IDENTIFICATION_NOT_ESTABLISHED',
        arithmetic=('EXACT_RATIONAL_AT_DECLARED_COEFFICIENTS' if method=='CONTINUOUSLY_UPDATED_GMM_OBJECTIVE' else
                    'BINARY64_ITERATES_EXACT_RATIONAL_CRITERION_DERIVATIVES_AND_STOPPING_TESTS'),
        parameter_order=q['model']['exogenous']+q['model']['endogenous'],coefficients=list(map(float,state['beta'])),
        coefficients_exact=list(map(str,state['beta'])),covariance_exact=[list(map(str,row)) for row in variance],
        covariance_estimator=cov['estimator'],covariance_specification=cov,
        covariance_convention='LINEAR_GMM_SANDWICH_NOT_CUE_OBJECTIVE_HESSIAN_INVERSE',
        structural_residuals_exact=list(map(str,state['errors'])),moment_sums_exact=list(map(str,state['moments'])),
        moment_covariance_exact=[list(map(str,row)) for row in state['meat']],
        weight_matrix_exact=[list(map(str,row)) for row in state['weight']],moment_centering='UNCENTERED',
        weight_convention='INVERSE_SUM_MOMENT_COVARIANCE_AT_REPORTED_COEFFICIENTS_NO_DF_SCALING',
        fixed_weight_normal_residual_exact=list(map(str,normal)),gmm_criterion_exact=str(state['criterion']),
        overidentification_test=dict(degrees_of_freedom=l-k,status=calibration,**probability),
        weak_instrument_robust=False,
        global_optimality_certified=False,observations=n,residual_df=n-k,options=options,**diagnostics)
    if l>k and not evaluation_only: result['distribution_arithmetic']='BINARY64_NOT_INTERVAL_CERTIFIED'
    if 'gradient' in state:
        result.update(gradient_exact=list(map(str,state['gradient'])),hessian_exact=[list(map(str,row)) for row in state['hessian']],
                      hessian_status='POSITIVE_DEFINITE' if state['hessian_status']=='PD' else state['hessian_status'],
                      scaled_gradient_squared_exact=str(state['scaled_gradient_squared']))
    bound=dict(q,gmm_iteration_options=dict(algorithm=method,**options))
    assurance=iv._assurance(bound,result,extra_code=_extra_code())
    if assurance['code_sha256']!=initial_code: iv._fail('GMM_CODE_CHANGED_DURING_RUN')
    result.update(assurance)
    return result


def _initial_code():
    return dict(iv._code_receipt(),**_extra_code())


def cue_objective(q,coefficients):
    """Exact uncentered CUE objective, gradient and Hessian at declared values.

    Q=m'S(beta)^-1*m, m=Z'(y-X beta), S=scores' K scores. Includes derivatives
    of S; accepts HC0/HC1, one-way CR0/CR1 and Bartlett HAC. Small-sample
    covariance factors do not scale Q. Singular S has no objective value.
    """
    checkpoint()
    owned=iv._snapshot(dict(study=q,coefficients=coefficients),limit=196608); q=owned['study']
    code=_initial_code(); data=_prepare(q); beta=_numbers(owned['coefficients'],len(data['x'][0]))
    state=_evaluate(data,beta,derivatives=True)
    return _finish(q,data,state,'CONTINUOUSLY_UPDATED_GMM_OBJECTIVE',dict(coefficients=owned['coefficients']),code)


def iterated_gmm(q,*,starting=None,weight_matrix=None,max_iterations=100,tolerance='0.000000001'):
    """Iterate efficient weights until the exact scaled predictor map residual
    is <= tolerance. This is not minimization of the CUE objective. Starting
    and initial weight are mutually exclusive; default starting is 2SLS.
    Bounds: existing IV dimensions/numbers, 1..200 updates, 0<tolerance<=.01.
    Iterates are rounded to binary64 and rechecked exactly; no false success
    on exhausted budget, singular covariance or resolution stagnation.
    """
    checkpoint()
    owned=iv._snapshot(dict(study=q,starting=starting,weight_matrix=weight_matrix,
                           max_iterations=max_iterations,tolerance=tolerance),limit=196608)
    q=owned.pop('study'); tolerance_exact=_options(owned); code=_initial_code(); data=_prepare(q)
    if owned['starting'] is not None and owned['weight_matrix'] is not None: iv._fail('GMM_UNUSED_INITIAL_WEIGHT')
    if owned['starting'] is not None: beta=_numbers(owned['starting'],len(data['x'][0]))
    elif owned['weight_matrix'] is not None:
        weight=iv._positive_weight(owned['weight_matrix'],len(data['z'][0]))
        beta=iv._gmm_fit(data['x'],data['y'],data['z'],weight,cross_products=(data['zx'],data['zy']))[0]
    else: beta=data['first_beta']
    beta=_binary_point(beta); trace=[]
    for iteration in range(max_iterations):
        checkpoint(); state=_evaluate(data,beta)
        fit=iv._gmm_fit(data['x'],data['y'],data['z'],state['weight'],cross_products=(data['zx'],data['zy']))
        direction=[v-w for v,w in zip(fit[0],beta)]
        residual=_predictor_residual(data,beta,direction)
        trace.append(dict(iteration=iteration,criterion=_diagnostic(state['criterion']),predictor_residual=_diagnostic(residual)))
        if residual<=tolerance_exact:
            return _finish(q,data,state,'ITERATED_LINEAR_GMM',owned,code,iterations=iteration+1,iteration_trace=trace,
                fixed_point_predictor_residual_exact=str(residual),fixed_point_coefficient_change_exact=list(map(str,direction)),
                convergence_scope='FIXED_POINT_RESIDUAL_NOT_PARAMETER_ERROR_CERTIFICATE')
        candidate=_binary_point(fit[0])
        if candidate==beta: iv._fail('GMM_BINARY64_RESOLUTION_LIMIT')
        beta=candidate
    iv._fail('ITERATED_GMM_NOT_CONVERGED')


def cue_gmm(q,*,starting=None,max_iterations=100,tolerance='0.000000001',max_backtracks=30):
    """Damped analytic Newton CUE with exact negative-curvature escape.

    Exact Armijo decrease is checked after binary64 rounding. Success requires
    both scaled gradient and Newton predictor residual <= tolerance, and exact
    positive-definite local Hessian. No global optimum or parameter error
    certificate follows. Limits: 1..200 steps, 1..40 backtracks, tolerance<=.01.
    """
    checkpoint()
    owned=iv._snapshot(dict(study=q,starting=starting,max_iterations=max_iterations,tolerance=tolerance,
                           max_backtracks=max_backtracks),limit=196608)
    q=owned.pop('study'); tolerance_exact=_options(owned)
    if type(max_backtracks) is not int or not 1<=max_backtracks<=40: iv._fail('CUE_BACKTRACK_LIMIT_INVALID')
    code=_initial_code(); data=_prepare(q)
    beta=_binary_point(data['first_beta'] if owned['starting'] is None else _numbers(owned['starting'],len(data['x'][0])))
    trace=[]
    for iteration in range(max_iterations+1):
        checkpoint(); state=_evaluate(data,beta,derivatives=True)
        positive=state['hessian_status']=='PD'
        curvature=state['hessian'] if positive else state['metric']
        inverse=iv._inverse_small(curvature,failure='CUE_DIRECTION_SINGULAR')
        direction=[-iv._dot(row,state['gradient']) for row in inverse]
        predictor=_predictor_residual(data,beta,direction)
        if positive and state['scaled_gradient_squared']<=tolerance_exact**2 and predictor<=tolerance_exact:
            return _finish(q,data,state,'CONTINUOUSLY_UPDATED_LINEAR_GMM',owned,code,iterations=iteration,iteration_trace=trace,
                newton_predictor_residual_exact=str(predictor),convergence_scope='LOCAL_POSITIVE_CURVATURE_STATIONARITY_NOT_GLOBAL_OR_PARAMETER_ERROR_CERTIFICATE')
        if iteration==max_iterations: iv._fail('CUE_NOT_CONVERGED')
        kind='NEWTON' if positive else 'POSITIVE_GMM_METRIC'
        negative=_negative_curvature(state['hessian']) if state['hessian_status']=='INDEFINITE' else None
        if negative is not None:
            value=iv._dot(negative,[iv._dot(row,negative) for row in state['hessian']])
            if value>=0: iv._fail('CUE_NEGATIVE_CURVATURE_IDENTITY_FAILED')
            scale=max(abs(iv._dot(row,negative)) for row in data['x'])
            if scale==0: iv._fail('CUE_NEGATIVE_CURVATURE_DESIGN_FAILED')
            sign=-1 if iv._dot(state['gradient'],negative)>0 else 1
            direction=[sign*v/scale for v in negative]; kind='NEGATIVE_CURVATURE'
        elif not any(state['gradient']): iv._fail('CUE_NONMINIMUM_STATIONARY_POINT')
        for backtrack in range(max_backtracks):
            checkpoint(); step=F(1,2**backtrack)
            candidate=_binary_point([b+step*d for b,d in zip(beta,direction)])
            actual=[v-w for v,w in zip(candidate,beta)]
            slope=iv._dot(state['gradient'],actual)
            if slope>0 or (slope==0 and kind!='NEGATIVE_CURVATURE'): continue
            try: trial=_evaluate(data,candidate)
            except ValueError as error:
                if str(error)!='IV_GMM_MOMENT_COVARIANCE_SINGULAR': raise
                continue
            if trial['criterion']<state['criterion'] and trial['criterion']<=state['criterion']+slope/F(10000):
                trace.append(dict(iteration=iteration+1,criterion=_diagnostic(trial['criterion']),step_fraction=float(step),
                                  direction=kind,backtracks=backtrack))
                beta=candidate
                break
        else: iv._fail('CUE_LINE_SEARCH_FAILED')
    iv._fail('CUE_NOT_CONVERGED')
