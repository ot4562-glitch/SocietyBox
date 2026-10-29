"""Independent Bernoulli logit and Poisson/PPML by pivoted-QR Fisher scoring.

Fixed dispersion one; unweighted observations with declared offsets. Robust
covariance uses response scores, not working-response OLS residuals. Exact
recession directions certify nonexistence only when actually found/verified;
the candidate search is not a complete separation LP solver. No regularizing
prior, response dropping, clipping, automatic weights or upstream execution.
"""
from fractions import Fraction as F
from pathlib import Path
import math
import re
import statistics
import sys

from . import instrumental_variables as iv
from .numeric_linear import _QR,_dot
from .execution_control import checkpoint
from .research_evidence import (EvidenceLedger,UnknownRegistry,make_evidence,make_obligation,evaluate_coverage)
from .workspace import canonical,digest

_FAMILIES=('BINOMIAL_LOGIT','POISSON_COUNT','POISSON_PSEUDO')


def _fail(reason): raise ValueError('GLM_'+reason)


def _text(v): return iv._text(v) and bool(v.strip())


def _number(v):
    try: return iv._num(v)
    except ValueError as error: _fail('NUMBER_INVALID:'+str(error))


def _validated(q):
    fields={'schema','family','observations','model','units','covariance','optimizer','source_declared_sha256','rights'}
    if type(q) is not dict or set(q)!=fields or q['schema']!='societybox-generalized-linear-model-v1': _fail('SCHEMA_INVALID')
    if type(q['family']) is not str or q['family'] not in _FAMILIES: _fail('FAMILY_UNSUPPORTED')
    if q['rights']!=dict(local_processing='PERMITTED',report_export='PERMITTED'): _fail('RIGHTS_REQUIRED')
    if type(q['source_declared_sha256']) is not str or not re.fullmatch('[a-f0-9]{64}',q['source_declared_sha256']): _fail('SOURCE_INVALID')
    model=q['model']
    if type(model) is not dict or set(model)!={'outcome','regressors','intercept'}: _fail('MODEL_INVALID')
    names=model['regressors']
    if (not _text(model['outcome']) or type(model['intercept']) is not bool or type(names) is not list or
            not 1<=len(names)+int(model['intercept'])<=8 or any(not _text(v) for v in names) or
            len(set(names+[model['outcome']]))!=len(names)+1 or '(Intercept)' in names): _fail('MODEL_NAMES_INVALID')
    if type(q['units']) is not dict or set(q['units'])!=set(names+[model['outcome']]) or any(not _text(v) for v in q['units'].values()):
        _fail('UNITS_REQUIRED')
    optimizer=q['optimizer']
    if (type(optimizer) is not dict or set(optimizer)!={'max_iterations','tolerance'} or
            type(optimizer['max_iterations']) is not int or not 1<=optimizer['max_iterations']<=200 or
            type(optimizer['tolerance']) not in (int,float) or not 1e-12<=optimizer['tolerance']<=1e-7):
        _fail('OPTIMIZER_INVALID')
    observations=q['observations']; p=len(names)+int(model['intercept'])
    if type(observations) is not list or not p<len(observations)<=512: _fail('OBSERVATION_DIMENSIONS_INVALID')
    exact_x=[]; exact_y=[]; offsets=[]; ids=[]
    for row in observations:
        if type(row) is not dict or set(row)!={'observation_id','response','regressors','offset'}: _fail('OBSERVATION_INVALID')
        if not _text(row['observation_id']) or row['observation_id'] in ids: _fail('OBSERVATION_ID_INVALID')
        if type(row['regressors']) is not dict or set(row['regressors'])!=set(names): _fail('REGRESSORS_REQUIRED')
        y=_number(row['response'])
        if q['family']=='BINOMIAL_LOGIT' and y not in (0,1): _fail('BINARY_RESPONSE_REQUIRED')
        if q['family']!='BINOMIAL_LOGIT' and y<0: _fail('NONNEGATIVE_RESPONSE_REQUIRED')
        if q['family']=='POISSON_COUNT' and y.denominator!=1: _fail('INTEGER_COUNTS_REQUIRED')
        exact_x.append([F(1)]*int(model['intercept'])+[_number(row['regressors'][v]) for v in names])
        exact_y.append(y); offsets.append(float(_number(row['offset']))); ids.append(row['observation_id'])
    cov=q['covariance']
    if (type(cov) is not dict or set(cov)!={'estimator','clusters'} or type(cov['estimator']) is not str or
            cov['estimator'] not in ('MODEL_BASED','HC0','HC1','CLUSTER_CR0','CLUSTER_CR1')): _fail('COVARIANCE_UNSUPPORTED')
    if cov['estimator'].startswith('CLUSTER_'):
        labels=cov['clusters']
        if type(labels) is not dict or set(labels)!=set(ids) or any(not _text(v) for v in labels.values()) or len(set(labels.values()))<2:
            _fail('CLUSTER_LABELS_INVALID')
    elif cov['clusters'] is not None: _fail('UNUSED_CLUSTER_LABELS')
    if len(canonical(q))>262144: _fail('INPUT_SIZE_LIMIT')
    return exact_x,exact_y,offsets,ids


def validate(q): _validated(q); return q


def _nullspace(rows,p):
    a=[list(row) for row in rows]; pivots=[]; r=0
    for j in range(p):
        checkpoint()
        index=next((i for i in range(r,len(a)) if a[i][j]),None)
        if index is None: continue
        a[r],a[index]=a[index],a[r]; scale=a[r][j]; a[r]=[v/scale for v in a[r]]
        for i in range(len(a)):
            if i!=r:
                factor=a[i][j]; a[i]=[v-factor*w for v,w in zip(a[i],a[r])]
        pivots.append(j); r+=1
        if r==len(a): break
    basis=[]
    for free in range(p):
        if free in pivots: continue
        vector=[F(0)]*p; vector[free]=F(1)
        for i,col in enumerate(pivots): vector[col]=-a[i][free]
        basis.append((free,vector))
    return basis


def _separation(exact_x,exact_y,family,proposals,basis):
    # With a full-rank positive-response design the Poisson recession nullspace
    # is {0}; no strict zero-response margin can exist. Avoid rechecking zero
    # vectors at every scoring step, without weakening any certificate test.
    if family!='BINOMIAL_LOGIT' and not basis: return None
    for proposal in proposals:
        if any(not math.isfinite(v) for v in proposal): continue
        raw=[F(str(v)) for v in proposal]; scale=max(map(abs,raw),default=F(0))
        candidates=[raw,[F(0) if abs(v)<=scale*F(1,10**10) else v.limit_denominator(10**6) for v in raw]]
        if family!='BINOMIAL_LOGIT':
            candidates=[[sum((vector[j]*candidate[free] for free,vector in basis),F(0)) for j in range(len(raw))] for candidate in candidates]
        for direction in candidates:
            values=[iv._dot(row,direction) for row in exact_x]
            if family=='BINOMIAL_LOGIT':
                margins=[(2*y-1)*v for y,v in zip(exact_y,values)]
                valid=all(v>=0 for v in margins) and any(v>0 for v in margins)
                kind='COMPLETE' if all(v>0 for v in margins) else 'QUASI_COMPLETE'
            else:
                margins=[-v for y,v in zip(exact_y,values) if y==0]
                valid=all(v==0 for y,v in zip(exact_y,values) if y>0) and all(v>=0 for v in margins) and any(v>0 for v in margins)
                kind='POISSON_RECESSION_DIRECTION'
            if valid:
                return dict(kind=kind,direction_exact=list(map(str,direction)),margins_exact=list(map(str,margins)),
                    scope='EXACT_VERIFIED_DIRECTION_CANDIDATE_SEARCH_NOT_COMPLETE_LP_SOLVER')
    return None


def _exp_minus_one_minus_x(x):
    if abs(x)>=1e-4: return math.expm1(x)-x
    term=x*x/2; total=term
    for k in range(3,15): term*=x/k; total+=term
    return total


def _evaluate(x,y,offset,beta,family):
    eta=[o+_dot(row,beta) for row,o in zip(x,offset)]; means=[]; variance=[]; residual=[]; objectives=[]
    for value,target in zip(eta,y):
        if not math.isfinite(value): _fail('NUMERICAL_LINEAR_PREDICTOR_RANGE')
        if family=='BINOMIAL_LOGIT':
            t=math.exp(-abs(value)); mean=(1/(1+t) if value>=0 else t/(1+t)); weight=t/(1+t)**2
            error=(target-1+t/(1+t)) if value>=0 else target-t/(1+t)
            signed=-value if target==1 else value
            objective=-(max(signed,0)+math.log1p(math.exp(-abs(signed))))
        else:
            if value>700 or value<-700: _fail('NUMERICAL_MEAN_OR_WEIGHT_RANGE')
            mean=math.exp(value); weight=mean; error=target-mean
            if target==0: objective=-mean
            else:
                delta=value-math.log(target)
                try: objective=-target*_exp_minus_one_minus_x(delta)
                except OverflowError as error: raise ValueError('GLM_NUMERICAL_OBJECTIVE_RANGE') from error
        if weight<=0 or not all(math.isfinite(v) for v in (mean,weight,error,objective)): _fail('NUMERICAL_MEAN_OR_WEIGHT_RANGE')
        means.append(mean); variance.append(weight); residual.append(error); objectives.append(objective)
    return eta,means,variance,residual,math.fsum(objectives)


def _covariance(q,x,y,means,residuals,information,ids,weights,predictor_step):
    n,p=len(x),len(x[0]); method=q['covariance']['estimator']; variance_error=[0.0]*p
    if method=='MODEL_BASED': return information,variance_error
    scores=[[v*error for v in row] for row,error in zip(x,residuals)]
    groups={}
    for i,score in enumerate(scores):
        label=q['covariance']['clusters'][ids[i]] if method.startswith('CLUSTER_') else ids[i]
        groups.setdefault(label,[]).append(i)
    influences=[]; bounds=[]
    for indices in groups.values():
        score=[math.fsum(scores[i][j] for i in indices) for j in range(p)]
        influences.append([_dot(row,score) for row in information])
        # A resolution estimate, not an interval certificate: include the local
        # remaining Newton mean change so nearly cancelled cluster scores do not
        # turn an unfinished last digit into apparently certain inference.
        errors=[math.fsum(abs(x[i][j])*(64*sys.float_info.epsilon*(abs(y[i])+abs(means[i]))+
                         weights[i]*math.expm1(2*abs(predictor_step[i]))) for i in indices) for j in range(p)]
        bounds.append([math.fsum(abs(v)*e for v,e in zip(row,errors)) for row in information])
    factor=n/(n-p) if method=='HC1' else (len(groups)/(len(groups)-1)*(n-1)/(n-p) if method=='CLUSTER_CR1' else 1.0)
    matrix=[[factor*math.fsum(row[i]*row[j] for row in influences) for j in range(p)] for i in range(p)]
    variance_error=[factor*math.fsum(2*abs(row[i])*bound[i]+bound[i]**2 for row,bound in zip(influences,bounds)) for i in range(p)]
    if any(not math.isfinite(v) for row in matrix for v in row): _fail('NUMERICAL_COVARIANCE_RANGE')
    return matrix,variance_error


def _delta_method(estimate,gradient,covariance,variance_error,parameter_status):
    """Scalar delta inference with a diagonally scaled covariance contraction.

    Scaling before squaring avoids overflow when a large gradient multiplies a
    small covariance. The resolution estimate is conservative local arithmetic
    bookkeeping, not an interval proof or a design-validity certification.
    """
    result=dict(estimate=estimate,gradient=gradient,standard_error=None,variance=None,z_statistic=None,
                p_value=None,confidence_interval_95=None,inference_status='MARGINAL_EFFECT_NUMERIC_RANGE')
    if estimate is None or not math.isfinite(estimate) or gradient is None or any(not math.isfinite(v) for v in gradient): return result
    result['inference_status']='COVARIANCE_NUMERICAL_RESOLUTION_LIMIT'
    if any(v!=0 and s!='CONDITIONAL_ASYMPTOTIC_REFERENCE' for v,s in zip(gradient,parameter_status)): return result
    try:
        diagonal=[math.sqrt(covariance[i][i]) for i in range(len(gradient))]
        amplitudes=[v*d for v,d in zip(gradient,diagonal)]; scale=max(map(abs,amplitudes))
        if not math.isfinite(scale): result['inference_status']='MARGINAL_EFFECT_UNCERTAINTY_NUMERIC_RANGE'; return result
        if scale==0: return result
        normalized=[v/scale for v in amplitudes]
        terms=[normalized[i]*normalized[j]*(covariance[i][j]/diagonal[i]/diagonal[j])
               for i in range(len(gradient)) for j in range(len(gradient)) if diagonal[i]>0 and diagonal[j]>0]
        variance=math.fsum(terms); rounding=64*sys.float_info.epsilon*math.fsum(map(abs,terms))
        error=math.fsum(abs(v)*math.sqrt(e)/d for v,e,d in zip(normalized,variance_error,diagonal) if d>0)
        resolution=rounding+2*math.sqrt(max(variance,0))*error+error*error
        if not math.isfinite(variance) or not math.isfinite(resolution): return result
        result.update(scaled_variance=variance,scaled_variance_resolution_estimate=resolution,standard_error_scale=scale)
        if variance<=resolution: return result
        se=scale*math.sqrt(variance)
        if not math.isfinite(se) or se<=0: result['inference_status']='MARGINAL_EFFECT_UNCERTAINTY_NUMERIC_RANGE'; return result
        result.update(standard_error=se,variance=se*se if math.isfinite(se*se) else None,
                      variance_numeric_status='FINITE' if math.isfinite(se*se) else 'OUTSIDE_BINARY64_RANGE')
        critical=statistics.NormalDist().inv_cdf(.975); interval=[estimate-critical*se,estimate+critical*se]
        if not all(math.isfinite(v) for v in interval) or not interval[0]<estimate<interval[1]:
            result['inference_status']='MARGINAL_EFFECT_INTERVAL_NUMERICAL_RESOLUTION_LIMIT'; return result
        z=estimate/se
        if not math.isfinite(z): result['inference_status']='MARGINAL_EFFECT_Z_NUMERIC_RANGE'; return result
        result.update(z_statistic=z,p_value=math.erfc(abs(z)/math.sqrt(2)),confidence_interval_95=interval,
                      inference_status='CONDITIONAL_ASYMPTOTIC_REFERENCE')
    except (ValueError,OverflowError,ZeroDivisionError):
        result['inference_status']='MARGINAL_EFFECT_UNCERTAINTY_NUMERIC_RANGE'
    return result


def _marginal_effect_inference(q,x,eta,weights,beta,covariance,variance_error,parameter_status):
    n=len(x); average=math.fsum(weights)/n
    # g''(eta) = g'(eta) * -tanh(eta/2) for logit, g'(eta) for log link.
    # tanh retains the sign and tail even when the fitted probability rounds to 1.
    ratios=[-math.tanh(v/2) for v in eta] if q['family']=='BINOMIAL_LOGIT' else [1.0]*n
    # Normalize weights before the signed design sum: its individual products
    # can overflow although their average is exactly zero. Reuse this vector
    # across all marginal-effect Jacobian rows.
    weight_scale=max(weights)
    normalized_second=[math.fsum((w/weight_scale)*ratio*row[k] for row,w,ratio in zip(x,weights,ratios))/n for k in range(len(beta))]
    effects={}
    for i,name in enumerate(q['model']['regressors'],int(q['model']['intercept'])):
        estimate=beta[i]*average; estimate=estimate if math.isfinite(estimate) else None
        try:
            gradient=[math.fsum([average if k==i else 0.0,
                       (beta[i]*normalized_second[k])*weight_scale]) for k in range(len(beta))]
            if any(not math.isfinite(v) for v in gradient): gradient=None
        except (ValueError,OverflowError): gradient=None
        effects[name]=_delta_method(estimate,gradient,covariance,variance_error,parameter_status)
    return dict(method='ANALYTIC_JACOBIAN_DELTA_METHOD',effects=effects,covariance_estimator=q['covariance']['estimator'],
                scope='FIXED_EMPIRICAL_DESIGN_AND_OFFSETS_COEFFICIENT_UNCERTAINTY_ONLY',
                variance_resolution_method='DIAGONAL_SCALED_ARITHMETIC_AND_PROPAGATED_LOCAL_ESTIMATE_NOT_INTERVAL_CERTIFIED',
                inference_reference='ASYMPTOTIC_STANDARD_NORMAL_ASSUMPTIONS_UNVERIFIED',
                unsupported=['RANDOM_REGRESSOR_DISTRIBUTION_UNCERTAINTY','DISCRETE_DUMMY_CHANGES','SECOND_ORDER_DELTA_METHOD'])


def _assurance(q,result):
    code=iv._code_receipt()
    for name in ('generalized_linear.py','numeric_linear.py'): code[name]=digest(Path(__file__).with_name(name).read_bytes())
    subject=digest(canonical(q)); scope=dict(family=q['family'],model=q['model'],units=q['units'],covariance=q['covariance'],
        observation_ids=[row['observation_id'] for row in q['observations']],optimizer=q['optimizer'])
    evidence=make_evidence(subject_sha256=subject,producer=dict(tool_id='econ.glm',version='1',code_sha256=digest(canonical(code))),
        operation=result['method'],authority='LOCAL_BINARY64_WITH_EXACT_SEPARATION_DIRECTION_CHECKS',
        epistemic_state='DERIVED',proof_kind='DECLARED_GLM_RESULT',scope=scope,
        coverage=dict(status='PARTIAL',unknown_dimensions=['conditional_mean','variance_dependence','finite_sample_inference',
                     'causal_identification','source_vintage_replication']),result=result,parent_evidence_ids=[],
        limitations=['Numerical convergence is not causal or model-validity certification.',
                     'An unfound separation certificate does not establish finite-MLE existence.',
                     'Offsets are held fixed; continuous derivatives and discrete changes have separate declared semantics.',
                     'Marginal-effect delta uncertainty conditions on the empirical design and offsets.'])
    ledger=EvidenceLedger([evidence]); registry=UnknownRegistry(ledger); unknowns=[]; obligations=[]
    for claim,question in (
        ('conditional_mean','Are the response family, conditional mean, offset and regressor exogeneity justified?'),
        ('variance_dependence','Are variance assumptions, independent observations or cluster independence justified?'),
        ('finite_sample_inference','Are asymptotic normal inference, cluster count and numerical resolution adequate?'),
        ('causal_identification','What independent design supports a causal interpretation of coefficients or partial derivatives?'),
        ('source_vintage_replication','Are source rights, units, historical information and independent replication verified?')):
        unknown=registry.register(question,scope,'EXTERNAL_SCIENTIFIC_VALIDATION','GLM_DESIGN_CHECK'); unknowns.append(unknown)
        obligations.append(make_obligation(claim_id=claim,subject_sha256=subject,scope=scope,required_authority='EXTERNAL_SCIENTIFIC_VALIDATION',
            required_proof_kind='GLM_DESIGN_CHECK',verifier_id='econ.glm',dependencies=[],unknown_ids=[unknown['unknown_id']],blocking_reasons=[],next_probe=question))
    return dict(input_sha256=subject,code_sha256=code,evidence=[evidence],unknowns=unknowns,obligations=obligations,
                coverage=evaluate_coverage(obligations,ledger,registry))


def fit(q):
    q=iv._snapshot(q,limit=262144)
    exact_x,exact_y,offset,ids=_validated(q); x=[[float(v) for v in row] for row in exact_x]; y=list(map(float,exact_y))
    n,p=len(y),len(x[0]); _QR(x)  # Distinguish numerical design rank failure before nonlinear iteration.
    beta=[0.0]*p; family=q['family']; trace=[]; certificate=None; reason=None; status='NOT_CONVERGED'
    if q['model']['intercept']:
        if family=='BINOMIAL_LOGIT': beta[0]=math.log((math.fsum(y)+.5)/(n-math.fsum(y)+.5))
        elif math.fsum(y)>0:
            largest=max(offset); log_exposure=largest+math.log(math.fsum(math.exp(v-largest) for v in offset))
            beta[0]=math.log(math.fsum(y))-log_exposure
    basis=_nullspace([row for row,target in zip(exact_x,exact_y) if target>0],p) if family!='BINOMIAL_LOGIT' else None
    axes=[[float(i==j)*sign for i in range(p)] for j in range(p) for sign in (1,-1)]
    initial=axes if basis is None else [[float(v)*sign for v in vector] for _,vector in basis for sign in (1,-1)]
    certificate=_separation(exact_x,exact_y,family,initial+[beta],basis)
    score_norm=None; last_step=None; state=None; factor=None; diagnostic_iterate=None
    for iteration in range(1,q['optimizer']['max_iterations']+1):
        checkpoint()
        if certificate is not None: status='NO_FINITE_MLE_CERTIFIED'; break
        try:
            state=_evaluate(x,y,offset,beta,family); eta,means,weights,residuals,objective=state
            roots=list(map(math.sqrt,weights)); factor=_QR([[v*root for v in row] for row,root in zip(x,roots)])
            direction=factor.solve([e/root for e,root in zip(residuals,roots)])
            # Form normalized scores directly: X'u may overflow even though
            # its QR-scaled value and the Newton direction are representable.
            scaled_score=[math.fsum((row[j]/factor.scales[j])*e for row,e in zip(x,residuals)) for j in range(p)]
            score_norm=max(map(abs,scaled_score)); predictor_step=[_dot(row,direction) for row in x]
            last_step=max(map(abs,predictor_step))/(1+max(map(abs,eta)))
            diagnostic_iterate=list(beta)
            certificate=_separation(exact_x,exact_y,family,[direction,beta],basis)
            if certificate is not None: status='NO_FINITE_MLE_CERTIFIED'; break
            if score_norm<=q['optimizer']['tolerance'] and last_step<=q['optimizer']['tolerance']:
                status='CONVERGED'; break
            ascent=_dot(residuals,predictor_step)
            if ascent<=0: _fail('NUMERICAL_NEWTON_DIRECTION_FAILED')
            step=1.0; allowance=8*math.ulp(objective)
            for _ in range(60):
                candidate=[b+step*d for b,d in zip(beta,direction)]
                try: proposed=_evaluate(x,y,offset,candidate,family)[-1]
                except ValueError as error:
                    if not str(error).startswith('GLM_NUMERICAL_'): raise
                    proposed=-math.inf
                if proposed>=objective+1e-4*step*ascent-allowance: break
                step/=2
            else: _fail('NUMERICAL_LINE_SEARCH_FAILED')
            trace.append(dict(iteration=iteration,objective=proposed,step_fraction=step,roundoff_allowance=allowance))
            beta=candidate
        except ValueError as error:
            if not str(error).startswith(('GLM_NUMERICAL_','NUMERIC_LINEAR_')): raise
            status='NUMERICAL_FAILURE'; reason=str(error); score_norm=None; last_step=None; diagnostic_iterate=None; break
    result=dict(schema='societybox-glm-result-v1',method='PIVOTED_QR_FISHER_SCORING',family=family,status=status,reason=reason,
        implementation_status='INDEPENDENT_IMPLEMENTED',native_execution='NOT_RUN',native_parity='NOT_RUN',ui_status='NOT_CONNECTED',
        parameter_order=(['(Intercept)'] if q['model']['intercept'] else [])+q['model']['regressors'],observations=n,residual_df=n-p,
        coefficients=None,covariance=None,standard_errors=None,average_marginal_effects=None,marginal_effect_inference=None,aic=None,
        separation_certificate=certificate,finite_mle_existence='CERTIFIED_NONEXISTENT' if certificate else 'NOT_FORMALLY_CERTIFIED',
        iterations=len(trace),iteration_trace=trace,scaled_score_norm=score_norm,last_scaled_predictor_step=last_step,
        last_iterate=list(beta),diagnostic_iterate=diagnostic_iterate,inference_reference='ASYMPTOTIC_STANDARD_NORMAL_ASSUMPTIONS_UNVERIFIED',
        covariance_estimator=q['covariance']['estimator'],dispersion='FIXED_ONE',structural_causal_identification='NOT_ESTABLISHED')
    if status=='CONVERGED':
        eta,means,weights,residuals,objective=state; information=factor.inverse_information()
        covariance,variance_error=_covariance(q,x,y,means,residuals,information,ids,weights,predictor_step)
        se=[math.sqrt(covariance[i][i]) for i in range(p)]; critical=statistics.NormalDist().inv_cdf(.975)
        z=[]; probabilities=[]; intervals=[]; inference_status=[]
        for i,(b,s) in enumerate(zip(beta,se)):
            usable=s>0 and covariance[i][i]>variance_error[i]
            interval=[b-critical*s,b+critical*s] if usable else None
            resolved=interval is not None and all(math.isfinite(v) for v in interval) and interval[0]<b<interval[1]
            inference_status.append('COVARIANCE_NUMERICAL_RESOLUTION_LIMIT' if not usable else
                                    'CONDITIONAL_ASYMPTOTIC_REFERENCE' if resolved else 'PARAMETER_INTERVAL_NUMERICAL_RESOLUTION_LIMIT')
            usable=usable and resolved
            z.append(b/s if usable else None); probabilities.append(math.erfc(abs(b/s)/math.sqrt(2)) if usable else None)
            intervals.append(interval if usable else None)
        saturated=(0.0 if family=='BINOMIAL_LOGIT' else math.fsum(v*math.log(v)-v-math.lgamma(v+1) if v>0 else 0.0 for v in y))
        likelihood=objective+saturated
        marginal=_marginal_effect_inference(q,x,eta,weights,beta,covariance,variance_error,inference_status)
        result.update(coefficients=list(beta),covariance=covariance,standard_errors=se,z_statistics=z,p_values=probabilities,
            confidence_intervals_95=intervals,parameter_inference_status=inference_status,variance_resolution_estimate=variance_error,
            variance_resolution_method='ARITHMETIC_AND_LOCAL_NEWTON_MEAN_CHANGE_NOT_INTERVAL_CERTIFIED',
            fitted_mean=means,response_residuals=residuals,deviance=-2*objective,
            log_likelihood=None if family=='POISSON_PSEUDO' else likelihood,quasi_objective=objective,
            aic=None if family=='POISSON_PSEUDO' else 2*p-2*likelihood,
            average_marginal_effects={name:effect['estimate'] for name,effect in marginal['effects'].items()},marginal_effect_inference=marginal,
            marginal_effect_semantics='CONTINUOUS_PARTIAL_DERIVATIVE_HOLDING_OFFSET_FIXED_NOT_DISCRETE_CHANGE_OR_CAUSAL_EFFECT',
            covariance_information_inverse=information,information_column_permutation=list(factor.permutation),
            numerical_arithmetic='BINARY64_NOT_INTERVAL_CERTIFIED')
    result.update(_assurance(q,result)); return result


def _inverse_mean(eta,family):
    if family=='BINOMIAL_LOGIT':
        t=math.exp(-abs(eta)); return 1/(1+t) if eta>=0 else t/(1+t)
    if not -700<=eta<=700: return None
    return math.exp(eta)


def _binary_change_at(eta_zero,coefficient,family):
    """Two conditional means, stable difference, and their link derivatives."""
    eta_one=eta_zero+coefficient
    if not math.isfinite(eta_zero) or not math.isfinite(eta_one): _fail('DISCRETE_CHANGE_MEAN_NUMERIC_RANGE')
    if family=='BINOMIAL_LOGIT':
        t0=math.exp(-abs(eta_zero));t1=math.exp(-abs(eta_one))
        p0=1/(1+t0) if eta_zero>=0 else t0/(1+t0);q0=t0/(1+t0) if eta_zero>=0 else 1/(1+t0)
        p1=1/(1+t1) if eta_one>=0 else t1/(1+t1);q1=t1/(1+t1) if eta_one>=0 else 1/(1+t1)
        difference=p1*q0*(-math.expm1(-coefficient)) if coefficient>=0 else p0*q1*math.expm1(coefficient)
        weight_one=t1/(1+t1)**2
        weight_difference=difference*(-.5*(math.tanh(eta_zero/2)+math.tanh(eta_one/2)))
    else:
        if not -700<=eta_zero<=700 or not -700<=eta_one<=700: _fail('DISCRETE_CHANGE_MEAN_NUMERIC_RANGE')
        p0=math.exp(eta_zero);p1=math.exp(eta_one)
        difference=p1*(-math.expm1(-coefficient)) if coefficient>=0 else p0*math.expm1(coefficient)
        weight_one=p1;weight_difference=difference
    return p0,p1,difference,weight_one,weight_difference


def discrete_changes(q,regressors):
    """Average 0-to-1 changes of declared binary columns, holding others fixed.

    This model contrast requires no causal interpretation or overlap claim.
    The selected coefficient covariance supplies fixed-design delta uncertainty.
    Every requested column must actually contain both binary values in training.
    """
    q=iv._snapshot(q,limit=262144)
    if type(regressors) is list:regressors=iv._snapshot(regressors,limit=8192)
    exact_x,_,offset,_=_validated(q);names=q['model']['regressors']
    if (type(regressors) is not list or not regressors or any(type(v) is not str or v not in names for v in regressors)
            or len(set(regressors))!=len(regressors)): _fail('DISCRETE_CHANGE_REGRESSORS_INVALID')
    columns=[names.index(name)+int(q['model']['intercept']) for name in regressors]
    if any({row[j] for row in exact_x}!={0,1} for j in columns): _fail('BINARY_REGRESSOR_WITH_BOTH_VALUES_REQUIRED')
    fitted=fit(q)
    if fitted['status']!='CONVERGED': _fail('DISCRETE_CHANGE_REQUIRES_CONVERGED_FIT')
    beta=fitted['coefficients'];x=[[float(v) for v in row] for row in exact_x];n=len(x);effects={}
    for name,j in zip(regressors,columns):
        checkpoint();design_zero=[list(row) for row in x]
        for row in design_zero: row[j]=0.0
        points=[_binary_change_at(o+_dot(row,beta),beta[j],q['family']) for row,o in zip(design_zero,offset)]
        estimate=math.fsum(v[2]/n for v in points)
        scale=max(max(v[3],abs(v[4])) for v in points)
        try:
            gradient=([0.0]*len(beta) if scale==0 else
                [scale*math.fsum(((v[4]/scale)*row[k]+(v[3]/scale if k==j else 0.0))/n
                      for row,v in zip(design_zero,points)) for k in range(len(beta))])
            if any(not math.isfinite(v) for v in gradient): gradient=None
        except (ValueError,OverflowError): gradient=None
        effect=_delta_method(estimate,gradient,fitted['covariance'],fitted['variance_resolution_estimate'],fitted['parameter_inference_status'])
        effect.update(mean_at_zero=math.fsum(v[0]/n for v in points),mean_at_one=math.fsum(v[1]/n for v in points))
        effects[name]=effect
    result=dict(schema='societybox-glm-discrete-changes-v1',method='AVERAGE_BINARY_DISCRETE_CHANGES_DELTA_METHOD',
        family=q['family'],implementation_status='INDEPENDENT_IMPLEMENTED',native_execution='NOT_RUN',native_parity='NOT_RUN',
        ui_status='NOT_CONNECTED',effects=effects,fit=fitted,fit_input_sha256=fitted['input_sha256'],
        scope='FIXED_EMPIRICAL_DESIGN_AND_OFFSETS_COEFFICIENT_UNCERTAINTY_ONLY',
        effect_semantics='AVERAGE_CONDITIONAL_MEAN_AT_ONE_MINUS_ZERO_HOLD_OTHER_COLUMNS_AND_OFFSETS_FIXED',
        limitations=['No automatic formula/interaction reconstruction; other declared columns are held fixed.',
                     'Counterfactual support, causal identification and source validity are not established.',
                     'Stable differences can retain changes below the rounding resolution of separately reported means.',
                     'Normal delta intervals are not clipped to response-domain bounds.'],
        numerical_arithmetic='BINARY64_NOT_INTERVAL_CERTIFIED',structural_causal_identification='NOT_ESTABLISHED')
    result.update(_assurance(dict(q,discrete_change_regressors=regressors),result));return result


def predict(q,observations):
    """Conditional mean and delta/link-scale confidence intervals at new designs.

    Intervals reflect the selected coefficient covariance only, not the
    variability of a future response. Binary64 saturation is labelled.
    """
    q=iv._snapshot(q,limit=262144)
    if type(observations) is list:observations=iv._snapshot(observations,limit=131072)
    fitted=fit(q)
    if fitted['status']!='CONVERGED': _fail('PREDICTION_REQUIRES_CONVERGED_FIT')
    if type(observations) is not list or not 1<=len(observations)<=512: _fail('PREDICTION_OBSERVATIONS_INVALID')
    names=q['model']['regressors']; seen=set(); points=[]; critical=statistics.NormalDist().inv_cdf(.975)
    for row in observations:
        if (type(row) is not dict or set(row)!={'observation_id','regressors','offset'} or
                not _text(row['observation_id']) or row['observation_id'] in seen or
                type(row['regressors']) is not dict or set(row['regressors'])!=set(names)):
            _fail('PREDICTION_ROW_INVALID')
        seen.add(row['observation_id'])
        x=[1.0]*int(q['model']['intercept'])+[float(_number(row['regressors'][v])) for v in names]
        eta=float(_number(row['offset']))+_dot(x,fitted['coefficients']); mean=_inverse_mean(eta,q['family'])
        if mean is None: _fail('PREDICTION_MEAN_NUMERIC_RANGE')
        if q['family']=='BINOMIAL_LOGIT':
            t=math.exp(-abs(eta)); derivative=t/(1+t)**2
            log_mean=-(max(-eta,0)+math.log1p(t)); log_complement=-(max(eta,0)+math.log1p(t))
        else: derivative=mean; log_mean=eta; log_complement=None
        try: variance=math.fsum(x[i]*x[j]*fitted['covariance'][i][j] for i in range(len(x)) for j in range(len(x)))
        except (OverflowError,ValueError): variance=None
        usable=(variance is not None and math.isfinite(variance) and variance>0 and derivative>0 and
                all(v==0 or status=='CONDITIONAL_ASYMPTOTIC_REFERENCE' for v,status in zip(x,fitted['parameter_inference_status'])))
        se=math.sqrt(variance) if usable else None
        interval=None if se is None else [_inverse_mean(eta-critical*se,q['family']),_inverse_mean(eta+critical*se,q['family'])]
        mean_se=None if se is None else derivative*se
        if mean_se is not None and not math.isfinite(mean_se): mean_se=None
        interval_range=interval is not None and any(v is None for v in interval)
        interval_resolved=interval is not None and not interval_range and interval[0]<mean<interval[1]
        interval_resolution=interval is not None and not interval_range and not interval_resolved
        if interval_resolution: interval=None
        points.append(dict(observation_id=row['observation_id'],linear_predictor=eta,mean=mean,log_mean=log_mean,
            log_one_minus_mean=log_complement,linear_predictor_standard_error=se,mean_standard_error=mean_se,
            mean_confidence_interval_95=interval,
            inference_status='COVARIANCE_OR_MEAN_NUMERIC_RESOLUTION_LIMIT' if not usable else
                             'MEAN_UNCERTAINTY_NUMERIC_RANGE' if mean_se is None else
                             'RESPONSE_INTERVAL_NUMERIC_RANGE' if interval_range else
                             'MEAN_INTERVAL_NUMERICAL_RESOLUTION_LIMIT' if interval_resolution else 'CONDITIONAL_ASYMPTOTIC_REFERENCE'))
    if len(canonical(observations))>131072: _fail('PREDICTION_INPUT_SIZE_LIMIT')
    result=dict(schema='societybox-glm-prediction-v1',method='CONDITIONAL_GLM_MEAN_PREDICTION',family=q['family'],
        implementation_status='INDEPENDENT_IMPLEMENTED',native_execution='NOT_RUN',fit_input_sha256=fitted['input_sha256'],
        predictions=points,fit=fitted,interval_scope='CONDITIONAL_MEAN_CONFIDENCE_NOT_NEW_OBSERVATION_PREDICTION',
        structural_causal_identification='NOT_ESTABLISHED',numerical_arithmetic='BINARY64_NOT_INTERVAL_CERTIFIED')
    result.update(_assurance(dict(q,prediction_rows=observations),result)); return result
