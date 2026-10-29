"""Bounded LIML and Fuller estimation with exact rational error certificates.

No external eigensolver or observation-square projection. The smallest
generalized eigenvalue of (W'M_exog W, W'M_Z W) is bracketed using exact
positive-definiteness tests. Returned rational values solve k-class at the
declared bracket midpoint, not at an allegedly exact irrational eigenvalue.
Neumann perturbation bounds enclose the true coefficients and the declared
covariance convention over the entire bracket, including binary64 rounding.
"""
from collections import Counter
from fractions import Fraction as F
from pathlib import Path
import math

from . import instrumental_variables as iv
from .execution_control import checkpoint
from .workspace import digest


def _positive_status(matrix):
    """Exact symmetric Schur test: PD, PSD (singular), or INDEFINITE."""
    work=[list(row) for row in matrix]; singular=False
    for j in range(len(work)):
        checkpoint()
        pivot=work[j][j]
        if pivot<0: return 'INDEFINITE'
        if pivot==0:
            if any(work[i][j] for i in range(j+1,len(work))): return 'INDEFINITE'
            singular=True
            continue
        for i in range(j+1,len(work)):
            for k in range(i,len(work)):
                value=work[i][k]-work[i][j]*work[k][j]/pivot
                work[i][k]=value; work[k][i]=value
    return 'PSD' if singular else 'PD'


def _norm(matrix):
    return max(sum(map(abs,row),F(0)) for row in matrix)


def _subtract(a,b,scale):
    return [[v-scale*w for v,w in zip(row,other)] for row,other in zip(a,b)]


def _prepare(q):
    y,x,z,exog,endog=iv._validated(q)
    zz=iv._cross(z,z); zx=iv._cross(z,x)
    zz_inverse=iv._inverse_small(zz,failure='INSTRUMENT_RANK_DEFICIENT')
    xhat=iv._multiply(z,iv._multiply(zz_inverse,zx))
    iv._inverse_small(iv._cross(xhat,xhat),failure='UNDERIDENTIFIED_PROJECTED_DESIGN')
    xres=[[v-w for v,w in zip(row,projected)] for row,projected in zip(x,xhat)]
    _,yres,_=iv._ols(z,y,'INSTRUMENT_RANK_DEFICIENT')
    w=[[v]+row for v,row in zip(y,endog)]
    residuals=[iv._ols(exog,list(column),'EXOGENOUS_RANK_DEFICIENT')[1] for column in zip(*w)]
    a=iv._cross([list(row) for row in zip(*residuals)],[list(row) for row in zip(*residuals)])
    ez=[[v]+row[len(exog[0]):] for v,row in zip(yres,xres)]
    b=iv._cross(ez,ez)
    if _positive_status(b)!='PD': iv._fail('LIML_REDUCED_FORM_COVARIANCE_SINGULAR')
    return dict(y=y,x=x,z=z,xhat=xhat,a=a,b=b,xx=iv._cross(x,x),hxx=iv._cross(xres,xres),
                xy=[iv._dot(col,y) for col in zip(*x)],hxy=[iv._dot(col,yres) for col in zip(*xres)])


def _fit_kappa(data,kappa,covariance):
    x,y,xhat=data['x'],data['y'],data['xhat']; n,k=len(y),len(x[0])
    information=_subtract(data['xx'],data['hxx'],kappa)
    if _positive_status(information)!='PD': iv._fail('LIML_KCLASS_INFORMATION_NOT_POSITIVE_DEFINITE')
    bread=iv._inverse_small(information,failure='LIML_KCLASS_INFORMATION_SINGULAR')
    rhs=[v-kappa*w for v,w in zip(data['xy'],data['hxy'])]
    beta=[iv._dot(row,rhs) for row in bread]
    errors=[v-iv._dot(row,beta) for v,row in zip(y,x)]
    normal=[iv._dot(row,beta)-v for row,v in zip(information,rhs)]
    if any(normal): iv._fail('LIML_KCLASS_NORMAL_IDENTITY_FAILED')
    method=covariance['estimator']
    if method=='CLASSICAL':
        meat=None; sigma2=iv._dot(errors,errors)/(n-k)
        variance=[[sigma2*v for v in row] for row in bread]
    else:
        meat=iv._moment_meat(xhat,errors,method,covariance['clusters'],k,covariance.get('bandwidth'))
        variance=iv._multiply(iv._multiply(bread,meat),bread)
    if any(variance[i][i]<0 for i in range(k)): iv._fail('LIML_COVARIANCE_NEGATIVE')
    return dict(beta=beta,errors=errors,bread=bread,variance=variance,meat=meat,normal=normal)


def _error_bounds(data,fitted,radius,covariance):
    """Uniform infinity-norm perturbation bounds; no derivative approximation."""
    beta,bread,errors=fitted['beta'],fitted['bread'],fitted['errors']
    bh=iv._multiply(bread,data['hxx']); rho=radius*_norm(bh)
    if rho>=1: return None
    direction=[iv._dot(row,beta)-v for row,v in zip(data['hxx'],data['hxy'])]
    sensitivity=[iv._dot(row,direction) for row in bread]
    beta_error=radius*max(map(abs,sensitivity))/(1-rho)
    bread_norm=_norm(bread); bread_error=rho*bread_norm/(1-rho)
    residual_error=[sum(map(abs,row),F(0))*beta_error for row in data['x']]
    n,k=len(errors),len(beta); method=covariance['estimator']
    if method=='CLASSICAL':
        sigma2=iv._dot(errors,errors)/(n-k)
        sigma_error=sum((2*abs(u)*e+e*e for u,e in zip(errors,residual_error)),F(0))/(n-k)
        variance_error=sigma2*bread_error+(bread_norm+bread_error)*sigma_error
    else:
        scores=[[v*u for v in row] for row,u in zip(data['xhat'],errors)]
        score_error=[[abs(v)*e for v in row] for row,e in zip(data['xhat'],residual_error)]
        row_norm=_norm(scores); transpose_norm=_norm([list(row) for row in zip(*scores)])
        error_norm=_norm(score_error); error_transpose=_norm([list(row) for row in zip(*score_error)])
        kernel_norm=F(1); correction=F(n,n-k) if method=='HC1' else F(1)
        if method.startswith('CLUSTER_'):
            counts=Counter(covariance['clusters']); kernel_norm=F(max(counts.values()))
            if method=='CLUSTER_CR1': correction=F(len(counts),len(counts)-1)*F(n-1,n-k)
        elif method=='HAC_BARTLETT': kernel_norm=F(covariance['bandwidth']+1)
        meat_error=correction*kernel_norm*(error_transpose*row_norm+transpose_norm*error_norm+error_transpose*error_norm)
        variance_error=((2*bread_norm*bread_error+bread_error*bread_error)*_norm(fitted['meat'])
                        +(bread_norm+bread_error)**2*meat_error)
    return beta_error,variance_error,rho


def _as_float(values):
    try: converted=list(map(float,values))
    except OverflowError as error: raise ValueError('IV_LIML_BINARY64_RANGE') from error
    if any(not math.isfinite(v) for v in converted): iv._fail('LIML_BINARY64_RANGE')
    rounding=max((abs(F(v)-exact) for v,exact in zip(converted,values)),default=F(0))
    return converted,rounding


def liml(q,*,fuller=0,tolerance='0.000000000001',max_bisections=96):
    """LIML/Fuller with coefficient and covariance absolute-error certificates.

    Same small unweighted design/rights/covariance contract as iv.calculate.
    Reduced-form W'M_Z W must be positive definite. Fuller >=0 subtracts
    fuller/(n-total_instruments) from LIML kappa. Exact Schur bisection is
    limited to 1..160 refinements. Positive tolerance <=.01 uses the existing
    bounded input numbers. For each output array, the certified uniform error
    must be <= tolerance*(1+largest absolute midpoint entry).

    Covariance follows the IV k-class convention: adjusted k-class bread,
    projected Xhat residual scores for HC/cluster/HAC; classical SSE/(n-K)
    times that bread. These are conditional asymptotic reference covariances,
    not weak-IV-robust uncertainty or an exogeneity/likelihood certificate.
    Exact rational fields are evaluated at the declared kappa midpoint;
    their approximation errors and binary64 rounding are reported separately.
    """
    checkpoint()
    owned=iv._snapshot(dict(study=q,fuller=fuller,tolerance=tolerance,max_bisections=max_bisections),limit=196608)
    q=owned['study']
    if type(max_bisections) is not int or not 1<=max_bisections<=160: iv._fail('LIML_BISECTION_LIMIT_INVALID')
    try: alpha=iv._num(fuller); target=iv._num(tolerance)
    except ValueError as error: iv._fail('LIML_OPTIONS_INVALID:'+str(error))
    if alpha<0 or not 0<target<=F(1,100): iv._fail('LIML_OPTIONS_INVALID')
    initial_code=iv._code_receipt()
    initial_code[Path(__file__).name]=digest(Path(__file__).read_bytes())
    data=_prepare(q); a,b=data['a'],data['b']
    status=_positive_status(_subtract(a,b,F(1)))
    if status=='INDEFINITE': iv._fail('LIML_PROJECTION_ORDER_IDENTITY_FAILED')
    lower=F(1); upper=F(1) if status=='PSD' else min(a[i][i]/b[i][i] for i in range(len(a)))
    n,k,l=len(data['y']),len(data['x'][0]),len(data['z'][0]); adjustment=alpha/(n-l)
    solution=None
    for iteration in range(max_bisections+1):
        checkpoint()
        if upper-lower<=target or iteration==max_bisections:
            midpoint=(lower+upper)/2-adjustment; radius=(upper-lower)/2
            try: fitted=_fit_kappa(data,midpoint,q['covariance'])
            except ValueError as error:
                if str(error) not in ('IV_LIML_KCLASS_INFORMATION_NOT_POSITIVE_DEFINITE',
                                      'IV_LIML_KCLASS_INFORMATION_SINGULAR'): raise
                fitted=None
            bounds=_error_bounds(data,fitted,radius,q['covariance']) if fitted else None
            if bounds is not None:
                beta_error,variance_error,rho=bounds
                coefficients,beta_rounding=_as_float(fitted['beta'])
                flat=[v for row in fitted['variance'] for v in row]
                covariance_flat,variance_rounding=_as_float(flat)
                if (beta_error+beta_rounding<=target*(1+max(map(abs,fitted['beta'])))
                        and variance_error+variance_rounding<=target*(1+max(map(abs,flat)))):
                    solution=(midpoint,fitted,bounds,coefficients,covariance_flat,beta_rounding,variance_rounding)
                    break
        if iteration==max_bisections: iv._fail('LIML_ACCURACY_NOT_CERTIFIED')
        middle=(lower+upper)/2
        status=_positive_status(_subtract(a,b,middle))
        if status=='PSD': lower=upper=middle
        elif status=='PD': lower=middle
        else: upper=middle
    midpoint,fitted,bounds,coefficients,covariance_flat,beta_rounding,variance_rounding=solution
    beta_error,variance_error,rho=bounds
    options=dict(fuller_exact=str(alpha),tolerance_exact=str(target),max_bisections=max_bisections,
                 declared_fuller=owned['fuller'],declared_tolerance=owned['tolerance'])
    result=dict(schema='societybox-liml-result-v1',method='FULLER_MODIFIED_LIML' if alpha else 'LIMITED_INFORMATION_MAXIMUM_LIKELIHOOD',
        status='COMPUTED_ASSUMPTIONS_UNVERIFIED',implementation_status='INDEPENDENT_IMPLEMENTED',
        native_execution='NOT_RUN',native_parity='NOT_RUN',ui_status='NOT_CONNECTED',identification='IDENTIFICATION_NOT_ESTABLISHED',
        arithmetic='EXACT_RATIONAL_KAPPA_BRACKET_AND_PERTURBATION_CERTIFICATE',
        parameter_order=q['model']['exogenous']+q['model']['endogenous'],coefficients=coefficients,
        coefficients_at_kappa_exact=list(map(str,fitted['beta'])),
        covariance=[covariance_flat[i*k:(i+1)*k] for i in range(k)],
        covariance_at_kappa_exact=[list(map(str,row)) for row in fitted['variance']],
        structural_residuals_at_kappa_exact=list(map(str,fitted['errors'])),
        k_class_normal_residual_exact=list(map(str,fitted['normal'])),
        coefficient_max_abs_error_bound_exact=str(beta_error+beta_rounding),
        covariance_max_abs_error_bound_exact=str(variance_error+variance_rounding),
        coefficient_rational_error_bound_exact=str(beta_error),covariance_rational_error_bound_exact=str(variance_error),
        coefficient_binary64_rounding_bound_exact=str(beta_rounding),covariance_binary64_rounding_bound_exact=str(variance_rounding),
        liml_kappa_interval_exact=[str(lower),str(upper)],adjusted_kappa_interval_exact=[str(lower-adjustment),str(upper-adjustment)],
        kappa_midpoint_exact=str(midpoint),fuller_adjustment_exact=str(adjustment),options=options,bisections=iteration,
        perturbation_neumann_radius_exact=str(rho),generalized_eigen_numerator_exact=[list(map(str,row)) for row in a],
        generalized_eigen_denominator_exact=[list(map(str,row)) for row in b],
        covariance_estimator=q['covariance']['estimator'],covariance_specification=q['covariance'],
        covariance_convention='KCLASS_ADJUSTED_BREAD_PROJECTED_SCORES_CLASSICAL_DF_N_MINUS_K',
        weak_instrument_robust=False,observations=n,residual_df=n-k,total_instruments=l,
        source_authentication='NOT_ESTABLISHED',historical_as_of='NOT_ESTABLISHED')
    bound=dict(q,k_class_options=options)
    assurance=iv._assurance(bound,result,extra_code={Path(__file__).name:digest(Path(__file__).read_bytes())})
    if assurance['code_sha256']!=initial_code: iv._fail('LIML_CODE_CHANGED_DURING_RUN')
    result.update(assurance)
    return result
