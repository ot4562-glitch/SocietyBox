"""Exact linear-contrast Wald arithmetic with conditional chi-square tails.

The covariance matrix is supplied by a separate, explicitly justified
estimator. Exact contrasts and matrix inversion do not prove that its sampling
assumptions, asymptotic chi-square calibration, or causal interpretation hold.
A singular restriction covariance is blocked, not silently pseudoinverted.
"""
from fractions import Fraction as F
import math
import re

from .execution_control import checkpoint
from .inference_distributions import log_chi_square_survival

_NUMBER = re.compile(r'-?(?:0|[1-9][0-9]{0,45})(?:\.[0-9]{1,45})?(?:/[1-9][0-9]{0,45})?\Z')


def _fail(reason):
    raise ValueError('WALD_INFERENCE_'+reason)


def _number(value):
    if type(value) not in (int,float,str,F): _fail('NUMBER_INVALID')
    if type(value) is float and not math.isfinite(value): _fail('NUMBER_INVALID')
    if type(value) is str and not _NUMBER.fullmatch(value): _fail('NUMBER_INVALID')
    if type(value) is int and value.bit_length()>4096: _fail('NUMBER_RANGE')
    try: parsed=F(str(value)) if type(value) in (str,float) else F(value)
    except (ValueError,ZeroDivisionError,OverflowError): _fail('NUMBER_INVALID')
    if parsed.numerator.bit_length()>4096 or parsed.denominator.bit_length()>4096:
        _fail('NUMBER_RANGE')
    return parsed


def _psd(matrix,*,strict=False):
    work=[row[:] for row in matrix]; n=len(work)
    for k in range(n):
        checkpoint()
        pivot=work[k][k]
        if pivot<0 or (strict and pivot==0): return False
        if pivot==0:
            if any(work[k][j] for j in range(k+1,n)): return False
            continue
        for i in range(k+1,n):
            for j in range(i,n):
                value=work[i][j]-work[i][k]*work[k][j]/pivot
                work[i][j]=value; work[j][i]=value
    return True


def _inverse(matrix):
    n=len(matrix)
    aug=[list(row)+[F(i==j) for j in range(n)] for i,row in enumerate(matrix)]
    for j in range(n):
        checkpoint()
        pivot=next((i for i in range(j,n) if aug[i][j]),None)
        if pivot is None: _fail('RESTRICTION_COVARIANCE_SINGULAR')
        aug[j],aug[pivot]=aug[pivot],aug[j]
        scale=aug[j][j]
        aug[j]=[x/scale for x in aug[j]]
        for i in range(n):
            if i==j: continue
            factor=aug[i][j]
            if factor: aug[i]=[x-factor*y for x,y in zip(aug[i],aug[j])]
    return [row[n:] for row in aug]


def linear_wald_test(coefficients, covariance, restrictions, null_values):
    """Test declared R * beta = r using a central asymptotic chi-square law.

    Covariance must be a symmetric PSD covariance *of the coefficients*, not
    an inverse information matrix or residual covariance. Restriction rank is
    checked through R*covariance*R', which must be positive definite.
    Returns exact rational Wald arithmetic and a conditional binary64 p-value.
    """
    if type(coefficients) is not list or not 1<=len(coefficients)<=16:
        _fail('COEFFICIENT_DIMENSIONS_INVALID')
    p=len(coefficients)
    if type(covariance) is not list or len(covariance)!=p or any(
            type(row) is not list or len(row)!=p for row in covariance):
        _fail('COVARIANCE_DIMENSIONS_INVALID')
    if type(restrictions) is not list or not 1<=len(restrictions)<=min(p,8) or any(
            type(row) is not list or len(row)!=p for row in restrictions):
        _fail('RESTRICTIONS_DIMENSIONS_INVALID')
    q=len(restrictions)
    if type(null_values) is not list or len(null_values)!=q:
        _fail('NULL_DIMENSIONS_INVALID')
    beta=list(map(_number,coefficients))
    cov=[[ _number(v) for v in row] for row in covariance]
    r=[[ _number(v) for v in row] for row in restrictions]
    null=list(map(_number,null_values))
    if any(cov[i][j]!=cov[j][i] for i in range(p) for j in range(i+1,p)):
        _fail('COVARIANCE_NOT_SYMMETRIC')
    if not _psd(cov): _fail('COVARIANCE_NOT_POSITIVE_SEMIDEFINITE')
    contrasts=[sum(row[j]*beta[j] for j in range(p)) for row in r]
    difference=[v-target for v,target in zip(contrasts,null)]
    rc=[[sum(r[i][k]*cov[k][j] for k in range(p)) for j in range(p)] for i in range(q)]
    information=[[sum(rc[i][k]*r[j][k] for k in range(p)) for j in range(q)] for i in range(q)]
    if not _psd(information,strict=True):
        _fail('RESTRICTION_COVARIANCE_SINGULAR_OR_INVALID')
    inverse=_inverse(information)
    distance=sum(difference[i]*inverse[i][j]*difference[j] for i in range(q) for j in range(q))
    if distance<0: _fail('NEGATIVE_WALD_STATISTIC')
    try:
        log_tail=log_chi_square_survival(distance,q)
        p_value=math.exp(log_tail)
    except (OverflowError,ValueError) as error:
        _fail('REFERENCE_TAIL_NOT_EVALUABLE')
    if not 0<=p_value<=1 or not math.isfinite(p_value):
        _fail('REFERENCE_TAIL_NOT_EVALUABLE')
    return dict(status='COMPUTED_ASSUMPTIONS_UNVERIFIED',
                method='LINEAR_WALD_ASYMPTOTIC_CHI_SQUARE',
                degrees_of_freedom=q,
                contrasts_exact=list(map(str,contrasts)),
                null_values_exact=list(map(str,null)),
                contrast_differences_exact=list(map(str,difference)),
                restriction_covariance_exact=[list(map(str,row)) for row in information],
                wald_statistic_exact=str(distance),
                chi_square_upper_tail=p_value,
                log_chi_square_upper_tail=log_tail,
                reference_distribution='CENTRAL_CHI_SQUARE_ASYMPTOTIC_UNVERIFIED',
                assumptions='COVARIANCE_ESTIMATOR_EXOGENEITY_AND_ASYMPTOTICS_NOT_VERIFIED',
                arithmetic='EXACT_FRACTION_GAUSS_JORDAN_CONTRASTS')
