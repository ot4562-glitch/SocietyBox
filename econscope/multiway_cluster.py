"""Exact two-way cluster-robust sandwich arithmetic for declared score rows.

This is the CR0 inclusion-exclusion estimator V_a + V_b - V_(a,b).
The caller supplies precomputed score rows w_i * x_i * residual_i and a
symmetric positive-definite inverse information matrix (bread). No fitting,
finite-cluster correction, sampling-design certification, causal inference or
externally authenticated evidence is performed here. Indefinite covariance
results fail closed rather than being passed on to a Wald test.
"""
from fractions import Fraction as F
import math
import re

from .execution_control import checkpoint

_DECIMAL = re.compile(r'-?(?:0|[1-9][0-9]{0,45})(?:\.[0-9]{1,45})?(?:/[1-9][0-9]{0,45})?\Z')


def _fail(code):
    raise ValueError('MULTIWAY_CLUSTER_'+code)


def _fraction(value):
    if type(value) not in (str, int, float, F):
        _fail('NUMBER_INVALID')
    if type(value) is str and not _DECIMAL.fullmatch(value):
        _fail('NUMBER_INVALID')
    if type(value) is float and not math.isfinite(value):
        _fail('NUMBER_INVALID')
    if type(value) is int and value.bit_length() > 4096:
        _fail('NUMBER_RANGE')
    try:
        result = F(str(value)) if type(value) in (str,float) else F(value)
    except (ValueError, ZeroDivisionError, OverflowError):
        _fail('NUMBER_INVALID')
    if result.numerator.bit_length() > 4096 or result.denominator.bit_length() > 4096:
        _fail('NUMBER_RANGE')
    return result


def _positive_definite(matrix):
    work=[row[:] for row in matrix]; size=len(work)
    for k in range(size):
        pivot=work[k][k]
        if pivot<=0: return False
        for i in range(k+1,size):
            for j in range(i,size):
                value=work[i][j]-work[i][k]*work[k][j]/pivot
                work[i][j]=value; work[j][i]=value
    return True


def _positive_semidefinite(matrix):
    work=[row[:] for row in matrix]; size=len(work)
    for k in range(size):
        pivot=work[k][k]
        if pivot<0: return False
        if pivot==0:
            if any(work[k][j] for j in range(k+1,size)): return False
            continue
        for i in range(k+1,size):
            for j in range(i,size):
                value=work[i][j]-work[i][k]*work[k][j]/pivot
                work[i][j]=value; work[j][i]=value
    return True


def two_way_cluster_covariance(scores, inverse_information, groups_a, groups_b):
    """Calculate exact CR0 two-way clustered covariance from declared scores.

    Scores must already reflect any regression/analytic weights. Bread is the
    inverse *unnormalized* information matrix; multiplying the scores by any
    common scaling without matching bread changes the covariance units.
    Cluster labels have at least two groups in both dimensions; neither cluster
    count nor the calculated PSD matrix verifies asymptotic validity.
    """
    if (type(scores) is not list or not 2<=len(scores)<=4096 or
            type(scores[0]) is not list or not 1<=len(scores[0])<=16 or
            len(scores)<=len(scores[0])):
        _fail('SCORES_DIMENSIONS_INVALID')
    n,p=len(scores),len(scores[0])
    if any(type(row) is not list or len(row)!=p for row in scores):
        _fail('SCORES_DIMENSIONS_INVALID')
    if type(inverse_information) is not list or len(inverse_information)!=p or any(
            type(row) is not list or len(row)!=p for row in inverse_information):
        _fail('BREAD_DIMENSIONS_INVALID')
    if (type(groups_a) is not list or type(groups_b) is not list or
            len(groups_a)!=n or len(groups_b)!=n):
        _fail('GROUPS_DIMENSIONS_INVALID')
    if any(type(s) is not str or not 1<=len(s)<=64 or not s.isascii() or
           any(not(ch.isalnum() or ch in '_-') for ch in s)
           for s in groups_a+groups_b):
        _fail('GROUPS_LABELS_INVALID')
    ca,cb=len(set(groups_a)),len(set(groups_b))
    if ca<2 or cb<2:
        _fail('AT_LEAST_TWO_CLUSTERS_PER_DIMENSION_REQUIRED')
    score=[[ _fraction(x) for x in row] for row in scores]
    bread=[[ _fraction(x) for x in row] for row in inverse_information]
    if any(bread[i][j]!=bread[j][i] for i in range(p) for j in range(i+1,p)):
        _fail('BREAD_NOT_SYMMETRIC')
    if not _positive_definite(bread):
        _fail('BREAD_NOT_POSITIVE_DEFINITE')

    def meat(keys):
        totals={}
        for index,(key,row) in enumerate(zip(keys,score)):
            if index%64==0: checkpoint()
            v=totals.setdefault(key,[F(0)]*p)
            for j in range(p): v[j]+=row[j]
        result=[[F(0)]*p for _ in range(p)]
        for value in totals.values():
            checkpoint()
            for i in range(p):
                for j in range(i,p): result[i][j]+=value[i]*value[j]
        for i in range(p):
            for j in range(i): result[i][j]=result[j][i]
        return result

    am,bm,im=meat(groups_a),meat(groups_b),meat(list(zip(groups_a,groups_b)))
    combined=[[am[i][j]+bm[i][j]-im[i][j] for j in range(p)] for i in range(p)]
    if not _positive_semidefinite(combined):
        _fail('CLUSTER_MEAT_NOT_POSITIVE_SEMIDEFINITE')
    left=[[sum(bread[i][k]*combined[k][j] for k in range(p)) for j in range(p)] for i in range(p)]
    cov=[[sum(left[i][k]*bread[j][k] for k in range(p)) for j in range(p)] for i in range(p)]
    if not _positive_semidefinite(cov):
        _fail('COVARIANCE_NOT_POSITIVE_SEMIDEFINITE')
    return dict(status='COMPUTED_ASSUMPTIONS_UNVERIFIED',method='TWO_WAY_CLUSTER_CR0',
                covariance_exact=[list(map(str,row)) for row in cov],
                cluster_meat_exact=[list(map(str,row)) for row in combined],
                observations=n,parameters=p,cluster_counts=dict(a=ca,b=cb,intersection=len(set(zip(groups_a,groups_b)))),
                assumptions='DECLARED_CLUSTERING_AND_EXOGENEITY_NOT_VERIFIED',
                finite_sample_correction='NONE_CR0',arithmetic='EXACT_FRACTION_INCLUSION_EXCLUSION')
