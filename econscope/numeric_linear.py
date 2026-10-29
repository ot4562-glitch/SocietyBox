"""Bounded binary64 least squares by equilibrated column-pivoted Householder QR.

No normal-equation inversion or observation-square projection. Rank decisions
are numerical, not exact identifiability certificates. The same compact factor
solves multiple right sides and constructs the inverse information when needed.
"""
import math
import sys

from .execution_control import checkpoint


def _fail(reason): raise ValueError('NUMERIC_LINEAR_'+reason)


def _finite(value):
    if type(value) not in (int,float): _fail('FINITE_NUMBER_REQUIRED')
    try: result=float(value)
    except OverflowError as error: raise ValueError('NUMERIC_LINEAR_FINITE_RANGE') from error
    if not math.isfinite(result): _fail('FINITE_NUMBER_REQUIRED')
    return result


def _dot(a,b):
    try: value=math.fsum(x*y for x,y in zip(a,b))
    except (OverflowError,ValueError) as error: raise ValueError('NUMERIC_LINEAR_FINITE_RANGE') from error
    if not math.isfinite(value): _fail('FINITE_RANGE')
    return value


class _QR:
    def __init__(self,matrix):
        if (type(matrix) is not list or not 1<=len(matrix)<=1024 or type(matrix[0]) is not list or
                not 1<=len(matrix[0])<=16 or len(matrix)<len(matrix[0]) or
                any(type(row) is not list or len(row)!=len(matrix[0]) for row in matrix)):
            _fail('DIMENSIONS_INVALID')
        self.n,self.p=len(matrix),len(matrix[0])
        a=[list(map(_finite,row)) for row in matrix]
        self.scales=[math.hypot(*column) for column in zip(*a)]
        if any(not math.isfinite(s) for s in self.scales): _fail('FINITE_RANGE')
        if any(s==0 for s in self.scales): _fail('NUMERICAL_RANK_DEFICIENT')
        a=[[v/s for v,s in zip(row,self.scales)] for row in a]
        self.permutation=list(range(self.p)); self.reflectors=[]
        self.rank_tolerance=64*sys.float_info.epsilon*max(self.n,self.p)
        for j in range(self.p):
            checkpoint()
            pivot=max(range(j,self.p),key=lambda col:math.hypot(*(a[i][col] for i in range(j,self.n))))
            for row in a: row[j],row[pivot]=row[pivot],row[j]
            self.permutation[j],self.permutation[pivot]=self.permutation[pivot],self.permutation[j]
            norm=math.hypot(*(a[i][j] for i in range(j,self.n)))
            if norm<=self.rank_tolerance: _fail('NUMERICAL_RANK_DEFICIENT')
            alpha=-math.copysign(norm,a[j][j]); vector=[a[i][j] for i in range(j,self.n)]; vector[0]-=alpha
            length=math.hypot(*vector); vector=[v/length for v in vector]
            for col in range(j,self.p):
                multiplier=2*_dot(vector,[a[i][col] for i in range(j,self.n)])
                for offset,v in enumerate(vector): a[j+offset][col]-=multiplier*v
            a[j][j]=alpha
            for i in range(j+1,self.n): a[i][j]=0.0
            self.reflectors.append(vector)
        self.upper=[row[:self.p] for row in a[:self.p]]

    def _backsolve(self,rhs):
        result=[0.0]*self.p
        for i in range(self.p-1,-1,-1):
            result[i]=(rhs[i]-_dot(self.upper[i][i+1:],result[i+1:]))/self.upper[i][i]
            if not math.isfinite(result[i]): _fail('FINITE_RANGE')
        return result

    def solve(self,rhs):
        if type(rhs) is not list or len(rhs)!=self.n: _fail('RHS_DIMENSIONS_INVALID')
        transformed=list(map(_finite,rhs))
        for j,vector in enumerate(self.reflectors):
            multiplier=2*_dot(vector,transformed[j:])
            for offset,v in enumerate(vector): transformed[j+offset]-=multiplier*v
        permuted=self._backsolve(transformed[:self.p]); result=[0.0]*self.p
        for j,index in enumerate(self.permutation): result[index]=permuted[j]/self.scales[index]
        if any(not math.isfinite(v) for v in result): _fail('FINITE_RANGE')
        return result

    def inverse_information(self):
        columns=[self._backsolve([float(i==j) for i in range(self.p)]) for j in range(self.p)]
        inverse=[list(row) for row in zip(*columns)]; result=[[0.0]*self.p for _ in range(self.p)]
        for i,index in enumerate(self.permutation):
            for j,other in enumerate(self.permutation):
                result[index][other]=_dot(inverse[i],inverse[j])/self.scales[index]/self.scales[other]
        if any(not math.isfinite(v) for row in result for v in row) or any(result[i][i]<=0 for i in range(self.p)):
            _fail('INVERSE_INFORMATION_RANGE')
        return result


def least_squares(design,target,*,weights=None):
    """Positive analytic-weight least squares and inverse X'WX, not covariance."""
    if type(design) is not list or not design or type(target) is not list or len(target)!=len(design):
        _fail('DIMENSIONS_INVALID')
    if weights is None: weights=[1.0]*len(design)
    if type(weights) is not list or len(weights)!=len(design): _fail('WEIGHTS_INVALID')
    weights=list(map(_finite,weights))
    if any(w<=0 for w in weights): _fail('POSITIVE_WEIGHTS_REQUIRED')
    roots=list(map(math.sqrt,weights))
    if any(type(row) is not list for row in design): _fail('DIMENSIONS_INVALID')
    factor=_QR([[_finite(v)*root for v in row] for row,root in zip(design,roots)])
    y=list(map(_finite,target)); coefficients=factor.solve([v*root for v,root in zip(y,roots)])
    residuals=[v-_dot(list(map(_finite,row)),coefficients) for row,v in zip(design,y)]
    return dict(coefficients=coefficients,inverse_information=factor.inverse_information(),residuals=residuals,
                weighted_sse=_dot([w*e for w,e in zip(weights,residuals)],residuals),
                column_permutation=list(factor.permutation),rank_tolerance=factor.rank_tolerance,
                arithmetic='BINARY64_COLUMN_EQUILIBRATED_PIVOTED_HOUSEHOLDER_QR')
