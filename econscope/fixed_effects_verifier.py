"""Independent explicit-dummy Gauss-Jordan and influence covariance oracle.

Never calls the absorber, its basis/within matrices, covariance builder or
native solver. Internal validation only; not external statistical replication.
"""
from fractions import Fraction as F

from .execution_control import checkpoint
from .workspace import canonical,digest


def _require(condition):
    if not condition: raise ValueError('FIXED_EFFECTS_DUMMY_ORACLE_MISMATCH')


def _pivots(matrix):
    a=[list(row) for row in matrix];row=0;chosen=[]
    for column in range(len(a[0])):
        checkpoint();pivot=next((r for r in range(row,len(a)) if a[r][column]),None)
        if pivot is None: continue
        a[row],a[pivot]=a[pivot],a[row];scale=a[row][column]
        a[row]=[value/scale for value in a[row]]
        for r in range(row+1,len(a)):
            factor=a[r][column]
            a[r]=[x-factor*y for x,y in zip(a[r],a[row])]
        chosen.append(column);row+=1
        if row==len(a): break
    return chosen


def _inverse(matrix):
    n=len(matrix);a=[list(row)+[F(i==j) for j in range(n)] for i,row in enumerate(matrix)]
    for column in range(n):
        checkpoint();pivot=next((r for r in range(column,n) if a[r][column]),None)
        if pivot is None:return None
        a[column],a[pivot]=a[pivot],a[column];scale=a[column][column]
        a[column]=[value/scale for value in a[column]]
        for r in range(n):
            if r==column:continue
            factor=a[r][column];a[r]=[x-factor*y for x,y in zip(a[r],a[column])]
    return [row[n:] for row in a]


def verify_computed(q,result):
    records=q['rows'];names=sorted(q['predictor_units']);n=len(records);k=len(names)
    dummy=[]
    for dim in q['fixed_effects']:
        for level in sorted({row['fixed_effects'][dim] for row in records}):
            dummy.append([F(row['fixed_effects'][dim]==level) for row in records])
    d=list(map(list,zip(*dummy)));independent=_pivots(d);r=len(independent)
    z=[[row[j] for j in independent]+[F(record['x'][name]) for name in names] for row,record in zip(d,records)]
    m=len(z[0]);rank=len(_pivots(z));df=n-rank
    _require(result['fixed_effects_rank']==r and result['total_model_rank']==rank and result['identified_slope_rank']==rank-r)
    _require(result['residual_degrees_of_freedom']==df and result['training']['days']==[row['id'] for row in records])
    w=[F(row['analytic_weight']) for row in records];y=[F(row['y']) for row in records]
    gram=[[sum(w[t]*z[t][i]*z[t][j] for t in range(n)) for j in range(m)] for i in range(m)]
    inverse=_inverse(gram)
    if inverse is None:
        _require(result['coefficients_exact'] is None and all(rec['matrix_exact'] is None for rec in result['covariances'].values()))
        return dict(status='VERIFIED_EXACT_NONIDENTIFIABILITY',method='EXPLICIT_DUMMY_COLUMN_RANK_NO_PSEUDOINVERSE')
    rhs=[sum(w[t]*z[t][j]*y[t] for t in range(n)) for j in range(m)]
    beta=[sum(inverse[i][j]*rhs[j] for j in range(m)) for i in range(m)]
    residual=[y[t]-sum(z[t][j]*beta[j] for j in range(m)) for t in range(n)]
    absorbed=[sum(z[t][j]*beta[j] for j in range(r)) for t in range(n)]
    hat=[w[t]*sum(z[t][i]*inverse[i][j]*z[t][j] for i in range(m) for j in range(m)) for t in range(n)]
    _require(result['coefficients_exact']==list(map(str,beta[-k:])))
    _require(result['residuals_exact']==list(map(str,residual)) and result['absorbed_effect_exact']==list(map(str,absorbed)))
    _require(result['full_model_leverage_exact']==list(map(str,hat)) and sum(hat)==rank)
    sse=sum(weight*e*e for weight,e in zip(w,residual))
    _require(F(result['weighted_mse_exact'])==sse/sum(w))
    influence=[[w[t]*residual[t]*sum(inverse[r+i][j]*z[t][j] for j in range(m)) for i in range(k)] for t in range(n)]
    clusters=q['covariance']['clusters'];groups={}
    if clusters:
        for index,row in enumerate(records):groups.setdefault(clusters[row['id']],[]).append(index)
    g=len(groups);expected={}
    for method in q['covariance']['estimators']:
        checkpoint()
        reason=('NO_RESIDUAL_DEGREES_OF_FREEDOM' if df<=0 else
                'LEVERAGE_ONE_CORRECTION_UNDEFINED' if method in ('HC2','HC3') and F(1) in hat else
                'AT_LEAST_TWO_CLUSTERS_REQUIRED' if method.startswith('CLUSTER_') and g<2 else None)
        if reason:
            expected[method]=dict(status='BLOCKED',reason=reason,matrix_exact=None);continue
        if method=='CLASSICAL_WLS':
            matrix=[[sse/df*inverse[r+i][r+j] for j in range(k)] for i in range(k)]
        else:
            matrix=[[F(0)]*k for _ in range(k)]
            pairs=((t,u) for members in groups.values() for t in members for u in members) if method.startswith('CLUSTER_') else ((t,t) for t in range(n))
            for t,u in pairs:
                correction=F(1)/(1-hat[t])**({'HC2':1,'HC3':2}.get(method,0))
                for i in range(k):
                    for j in range(k):matrix[i][j]+=influence[t][i]*influence[u][j]*correction
            factor=F(n,df) if method=='HC1' else F(g,g-1)*F(n-1,df) if method=='CLUSTER_CR1' else F(1)
            matrix=[[value*factor for value in row] for row in matrix]
        expected[method]=dict(status='COMPUTED_ASSUMPTIONS_UNVERIFIED',reason=None,matrix_exact=[list(map(str,row)) for row in matrix])
    _require(canonical(result['covariances'])==canonical(expected))
    return dict(status='VERIFIED_EXPLICIT_DUMMY_REGRESSION_AND_COVARIANCE',
        method='INDEPENDENT_DUMMY_RANK_GAUSS_JORDAN_AND_PAIRWISE_INFLUENCE',covariance_sha256=digest(canonical(expected)),
        scope='DECLARED_DENSE_REFERENCE_MATH_NOT_ORIGINAL_STATISTICS_PACKAGE_OR_INFERENTIAL_VALIDITY')
