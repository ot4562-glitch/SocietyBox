"""Independent cofactor/Cramer and pairwise-influence arithmetic for declared WLS.

Does not use the estimator's design builder, elimination, bread or meat. This
internal oracle checks already validated declarations; not an external result
import/authentication interface or proof of statistical assumptions.
"""
from fractions import Fraction as F

from .inverse_numeric_verifier import _determinant
from .workspace import canonical, digest


def _require(value):
    if not value: raise ValueError('REGRESSION_COVARIANCE_ORACLE_MISMATCH')


def verify_computed(parameters,result):
    data=parameters['data']; features=parameters['features']; records=data['observations']
    first=max(f['lag'] for f in features); last=next(i for i,r in enumerate(records) if r['day']==data['training_end_day'])
    training=records[first:last+1]; n=len(training); p=1+len(features)
    design=[[F(1)]+[F(records[t-f['lag']]['values'][f['series']]) for f in features] for t in range(first,last+1)]
    y=[F(r['values'][data['target']]) for r in training]; w=[F(parameters['weights'][r['day']]) for r in training]
    a=[[sum(w[t]*design[t][i]*design[t][j] for t in range(n)) for j in range(p)] for i in range(p)]
    b=[sum(w[t]*design[t][j]*y[t] for t in range(n)) for j in range(p)]; determinant=_determinant(a)
    _require(result['training']['days']==[r['day'] for r in training] and result['training']['count']==n)
    _require(result['parameter_order']==['intercept']+[f['series']+'[t-'+str(f['lag'])+']' for f in features])
    if not determinant:
        expected={method:dict(status='BLOCKED',reason='RANK_DEFICIENT',matrix_exact=None) for method in parameters['covariance']['estimators']}
        _require(result['coefficients_exact'] is None and result['residuals_exact'] is None and result['covariances']==expected)
        return dict(status='VERIFIED_EXACT_RANK_DEFICIENCY',method='LEIBNIZ_DETERMINANT_NO_PSEUDOINVERSE')
    beta=[_determinant([[b[i] if k==j else a[i][k] for k in range(p)] for i in range(p)])/determinant for j in range(p)]
    residual=[y[t]-sum(design[t][j]*beta[j] for j in range(p)) for t in range(n)]
    _require(result['coefficients_exact']==list(map(str,beta)) and result['residuals_exact']==list(map(str,residual)))
    _require(F(result['weighted_mse_exact'])==sum(wi*e*e for wi,e in zip(w,residual))/sum(w))
    # Inverse entries by adjugate, rather than solving identity RHS columns.
    inverse=[[F((-1)**(i+j))*_determinant([[a[r][c] for c in range(p) if c!=i] for r in range(p) if r!=j])/determinant for j in range(p)] for i in range(p)]
    # Influence for raw residual e_t: (X'WX)^(-1) x_t w_t e_t.
    influence=[[sum(inverse[j][k]*design[t][k] for k in range(p))*w[t]*residual[t] for j in range(p)] for t in range(n)]
    hat=[w[t]*sum(design[t][i]*inverse[i][j]*design[t][j] for i in range(p) for j in range(p)) for t in range(n)]
    cov=parameters['covariance']; groups=cov['clusters']; labels=[groups[r['day']] for r in training] if groups else []
    # Independent pairwise-influence oracle: enumerate exactly the nonzero
    # kernel cells instead of n*n cells for every covariance method. This
    # retains the original formula, not the estimator's bread/meat algorithm.
    cluster_members={}
    for index,label in enumerate(labels):
        cluster_members.setdefault(label,[]).append(index)
    g=len(cluster_members); expected={}
    for method in cov['estimators']:
        reason='LEVERAGE_ONE_CORRECTION_UNDEFINED' if method in ('HC2','HC3') and F(1) in hat else 'AT_LEAST_TWO_CLUSTERS_REQUIRED' if method.startswith('CLUSTER_') and g<2 else None
        entry=dict(status='BLOCKED' if reason else 'COMPUTED_ASSUMPTIONS_UNVERIFIED',reason=reason,matrix_exact=None)
        if reason: expected[method]=entry; continue
        if method=='CLASSICAL_WLS':
            s2=sum(w[t]*residual[t]**2 for t in range(n))/(n-p)
            matrix=[[s2*inverse[i][j] for j in range(p)] for i in range(p)]
        else:
            matrix=[[F(0)]*p for _ in range(p)]
            if method.startswith('CLUSTER_'):
                pairs=((t,u) for members in cluster_members.values()
                       for t in members for u in members)
            elif method=='HAC_BARTLETT':
                bandwidth=cov['bandwidth']
                pairs=((t,u) for t in range(n)
                       for u in range(max(0,t-bandwidth),min(n,t+bandwidth+1)))
            else:
                pairs=((t,t) for t in range(n))
            for t,u in pairs:
                if method.startswith('CLUSTER_'):
                    kernel=F(1)
                elif method=='HAC_BARTLETT':
                    kernel=F(bandwidth+1-abs(t-u),bandwidth+1)
                else:
                    kernel=F(1)/(1-hat[t])**({'HC2':1,'HC3':2}.get(method,0))
                for i in range(p):
                    for j in range(p): matrix[i][j]+=kernel*influence[t][i]*influence[u][j]
            factor=F(n,n-p) if method=='HC1' else F(g,g-1)*F(n-1,n-p) if method=='CLUSTER_CR1' else F(1)
            matrix=[[v*factor for v in row] for row in matrix]
        entry['matrix_exact']=[list(map(str,row)) for row in matrix]; expected[method]=entry
    _require(canonical(result['covariances'])==canonical(expected))
    return dict(status='VERIFIED_EXACT_COEFFICIENTS_AND_COVARIANCE',method='LEIBNIZ_CRAMER_ADJUGATE_AND_PAIRWISE_INFLUENCE_KERNELS',
        covariance_sha256=digest(canonical(expected)),scope='DECLARED_TRAINING_ARITHMETIC_NOT_STATISTICAL_ASSUMPTION_OR_EXTERNAL_REPLICATION')
