"""Independent dense GLS and residual-maker moment oracles, no native runtime."""
import copy
from fractions import Fraction as F
import importlib
import random
import unittest
from unittest.mock import patch

from econscope.workspace import canonical,digest


def api(): return importlib.import_module('econscope.random_effects')


def case(feasible=False,unbalanced=False):
    values=[[1,3,2],[4,6,9],[3,5,8],[7,8,12]]
    rows=[]
    for i,ys in enumerate(values):
        for t,y in enumerate(ys[:2] if unbalanced and i%2 else ys):
            rows.append(dict(entity=str(i),period=t,outcome=str(y),
                             regressors={'x':str((i+2*t)%5),'z':str(i%2)}))
    return dict(schema='societybox-random-effects-v1',rows=rows,
        model=dict(outcome='y',regressors=['x','z'],include_intercept=True),
        units={'y':'dollars','x':'hours','z':'indicator'},period_unit='year',
        source_declared_sha256='a'*64,rights=dict(local_processing='PERMITTED',report_export='PERMITTED'),
        variance_components=({'method':'BETWEEN_TRACE_MOMENTS'} if feasible else
                             dict(method='DECLARED',idiosyncratic='2',entity='3')),
        covariance='ENTITY_CR0')


def maximum_case(feasible=False,long_blocks=False):
    q=case(feasible);rng=random.Random(81627);q['model']['regressors']=['x'+str(j) for j in range(7)]
    q['units']={'y':'units',**{v:'units' for v in q['model']['regressors']}}
    g,t=(4,128) if long_blocks else (128,4)
    q['rows']=[dict(entity=str(i),period=3*s,outcome=str(3*(i%11)+rng.randint(-9,9)),
        regressors={v:str(rng.randint(-5,5)) for v in q['model']['regressors']}) for i in range(g) for s in range(t)]
    return q


def mm(a,b):return [[sum(x*y for x,y in zip(row,col)) for col in zip(*b)] for row in a]
def tr(a):return list(map(list,zip(*a)))
def inverse(a):
    # Independent full augmented elimination, only for tiny oracle matrices.
    n=len(a); rows=[list(row)+[F(i==j) for j in range(n)] for i,row in enumerate(a)]
    for j in range(n):
        pivot=next(i for i in range(j,n) if rows[i][j]);rows[j],rows[pivot]=rows[pivot],rows[j]
        c=rows[j][j];rows[j]=[x/c for x in rows[j]]
        for i in range(n):
            if i!=j:
                c=rows[i][j];rows[i]=[x-c*y for x,y in zip(rows[i],rows[j])]
    return [row[n:] for row in rows]


def raw(q):
    x=[[F(1)]*q['model']['include_intercept']+[F(r['regressors'][v]) for v in q['model']['regressors']] for r in q['rows']]
    y=[[F(r['outcome'])] for r in q['rows']]
    return x,y


def dense_oracle(q,se,su):
    x,y=raw(q);n=len(x);k=len(x[0]);entities=sorted({r['entity'] for r in q['rows']})
    omega=[[F(se*(i==j)+su*(q['rows'][i]['entity']==q['rows'][j]['entity'])) for j in range(n)] for i in range(n)]
    precision=inverse(omega);bread=inverse(mm(mm(tr(x),precision),x))
    beta=mm(mm(mm(bread,tr(x)),precision),y)
    e=[[a[0]-b[0]] for a,b in zip(y,mm(x,beta))]
    pe=mm(precision,e);scores=[]
    for entity in entities:
        scores.append([sum(x[i][j]*pe[i][0] for i in range(n) if q['rows'][i]['entity']==entity) for j in range(k)])
    meat=mm(tr(scores),scores);cov=mm(mm(bread,meat),bread)
    if q['covariance']=='MODEL_BASED':cov=bread
    elif q['covariance']=='ENTITY_CR1':
        factor=F(len(entities),len(entities)-1)*F(n-1,n-k);cov=[[factor*v for v in row] for row in cov]
    return dict(beta=[r[0] for r in beta],bread=bread,cov=cov,residual=[r[0] for r in e],objective=mm(tr(e),pe)[0][0])


def moments_oracle(q):
    # Full residual makers for within y on entity dummies + varying x,
    # and unweighted between regression. Test fixture z/intercept absorb.
    x,y=raw(q);n=len(x);groups=sorted({r['entity'] for r in q['rows']});g=len(groups)
    d=[[F(r['entity']==s) for s in groups] for r in q['rows']]
    xd=[row+[x[i][1]] for i,row in enumerate(d)]
    pw=mm(mm(xd,inverse(mm(tr(xd),xd))),tr(xd))
    mw=[[F(i==j)-pw[i][j] for j in range(n)] for i in range(n)]
    sse=mm(mm(tr(y),mw),y)[0][0];dfw=n-g-1;se=sse/dfw
    counts=[sum(r['entity']==s for r in q['rows']) for s in groups]
    a=[[F(r['entity']==s,nt) for r in q['rows']] for s,nt in zip(groups,counts)]
    xb,yb=mm(a,x),mm(a,y)
    pb=mm(mm(xb,inverse(mm(tr(xb),xb))),tr(xb))
    mb=[[F(i==j)-pb[i][j] for j in range(g)] for i in range(g)]
    ssb=mm(mm(tr(yb),mb),yb)[0][0];dfb=g-len(x[0])
    trace=sum(mb[i][i]/counts[i] for i in range(g));su=(ssb-se*trace)/dfb
    return dict(se=se,su_raw=su,su=max(F(0),su),within_ssr=sse,between_ssr=ssb,trace=trace,within_df=dfw,between_df=dfb)


class RandomEffectsTests(unittest.TestCase):
    def test_large_full_gram_and_cluster_covariance_match_independent_non_symmetric_loops(self):
        # The production GLS and sandwich may exploit symmetry only if it is
        # exact; this oracle computes every ordered pair independently.
        from econscope import instrumental_variables as iv
        q=maximum_case();q['rows']=q['rows'][:96];q['covariance']='ENTITY_CR0'
        result=api().calculate(q)
        x,y,groups=api()._validated(q)
        transformed=api()._precision_apply([[v]+row for v,row in zip(y,x)],groups,F(2),F(3))
        k=len(x[0])
        full=[[sum(row[i]*p[j+1] for row,p in zip(x,transformed)) for j in range(k)] for i in range(k)]
        self.assertEqual(full,[list(row) for row in zip(*full)])
        bread=inverse(full)
        self.assertEqual([[F(v) for v in row] for row in result['gls_bread_exact']],bread)
        scores=[[F(v) for v in row] for row in result['entity_scores_exact']]
        meat=mm(tr(scores),scores)
        oracle_covariance=mm(mm(bread,meat),bread)
        self.assertEqual([[F(v) for v in row] for row in result['covariance_exact']],oracle_covariance)

    def test_intercept_only_gls_has_closed_form_and_named_linear_predictions(self):
        q=case();q['model']['regressors']=[];q['units']={'y':'units'};q['covariance']='MODEL_BASED'
        q['rows']=[dict(entity=g,period=t,outcome=str(y),regressors={}) for g,ys in [('a',[1,3]),('b',[5,7,9])] for t,y in enumerate(ys)]
        r=api().calculate(q)
        self.assertEqual(r['coefficients_exact'],['106/23']);self.assertEqual(r['covariance_exact'],[['44/23']])
        self.assertEqual(r['entity_effects_exact'],{'a':'-45/23','b':'45/23'})

    def test_declared_balanced_unbalanced_gls_and_all_covariances_match_full_dense_inverse(self):
        for unbalanced in (False,True):
            for cov in ('MODEL_BASED','ENTITY_CR0','ENTITY_CR1'):
                q=case(unbalanced=unbalanced);q['covariance']=cov;r=api().calculate(q);o=dense_oracle(q,F(2),F(3))
                self.assertEqual(list(map(F,r['coefficients_exact'])),o['beta'])
                self.assertEqual([[F(v) for v in row] for row in r['covariance_exact']],o['cov'])
                self.assertEqual(F(r['gls_objective_exact']),o['objective'])
                self.assertEqual(r['normal_equation_residual_exact'],['0']*3)

    def test_trace_moments_match_full_dummy_and_between_projection_with_time_invariant_regressor(self):
        for unbalanced in (False,True):
            q=case(feasible=True,unbalanced=unbalanced);o=moments_oracle(q);r=api().calculate(q);m=r['variance_components']
            for k,ok in [('idiosyncratic_exact','se'),('entity_untruncated_exact','su_raw'),('entity_exact','su'),('between_noise_trace_exact','trace')]:
                self.assertEqual(F(m[k]),o[ok])
            self.assertEqual(m['within_rank'],1);self.assertEqual(m['between_rank'],3)
            self.assertEqual(m['within_df'],o['within_df']);self.assertEqual(m['between_df'],o['between_df'])
            self.assertEqual(list(map(F,r['coefficients_exact'])),dense_oracle(q,o['se'],o['su'])['beta'])

    def test_unbalanced_noise_trace_is_not_naive_harmonic_time_subtraction(self):
        q=case(feasible=True,unbalanced=True);o=moments_oracle(q)
        sizes=[3,2,3,2];naive=o['between_df']*sum(F(1,t) for t in sizes)/4
        self.assertNotEqual(o['trace'],naive)
        self.assertEqual(F(api().calculate(q)['variance_components']['between_noise_trace_exact']),o['trace'])

    def test_zero_entity_variance_is_exactly_pooled_ols_and_shrinkage_zero(self):
        q=case();q['variance_components']['entity']='0';r=api().calculate(q)
        self.assertEqual(list(map(F,r['coefficients_exact'])),dense_oracle(q,F(2),F(0))['beta'])
        self.assertTrue(all(v=='0' for v in r['entity_effects_exact'].values()))

    def test_no_intercept_and_singleton_entities_are_supported_when_variances_declared(self):
        q=case();q['model']['include_intercept']=False;q['rows']=[r for r in q['rows'] if r['period']==0]
        r=api().calculate(q);self.assertEqual(r['parameter_order'],['x','z'])
        self.assertEqual(list(map(F,r['coefficients_exact'])),dense_oracle(q,F(2),F(3))['beta'])
        q['variance_components']={'method':'BETWEEN_TRACE_MOMENTS'}
        with self.assertRaisesRegex(ValueError,'WITHIN_DF'):api().calculate(q)

    def test_all_regressors_time_invariant_can_have_zero_within_rank_and_identified_gls(self):
        q=case(feasible=True);q['model']['regressors']=['z'];q['units'].pop('x')
        for row in q['rows']:row['regressors'].pop('x')
        r=api().calculate(q);self.assertEqual(r['variance_components']['within_rank'],0)
        self.assertEqual(r['variance_components']['within_df'],8)
        m=r['variance_components'];o=dense_oracle(q,F(m['idiosyncratic_exact']),F(m['entity_exact']))
        self.assertEqual(list(map(F,r['coefficients_exact'])),o['beta'])

    def test_between_rank_deficiency_is_separate_from_overall_gls_identification(self):
        q=case(feasible=True)
        for row in q['rows']:row['regressors']['x']=str(row['period']-1)
        r=api().calculate(q);m=r['variance_components']
        self.assertEqual(m['between_rank'],2);self.assertEqual(m['between_df'],2)
        self.assertEqual(F(m['between_noise_trace_exact']),F(2,3))
        self.assertEqual(list(map(F,r['coefficients_exact'])),dense_oracle(q,F(m['idiosyncratic_exact']),F(m['entity_exact']))['beta'])

    def test_zero_rank_between_design_stays_exact_and_does_not_make_gls_unidentified(self):
        q=case(feasible=True);q['model'].update(include_intercept=False,regressors=['x']);q['units'].pop('z')
        for row in q['rows']:row['regressors']={'x':str(row['period']-1)}
        r=api().calculate(q);m=r['variance_components']
        self.assertEqual(m['between_rank'],0);self.assertEqual(m['between_df'],4)
        self.assertEqual(m['between_noise_trace_exact'],'4/3')
        self.assertEqual(list(map(F,r['coefficients_exact'])),dense_oracle(q,F(m['idiosyncratic_exact']),F(m['entity_exact']))['beta'])

    def test_common_declared_variance_scale_changes_model_covariance_but_not_cluster_covariance(self):
        for covariance in ('MODEL_BASED','ENTITY_CR0','ENTITY_CR1'):
            q=case();q['covariance']=covariance;r=api().calculate(q)
            q['variance_components'].update(idiosyncratic='18',entity='27');s=api().calculate(q)
            self.assertEqual(r['coefficients_exact'],s['coefficients_exact'])
            factor=9 if covariance=='MODEL_BASED' else 1
            self.assertEqual([[F(v) for v in row] for row in s['covariance_exact']],[[factor*F(v) for v in row] for row in r['covariance_exact']])

    def test_blup_conditional_residuals_and_covariance_scale_have_distinct_meanings(self):
        q=case(unbalanced=True);r=api().calculate(q)
        for g in r['entity_order']:
            positions=[i for i,row in enumerate(q['rows']) if row['entity']==g]
            expected=F(3,2+3*len(positions))*sum(F(r['marginal_residuals_exact'][i]) for i in positions)
            self.assertEqual(F(r['entity_effects_exact'][g]),expected)
            for i in positions:self.assertEqual(F(r['conditional_residuals_exact'][i]),F(r['marginal_residuals_exact'][i])-expected)
        self.assertEqual(r['entity_effect_interpretation'],'MODEL_CONDITIONAL_LINEAR_PREDICTION_NOT_OBSERVED_EFFECT')

    def test_order_entity_labels_and_period_coordinates_do_not_change_estimates(self):
        q=case(feasible=True,unbalanced=True);r=api().calculate(q)
        q['rows'].reverse()
        for row in q['rows']:row['entity']='new'+row['entity'];row['period']=10000-7*row['period']
        s=api().calculate(q)
        self.assertEqual(r['coefficients_exact'],s['coefficients_exact']);self.assertEqual(r['covariance_exact'],s['covariance_exact'])

    def test_response_and_covariate_units_preserve_exact_transformation(self):
        q=case(feasible=True);r=api().calculate(q)
        for row in q['rows']:row['outcome']=str(7*F(row['outcome']));row['regressors']['x']=str(3*F(row['regressors']['x']))
        s=api().calculate(q);factors=[F(7),F(7,3),F(7)]
        self.assertEqual(list(map(F,s['coefficients_exact'])),[v*F(b) for v,b in zip(factors,r['coefficients_exact'])])
        self.assertEqual([[F(v) for v in row] for row in s['covariance_exact']],[[factors[i]*factors[j]*F(r['covariance_exact'][i][j]) for j in range(3)] for i in range(3)])
        self.assertEqual(F(s['variance_components']['idiosyncratic_exact']),49*F(r['variance_components']['idiosyncratic_exact']))

    def test_feasible_negative_entity_moment_is_reported_as_explicit_boundary(self):
        q=case(feasible=True)
        for row in q['rows']:row['outcome']=str([1,-2,1][row['period']])
        r=api().calculate(q);m=r['variance_components']
        self.assertLess(F(m['entity_untruncated_exact']),0);self.assertEqual(m['entity_exact'],'0')
        self.assertEqual(m['boundary'],'ENTITY_VARIANCE_TRUNCATED_AT_ZERO')

    def test_rank_df_zero_noise_and_variance_boundaries_refuse_estimation(self):
        q=case();q['model']['regressors'].append('duplicate');q['units']['duplicate']='hours'
        for row in q['rows']:row['regressors']['duplicate']=row['regressors']['x']
        with self.assertRaisesRegex(ValueError,'DESIGN_RANK'):api().calculate(q)
        q=case(feasible=True)
        for row in q['rows']:row['outcome']=str(2*F(row['regressors']['x'])+4*F(row['regressors']['z']))
        with self.assertRaisesRegex(ValueError,'POSITIVE_IDIOSYNCRATIC'):api().calculate(q)
        q=case(feasible=True);q['rows']=[r for r in q['rows'] if r['entity']!='3']
        with self.assertRaisesRegex(ValueError,'BETWEEN_DF'):api().calculate(q)
        for key,value in [('idiosyncratic','0'),('entity','-1')]:
            q=case();q['variance_components'][key]=value
            with self.assertRaisesRegex(ValueError,'VARIANCE'):api().calculate(q)

    def test_strict_schema_rights_units_duplicates_and_resource_limits(self):
        changes=[lambda q:q.update(extra=1),lambda q:q['rights'].update(local_processing='DENIED'),
                 lambda q:q['units'].pop('y'),lambda q:q['rows'].append(copy.deepcopy(q['rows'][0])),
                 lambda q:q['rows'][0].update(period=True),lambda q:q['rows'][0].update(outcome=1.5),
                 lambda q:q.update(rows=tuple(q['rows'])),lambda q:q.update(covariance='HC3'),
                 lambda q:q['variance_components'].update(extra='3')]
        for change in changes:
            q=case();change(q)
            with self.assertRaises(ValueError):api().calculate(q)
        q=case();q['rows']=[dict(entity='g'+str(i),period=0,outcome='1',regressors={'x':'1','z':'1'}) for i in range(513)]
        with self.assertRaisesRegex(ValueError,'DIMENSION'):api().calculate(q)

    def test_snapshot_source_guard_and_cancellation(self):
        m=api();q=case();baseline=m.calculate(copy.deepcopy(q));original=m._fit
        def mutate(*args):q['rows'][0]['outcome']='999';return original(*args)
        with patch.object(m,'_fit',side_effect=mutate):r=m.calculate(q)
        self.assertEqual(r['input_sha256'],baseline['input_sha256']);self.assertEqual(r['coefficients_exact'],baseline['coefficients_exact'])
        with patch.object(m,'_code_receipt',side_effect=[{'v':'before'},{'v':'after'}]):
            with self.assertRaisesRegex(ValueError,'CODE_CHANGED'):m.calculate(case())
        with patch.object(m,'checkpoint',side_effect=TimeoutError('cancelled')):
            with self.assertRaises(TimeoutError):m.calculate(case())

    def test_capabilities_and_evidence_preserve_unknown_exogeneity_and_vintage(self):
        from econscope.study_capabilities import negotiate_capabilities
        r=api().calculate(case());c=negotiate_capabilities(required=['exact_random_effects_gls'])['random_effects_capabilities']
        self.assertIn('econscope.random_effects.calculate',c['python'])
        self.assertEqual(r['native_execution'],'NOT_RUN');self.assertEqual(r['causal_identification'],'NOT_ESTABLISHED')
        self.assertEqual(r['source_authentication'],'NOT_ESTABLISHED');self.assertTrue(r['unknowns'])
        self.assertEqual(r['input_sha256'],digest(canonical(case())))
        self.assertEqual(r['calculation_sha256'],digest(canonical({k:r[k] for k in r['calculation_fields']})))
        self.assertEqual(r['evidence'][0]['result']['calculation_sha256'],r['calculation_sha256'])

    def test_maximum_rows_entities_and_parameters_execute_declared_and_feasible_paths(self):
        for feasible in (False,True):
            q=maximum_case(feasible);r=api().calculate(q)
            self.assertEqual(r['sample']['observations'],512);self.assertEqual(r['sample']['entities'],128)
            self.assertEqual(r['sample']['parameters'],8);self.assertEqual(r['normal_equation_residual_exact'],['0']*8)
            self.assertTrue(all(F(r['covariance_exact'][i][i])>=0 for i in range(8)))

    def test_fraction_growth_has_explicit_budget_instead_of_python_decimal_conversion_failure(self):
        q=case();q['model']['regressors']=[];q['units']={'y':'units'}
        q['rows']=[dict(entity=str(i//256),period=i%256,outcome='1/'+str(2**61+2*i+1),regressors={}) for i in range(512)]
        with self.assertRaisesRegex(ValueError,'RANDOM_EFFECTS_EXACT_ARITHMETIC_BIT_LIMIT'):api().calculate(q)


if __name__=='__main__':unittest.main()
