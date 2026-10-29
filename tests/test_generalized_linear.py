import copy
from fractions import Fraction as F
import json
import math
import unittest
from unittest.mock import patch

from econscope import generalized_linear as glm


def case(family='BINOMIAL_LOGIT'):
    y=([0,0,0,1,0,1,1,1] if family=='BINOMIAL_LOGIT' else [0,1,3,4,1,4,5,6])
    return dict(schema='societybox-generalized-linear-model-v1',family=family,
        observations=[dict(observation_id=str(i),response=value,regressors={'x':int(i>=4)},offset=0) for i,value in enumerate(y)],
        model=dict(outcome='outcome',regressors=['x'],intercept=True),
        units={'outcome':'indicator' if family=='BINOMIAL_LOGIT' else 'events','x':'units'},
        covariance=dict(estimator='MODEL_BASED',clusters=None),optimizer=dict(max_iterations=100,tolerance=1e-10),
        source_declared_sha256='a'*64,rights=dict(local_processing='PERMITTED',report_export='PERMITTED'))


class GeneralizedLinearTests(unittest.TestCase):
    def test_fit_does_not_accept_tuple_observations_rejected_by_validator(self):
        q=case();q['observations']=tuple(q['observations'])
        with self.assertRaisesRegex(ValueError,'OBSERVATION'): glm.validate(q)
        with self.assertRaisesRegex(ValueError,'JSON_INPUT|OBSERVATION'): glm.fit(q)

    def test_fit_receipt_binds_actual_training_snapshot_under_caller_mutation(self):
        q=case();baseline=glm.fit(copy.deepcopy(q));original=glm._QR;mutated=[False]
        def qr(matrix):
            if not mutated[0]:
                mutated[0]=True
                q['observations'][0]['response']=1
            return original(matrix)
        with patch.object(glm,'_QR',side_effect=qr):
            observed=glm.fit(q)
        self.assertTrue(mutated[0])
        self.assertEqual(observed['coefficients'],baseline['coefficients'])
        self.assertEqual(observed['input_sha256'],baseline['input_sha256'])

    def test_binary_logit_matches_grouped_bernoulli_closed_form(self):
        r=glm.fit(case()); self.assertEqual(r['status'],'CONVERGED')
        for a,b in zip(r['coefficients'],[-math.log(3),2*math.log(3)]): self.assertAlmostEqual(a,b,delta=1e-9)
        for row,expected in zip(r['covariance'],[[4/3,-4/3],[-4/3,8/3]]):
            for a,b in zip(row,expected): self.assertAlmostEqual(a,b,delta=1e-9)
        self.assertAlmostEqual(r['average_marginal_effects']['x'],3*math.log(3)/8,delta=1e-9)
        self.assertLess(r['scaled_score_norm'],1e-9)
        self.assertEqual(r['native_execution'],'NOT_RUN')
        self.assertNotEqual(r['coverage']['status'],'PASS')
        json.dumps(r,allow_nan=False)

    def test_poisson_counts_and_offset_have_known_solution(self):
        q=case('POISSON_COUNT'); r=glm.fit(q)
        for value in r['coefficients']: self.assertAlmostEqual(value,math.log(2),delta=1e-9)
        for row,expected in zip(r['covariance'],[[1/8,-1/8],[-1/8,3/16]]):
            for a,b in zip(row,expected): self.assertAlmostEqual(a,b,delta=1e-9)
        self.assertAlmostEqual(r['average_marginal_effects']['x'],3*math.log(2),delta=1e-9)
        for row in q['observations']: row['offset']=math.log(2)
        shifted=glm.fit(q)
        self.assertAlmostEqual(shifted['coefficients'][0],0,delta=1e-9)
        self.assertAlmostEqual(shifted['coefficients'][1],math.log(2),delta=1e-9)
        for a,b in zip(shifted['fitted_mean'],r['fitted_mean']): self.assertAlmostEqual(a,b,delta=1e-9)

    def test_two_regressors_and_varying_offsets_match_grouped_score_and_cofactor_oracle(self):
        q=case(); q['model']['regressors']=['x','z']; q['units']['z']='units'
        # Odds = exp(-ln2 + x*ln2 + z*ln3 + offset).
        groups=[(0,0,1,1,3),(1,0,2,2,3),(0,1,3,9,11),(1,1,4,12,13)]
        rows=[]; information=[[F(0) for _ in range(3)] for _ in range(3)]
        for x,z,exposure,successes,n in groups:
            design=[1,x,z]; probability=F(successes,n); weight=n*probability*(1-probability)
            for i in range(3):
                for j in range(3): information[i][j]+=weight*design[i]*design[j]
            for i in range(n):
                rows.append(dict(observation_id=str(len(rows)),response=int(i<successes),regressors={'x':x,'z':z},offset=math.log(exposure)))
        q['observations']=rows; r=glm.fit(q)
        self.assertEqual(r['status'],'CONVERGED')
        for actual,expected in zip(r['coefficients'],[-math.log(2),math.log(2),math.log(3)]):
            self.assertAlmostEqual(actual,expected,delta=1e-9)
        def minor(i,j):
            m=[[information[a][b] for b in range(3) if b!=j] for a in range(3) if a!=i]
            return (-1)**(i+j)*(m[0][0]*m[1][1]-m[0][1]*m[1][0])
        determinant=sum(information[0][j]*minor(0,j) for j in range(3))
        for i in range(3):
            for j in range(3): self.assertAlmostEqual(r['covariance'][i][j],float(minor(j,i)/determinant),delta=1e-9)

    def test_poisson_robust_covariance_uses_score_variation(self):
        q=case('POISSON_COUNT'); q['covariance']['estimator']='HC0'; r=glm.fit(q)
        for row,expected in zip(r['covariance'],[[5/32,-5/32],[-5/32,27/128]]):
            for a,b in zip(row,expected): self.assertAlmostEqual(a,b,delta=1e-9)
        q['covariance']['estimator']='HC1'; hc1=glm.fit(q)
        self.assertAlmostEqual(hc1['covariance'][1][1],(27/128)*4/3,delta=1e-9)

    def test_average_marginal_effect_delta_variance_matches_closed_forms(self):
        b=math.log(3); r=glm.fit(case()); effect=r['marginal_effect_inference']['effects']['x']
        derivative=3/16-3*b/32
        self.assertAlmostEqual(effect['estimate'],3*b/8,delta=1e-9)
        self.assertAlmostEqual(effect['gradient'][0],0,delta=1e-9)
        self.assertAlmostEqual(effect['gradient'][1],derivative,delta=1e-9)
        self.assertAlmostEqual(effect['standard_error']**2,derivative**2*8/3,delta=1e-9)
        self.assertEqual(effect['inference_status'],'CONDITIONAL_ASYMPTOTIC_REFERENCE')
        self.assertEqual(r['marginal_effect_inference']['scope'],'FIXED_EMPIRICAL_DESIGN_AND_OFFSETS_COEFFICIENT_UNCERTAINTY_ONLY')
        b=math.log(2); r=glm.fit(case('POISSON_COUNT')); effect=r['marginal_effect_inference']['effects']['x']
        self.assertAlmostEqual(effect['standard_error']**2,27/16+3*b*b/8,delta=1e-9)
        self.assertAlmostEqual(effect['gradient'][0],3*b,delta=1e-9)
        self.assertAlmostEqual(effect['gradient'][1],3+2*b,delta=1e-9)

    def test_marginal_effect_delta_variance_uses_selected_robust_and_cluster_covariance(self):
        q=case('POISSON_COUNT'); q['covariance']['estimator']='HC0'; b=math.log(2)
        e=glm.fit(q)['marginal_effect_inference']['effects']['x']
        self.assertAlmostEqual(e['standard_error']**2,243/128-9*b/32+3*b*b/8,delta=1e-9)
        q=case(); q['covariance']=dict(estimator='CLUSTER_CR1',clusters=dict(zip([str(i) for i in range(8)],['a','b','c','d']*2)))
        e=glm.fit(q)['marginal_effect_inference']['effects']['x']
        self.assertAlmostEqual(e['standard_error']**2,(3/16-3*math.log(3)/32)**2*224/81,delta=1e-9)

    def test_marginal_effect_jacobian_matches_independent_central_differences(self):
        for family in ('BINOMIAL_LOGIT','POISSON_COUNT','POISSON_PSEUDO'):
            q=case(family); q['model']['regressors']=['x','z']; q['units']['z']='units'
            for i,row in enumerate(q['observations']): row['regressors']['z']=i%3; row['offset']=i/10
            r=glm.fit(q); self.assertEqual(r['status'],'CONVERGED')
            design=[[1,row['regressors']['x'],row['regressors']['z']] for row in q['observations']]
            def effect(beta,j):
                slopes=[]
                for x,row in zip(design,q['observations']):
                    eta=sum(a*b for a,b in zip(x,beta))+row['offset']
                    if family=='BINOMIAL_LOGIT':
                        mean=1/(1+math.exp(-eta)); slopes.append(mean*(1-mean))
                    else: slopes.append(math.exp(eta))
                return beta[j]*sum(slopes)/len(slopes)
            for j,name in enumerate(('x','z'),1):
                gradient=r['marginal_effect_inference']['effects'][name]['gradient']
                for k in range(3):
                    plus=list(r['coefficients']);minus=list(plus);plus[k]+=1e-5;minus[k]-=1e-5
                    self.assertAlmostEqual(gradient[k],(effect(plus,j)-effect(minus,j))/2e-5,delta=2e-8)

    def test_marginal_effect_uncertainty_transforms_with_units_and_rejects_cancelled_scores(self):
        q=case('POISSON_COUNT'); q['covariance']['estimator']='HC1'
        a=glm.fit(q)['marginal_effect_inference']['effects']['x']
        for row in q['observations']: row['regressors']['x']*=1000
        b=glm.fit(q)['marginal_effect_inference']['effects']['x']
        self.assertAlmostEqual(a['estimate'],1000*b['estimate'],delta=1e-9)
        self.assertAlmostEqual(a['standard_error'],1000*b['standard_error'],delta=1e-9)
        self.assertAlmostEqual(a['p_value'],b['p_value'],delta=1e-9)
        q=case();q['covariance']=dict(estimator='CLUSTER_CR0',clusters={str(i):'a' if i<4 else 'b' for i in range(8)})
        e=glm.fit(q)['marginal_effect_inference']['effects']['x']
        self.assertIsNone(e['standard_error']);self.assertIsNone(e['p_value'])
        self.assertEqual(e['inference_status'],'COVARIANCE_NUMERICAL_RESOLUTION_LIMIT')

    def test_delta_method_scales_before_squaring_and_blocks_cancelled_quadratic_form(self):
        r=glm._delta_method(0.0,[1e200],[[1e-200]],[0.0],['CONDITIONAL_ASYMPTOTIC_REFERENCE'])
        self.assertAlmostEqual(r['standard_error']/1e100,1,delta=1e-12)
        r=glm._delta_method(1.0,[1.0,-1.0],[[1.0,1.0],[1.0,1.0]],[0.0,0.0],['CONDITIONAL_ASYMPTOTIC_REFERENCE']*2)
        self.assertIsNone(r['standard_error'])
        self.assertEqual(r['inference_status'],'COVARIANCE_NUMERICAL_RESOLUTION_LIMIT')
        json.dumps(r,allow_nan=False)

    def test_delta_method_retains_finite_standard_error_when_variance_overflows(self):
        r=glm._delta_method(0.0,[1e200],[[1.0]],[0.0],['CONDITIONAL_ASYMPTOTIC_REFERENCE'])
        self.assertAlmostEqual(r['standard_error']/1e200,1,delta=1e-12)
        self.assertIsNone(r['variance'])
        self.assertEqual(r['variance_numeric_status'],'OUTSIDE_BINARY64_RANGE')
        self.assertEqual(r['inference_status'],'CONDITIONAL_ASYMPTOTIC_REFERENCE')
        json.dumps(r,allow_nan=False)

    def test_marginal_effect_gradient_scales_weights_before_cancelling_large_terms(self):
        q=case('POISSON_COUNT');q['model']=dict(outcome='outcome',regressors=['x','z'],intercept=False)
        x=[[1.0,1e6],[1.0,-1e6],[-1.0,1e6],[-1.0,-1e6]]
        r=glm._marginal_effect_inference(q,x,[700.0]*4,[1e304]*4,[1.0,0.0],[[1e-304,0.0],[0.0,1e-316]],
                                      [0.0,0.0],['CONDITIONAL_ASYMPTOTIC_REFERENCE']*2)
        e=r['effects']['x'];self.assertEqual(e['gradient'],[1e304,0.0])
        self.assertEqual(e['estimate'],1e304)
        json.dumps(r,allow_nan=False)

    def test_zero_effect_has_nonzero_delta_uncertainty_and_unconverged_fit_has_none(self):
        q=case()
        for i,row in enumerate(q['observations']): row['response']=int(i%4>=2)
        effect=glm.fit(q)['marginal_effect_inference']['effects']['x']
        self.assertAlmostEqual(effect['estimate'],0,delta=1e-12)
        self.assertAlmostEqual(effect['standard_error']**2,1/8,delta=1e-12)
        self.assertAlmostEqual(effect['p_value'],1,delta=1e-12)
        q=case('POISSON_COUNT');q['optimizer']['max_iterations']=1
        self.assertIsNone(glm.fit(q)['marginal_effect_inference'])

    def test_binary_discrete_changes_differ_from_derivatives_and_match_closed_forms(self):
        r=glm.discrete_changes(case(),['x']); effect=r['effects']['x']
        self.assertAlmostEqual(effect['mean_at_zero'],.25,delta=1e-9)
        self.assertAlmostEqual(effect['mean_at_one'],.75,delta=1e-9)
        self.assertAlmostEqual(effect['estimate'],.5,delta=1e-9)
        self.assertAlmostEqual(effect['standard_error']**2,3/32,delta=1e-9)
        self.assertAlmostEqual(effect['gradient'][0],0,delta=1e-9)
        self.assertAlmostEqual(effect['gradient'][1],3/16,delta=1e-9)
        self.assertNotAlmostEqual(effect['estimate'],r['fit']['average_marginal_effects']['x'],delta=1e-3)
        self.assertEqual(r['structural_causal_identification'],'NOT_ESTABLISHED')
        e=glm.discrete_changes(case('POISSON_COUNT'),['x'])['effects']['x']
        self.assertAlmostEqual(e['estimate'],2,delta=1e-9)
        self.assertAlmostEqual(e['standard_error']**2,1.5,delta=1e-9)
        for actual,expected in zip(e['gradient'],[2,4]): self.assertAlmostEqual(actual,expected,delta=1e-9)
        json.dumps(r,allow_nan=False)

    def test_discrete_change_jacobian_with_other_covariates_and_offsets(self):
        for family in ('BINOMIAL_LOGIT','POISSON_COUNT','POISSON_PSEUDO'):
            q=case(family);q['model']['regressors']=['x','z'];q['units']['z']='units'
            for i,row in enumerate(q['observations']): row['regressors']['z']=i%3;row['offset']=i/10
            r=glm.discrete_changes(q,['x']);effect=r['effects']['x'];beta=r['fit']['coefficients']
            def contrast(b):
                diffs=[]
                for row in q['observations']:
                    base=b[0]+b[2]*row['regressors']['z']+row['offset']
                    def mean(v): return 1/(1+math.exp(-v)) if family=='BINOMIAL_LOGIT' else math.exp(v)
                    diffs.append(mean(base+b[1])-mean(base))
                return sum(diffs)/len(diffs)
            self.assertAlmostEqual(effect['estimate'],contrast(beta),delta=1e-9)
            for j in range(3):
                plus=list(beta);minus=list(beta);plus[j]+=1e-5;minus[j]-=1e-5
                self.assertAlmostEqual(effect['gradient'][j],(contrast(plus)-contrast(minus))/2e-5,delta=2e-8)

    def test_discrete_changes_use_cluster_covariance_and_bind_requested_variables(self):
        q=case();q['covariance']=dict(estimator='CLUSTER_CR1',clusters=dict(zip([str(i) for i in range(8)],['a','b','c','d']*2)))
        r=glm.discrete_changes(q,['x']); self.assertAlmostEqual(r['effects']['x']['standard_error']**2,7/72,delta=1e-9)
        self.assertNotEqual(r['input_sha256'],r['fit']['input_sha256'])
        self.assertEqual(r['scope'],'FIXED_EMPIRICAL_DESIGN_AND_OFFSETS_COEFFICIENT_UNCERTAINTY_ONLY')

    def test_discrete_change_validation_and_nonconvergence_are_explicit(self):
        q=case()
        for names in ([],['unknown'],['x','x'],'x'):
            with self.assertRaisesRegex(ValueError,'DISCRETE_CHANGE_REGRESSORS'): glm.discrete_changes(q,names)
        for row in q['observations']: row['regressors']['x']*=2
        with self.assertRaisesRegex(ValueError,'BINARY_REGRESSOR'): glm.discrete_changes(q,['x'])
        q=case('POISSON_COUNT');q['optimizer']['max_iterations']=1
        with self.assertRaisesRegex(ValueError,'CONVERGED'): glm.discrete_changes(q,['x'])

    def test_discrete_change_stable_differences_retain_small_and_saturated_changes(self):
        p0,p1,change,_,derivative_change=glm._binary_change_at(0.0,1e-18,'BINOMIAL_LOGIT')
        self.assertEqual(p0,p1)
        self.assertAlmostEqual(change/2.5e-19,1,delta=1e-12)
        self.assertAlmostEqual(derivative_change/-6.25e-38,1,delta=1e-12)
        p0,p1,change,_,_=glm._binary_change_at(50.0,1.0,'BINOMIAL_LOGIT')
        self.assertEqual(p0,p1)
        self.assertAlmostEqual(change/(math.exp(-50)*(-math.expm1(-1))),1,delta=1e-12)
        forward=glm._binary_change_at(-2.0,3.0,'BINOMIAL_LOGIT')[2]
        backward=glm._binary_change_at(1.0,-3.0,'BINOMIAL_LOGIT')[2]
        self.assertAlmostEqual(forward,-backward,delta=1e-15)
        p0,p1,change,_,_=glm._binary_change_at(math.log(2),1e-18,'POISSON_COUNT')
        self.assertEqual(p0,p1)
        self.assertAlmostEqual(change/2e-18,1,delta=1e-12)
        with self.assertRaisesRegex(ValueError,'DISCRETE_CHANGE_MEAN_NUMERIC_RANGE'): glm._binary_change_at(700.0,1.0,'POISSON_COUNT')

    def test_ppml_noninteger_response_rescaling_transforms_both_effect_uncertainties(self):
        q=case('POISSON_PSEUDO');q['covariance']['estimator']='HC0'
        continuous=glm.fit(q)['marginal_effect_inference']['effects']['x']
        discrete=glm.discrete_changes(q,['x'])['effects']['x']
        for row in q['observations']: row['response']=str(F(row['response'],2))
        fitted=glm.fit(q);scaled_continuous=fitted['marginal_effect_inference']['effects']['x']
        scaled_discrete=glm.discrete_changes(q,['x'])['effects']['x']
        for base,scaled in ((continuous,scaled_continuous),(discrete,scaled_discrete)):
            self.assertAlmostEqual(scaled['estimate']*2,base['estimate'],delta=1e-9)
            self.assertAlmostEqual(scaled['standard_error']*2,base['standard_error'],delta=1e-9)
            self.assertAlmostEqual(scaled['p_value'],base['p_value'],delta=1e-9)
        self.assertIsNone(fitted['log_likelihood']);self.assertIsNone(fitted['aic'])

    def test_discrete_changes_preserve_input_and_bind_variable_order(self):
        q=case('POISSON_COUNT');q['model']['regressors']=['x','z'];q['units']['z']='indicator'
        for i,row in enumerate(q['observations']): row['regressors']['z']=i%2
        original=copy.deepcopy(q);a=glm.discrete_changes(q,['x','z']);b=glm.discrete_changes(q,['z','x'])
        self.assertEqual(q,original)
        self.assertNotEqual(a['input_sha256'],b['input_sha256'])
        self.assertEqual(a['effects']['x'],b['effects']['x'])
        q=case();q['covariance']=dict(estimator='CLUSTER_CR0',clusters={str(i):'a' if i<4 else 'b' for i in range(8)})
        e=glm.discrete_changes(q,['x'])['effects']['x']
        self.assertIsNone(e['standard_error']);self.assertIsNone(e['p_value'])

    def test_cluster_logit_sandwich_matches_group_totals(self):
        q=case(); q['covariance']=dict(estimator='CLUSTER_CR0',clusters=dict(zip([str(i) for i in range(8)],['a','b','c','d']*2)))
        r=glm.fit(q)
        for row,expected in zip(r['covariance'],[[4/3,-8/9],[-8/9,16/9]]):
            for a,b in zip(row,expected): self.assertAlmostEqual(a,b,delta=1e-9)
        q['covariance']['estimator']='CLUSTER_CR1'
        self.assertAlmostEqual(glm.fit(q)['covariance'][1][1],224/81,delta=1e-9)

    def test_ppml_accepts_noninteger_response_without_count_likelihood_claim(self):
        q=case('POISSON_PSEUDO')
        for row in q['observations']: row['response']=str(F(row['response'],2))
        r=glm.fit(q)
        self.assertAlmostEqual(r['coefficients'][0],0,delta=1e-9)
        self.assertAlmostEqual(r['coefficients'][1],math.log(2),delta=1e-9)
        self.assertIsNone(r['aic'])
        q['family']='POISSON_COUNT'
        with self.assertRaisesRegex(ValueError,'INTEGER_COUNTS'): glm.fit(q)

    def test_separation_is_certified_by_exact_margins(self):
        for quasi in (False,True):
            q=case(); xs=[-2,-1,0,0,1,2] if quasi else [-3,-2,-1,1,2,3]
            ys=[0,0,0,1,1,1]
            q['observations']=[dict(observation_id=str(i),response=y,regressors={'x':x},offset=0) for i,(x,y) in enumerate(zip(xs,ys))]
            r=glm.fit(q)
            self.assertEqual(r['status'],'NO_FINITE_MLE_CERTIFIED')
            self.assertIsNone(r['coefficients']); self.assertIsNone(r['covariance'])
            direction=list(map(F,r['separation_certificate']['direction_exact']))
            margins=[(2*y-1)*(direction[0]+x*direction[1]) for x,y in zip(xs,ys)]
            self.assertTrue(all(v>=0 for v in margins)); self.assertTrue(any(v>0 for v in margins))
            self.assertEqual(r['separation_certificate']['kind'],'QUASI_COMPLETE' if quasi else 'COMPLETE')

    def test_poisson_zero_response_has_no_finite_intercept_but_no_intercept_can_be_finite(self):
        q=case('POISSON_COUNT')
        for row in q['observations']: row['response']=0
        self.assertEqual(glm.fit(q)['status'],'NO_FINITE_MLE_CERTIFIED')
        q['model']['intercept']=False
        for i,row in enumerate(q['observations']): row['regressors']['x']=(-1)**i
        r=glm.fit(q); self.assertEqual(r['status'],'CONVERGED')
        self.assertAlmostEqual(r['coefficients'][0],0,delta=1e-12)

    def test_iteration_limit_does_not_return_mle_or_separation_claim(self):
        q=case('POISSON_COUNT'); q['optimizer']['max_iterations']=1
        r=glm.fit(q)
        self.assertEqual(r['status'],'NOT_CONVERGED'); self.assertIsNone(r['coefficients'])
        self.assertIsNone(r['separation_certificate'])
        self.assertAlmostEqual(r['diagnostic_iterate'][0],math.log(3),delta=1e-12)
        self.assertAlmostEqual(r['diagnostic_iterate'][1],0,delta=1e-12)
        self.assertAlmostEqual(r['scaled_score_norm'],4/math.sqrt(12),delta=1e-12)

    def test_rights_rank_sample_and_response_failures_are_explicit(self):
        changes=[('RIGHTS',lambda q:q['rights'].update(local_processing='DENIED')),
                 ('RESPONSE',lambda q:q['observations'][0].update(response=2)),
                 ('OBSERVATION',lambda q:q['observations'][0].update(observation_id='1')),
                 ('RANK',lambda q:[row['regressors'].update(x=1) for row in q['observations']]),
                 ('NUMBER',lambda q:q['observations'][0].update(offset=float('nan'))),
                 ('OPTIMIZER',lambda q:q['optimizer'].update(tolerance=1))]
        for reason,mutate in changes:
            q=case(); mutate(q)
            with self.subTest(reason=reason),self.assertRaisesRegex(ValueError,reason): glm.fit(q)

    def test_large_finite_offset_does_not_overflow_unscaled_score(self):
        q=case('POISSON_COUNT'); q['model']['intercept']=False; q['optimizer']['max_iterations']=1
        for row in q['observations']: row.update(response=1,regressors={'x':1000000},offset=700)
        r=glm.fit(q)
        self.assertEqual(r['status'],'NOT_CONVERGED')
        self.assertTrue(math.isfinite(r['scaled_score_norm']))
        self.assertAlmostEqual(r['last_iterate'][0],-1e-6,delta=1e-18)
        json.dumps(r,allow_nan=False)

    def test_poisson_positive_response_constraints_are_exact_in_separation_certificate(self):
        q=case('POISSON_COUNT')
        xs=[0,0,0,1,2,3]; ys=[1,2,3,0,0,0]
        q['observations']=[dict(observation_id=str(i),response=y,regressors={'x':x},offset=0) for i,(x,y) in enumerate(zip(xs,ys))]
        r=glm.fit(q); self.assertEqual(r['status'],'NO_FINITE_MLE_CERTIFIED')
        b=list(map(F,r['separation_certificate']['direction_exact']))
        for x,y in zip(xs,ys):
            if y: self.assertEqual(b[0]+b[1]*x,0)
            else: self.assertLessEqual(b[0]+b[1]*x,0)

    def test_row_order_and_design_unit_scaling_preserve_estimates(self):
        q=case(); r=glm.fit(q); q['observations'].reverse()
        for row in q['observations']: row['regressors']['x']*=1000
        changed=glm.fit(q)
        self.assertAlmostEqual(changed['coefficients'][0],r['coefficients'][0],delta=1e-9)
        self.assertAlmostEqual(changed['coefficients'][1]*1000,r['coefficients'][1],delta=1e-9)
        self.assertAlmostEqual(changed['covariance'][1][1]*1e6,r['covariance'][1][1],delta=1e-8)

    def test_fit_does_not_mutate_input_and_reports_binary64_reference(self):
        q=case(); original=copy.deepcopy(q); r=glm.fit(q)
        self.assertEqual(q,original)
        self.assertEqual(r['numerical_arithmetic'],'BINARY64_NOT_INTERVAL_CERTIFIED')
        self.assertIn('numeric_linear.py',r['code_sha256'])

    def test_prediction_mean_uncertainty_is_not_observation_prediction_uncertainty(self):
        q=case(); rows=[dict(observation_id='new',regressors={'x':0},offset=0)]
        r=glm.predict(q,rows); point=r['predictions'][0]
        self.assertAlmostEqual(point['mean'],.25,delta=1e-10)
        self.assertAlmostEqual(point['linear_predictor_standard_error'],math.sqrt(4/3),delta=1e-9)
        self.assertAlmostEqual(point['mean_standard_error'],math.sqrt(3)/8,delta=1e-9)
        self.assertEqual(r['interval_scope'],'CONDITIONAL_MEAN_CONFIDENCE_NOT_NEW_OBSERVATION_PREDICTION')
        q=case('POISSON_COUNT'); point=glm.predict(q,rows)['predictions'][0]
        self.assertAlmostEqual(point['mean'],2,delta=1e-9)
        self.assertAlmostEqual(point['mean_standard_error'],math.sqrt(.5),delta=1e-9)

    def test_predictions_bind_new_design_and_reject_unconverged_fit(self):
        q=case(); rows=[dict(observation_id='new',regressors={'x':0},offset=0)]
        a=glm.predict(q,rows); rows[0]['regressors']['x']=1; b=glm.predict(q,rows)
        self.assertNotEqual(a['input_sha256'],b['input_sha256'])
        self.assertAlmostEqual(b['predictions'][0]['mean'],.75,delta=1e-9)
        q=case('POISSON_COUNT'); q['optimizer']['max_iterations']=1
        with self.assertRaisesRegex(ValueError,'CONVERGED'): glm.predict(q,rows)

    def test_model_capabilities_expose_only_supported_glm_domains(self):
        from econscope.study_capabilities import negotiate_capabilities
        cap=negotiate_capabilities(['binary_logit_poisson'])['glm_capabilities']
        self.assertIn('econscope.generalized_linear.fit',cap['python'])
        self.assertEqual(cap['chat_ui'],'NOT_CONNECTED')
        self.assertEqual(cap['native_upstream_parity'],'NOT_RUN')
        self.assertIn('FIXED_EMPIRICAL_DESIGN',cap['marginal_effect_inference'])
        self.assertNotIn('MARGINAL_EFFECT_STANDARD_ERRORS',cap['unsupported'])
        self.assertIn('UNCONDITIONAL_MARGINAL_EFFECT_VARIANCE',cap['unsupported'])
        self.assertIn('OFFSET_NOT_IMPLEMENTED',cap['upstream_marginal_effect_offsets'])

    def test_finite_counterfactual_mean_is_retained_when_uncertainty_overflows(self):
        q=case('POISSON_COUNT')
        r=glm.predict(q,[dict(observation_id='large',regressors={'x':1000000},offset=-692448)])
        point=r['predictions'][0]
        self.assertTrue(math.isfinite(point['mean']))
        self.assertIsNone(point['mean_standard_error'])
        self.assertEqual(point['inference_status'],'MEAN_UNCERTAINTY_NUMERIC_RANGE')
        json.dumps(r,allow_nan=False)

    def test_cancelled_cluster_scores_do_not_produce_certain_parameter_inference(self):
        q=case(); q['covariance']=dict(estimator='CLUSTER_CR0',clusters={str(i):'a' if i<4 else 'b' for i in range(8)})
        r=glm.fit(q)
        self.assertEqual(r['status'],'CONVERGED')
        self.assertTrue(all(value is None for value in r['p_values']))

    def test_unrepresentable_parameter_interval_is_not_reported_as_certain_inference(self):
        q=case('POISSON_COUNT'); q['model']['intercept']=False
        q['observations']=[dict(observation_id=str(i),response=0,regressors={'x':(-1)**i},offset=699-(-1)**i) for i in range(4)]
        r=glm.fit(q)
        self.assertEqual(r['status'],'CONVERGED')
        self.assertAlmostEqual(r['coefficients'][0],1,delta=1e-12)
        self.assertIsNone(r['confidence_intervals_95'][0])
        self.assertIsNone(r['p_values'][0])
        self.assertEqual(r['parameter_inference_status'][0],'PARAMETER_INTERVAL_NUMERICAL_RESOLUTION_LIMIT')
        json.dumps(r,allow_nan=False)

    def test_saturated_prediction_mean_does_not_report_zero_width_confidence_interval(self):
        point=glm.predict(case(),[dict(observation_id='saturated',regressors={'x':0},offset=50)])['predictions'][0]
        self.assertEqual(point['mean'],1)
        self.assertTrue(math.isfinite(point['mean_standard_error']))
        self.assertIsNone(point['mean_confidence_interval_95'])
        self.assertEqual(point['inference_status'],'MEAN_INTERVAL_NUMERICAL_RESOLUTION_LIMIT')

    def test_covariance_resolution_estimate_transforms_with_regressor_units(self):
        q=case(); ids=[str(i) for i in range(8)]
        q['covariance']=dict(estimator='CLUSTER_CR0',clusters=dict(zip(ids,['a','b','c','d']*2)))
        x=[[1.0,float(i>=4)] for i in range(8)]; y=[float(row['response']) for row in q['observations']]
        means=[.25]*4+[.75]*4; residuals=[a-b for a,b in zip(y,means)]; info=[[4/3,-4/3],[-4/3,8/3]]
        _,baseline=glm._covariance(q,x,y,means,residuals,info,ids,[3/16]*8,[1e-4]*8)
        scaled_x=[[1.0,row[1]*1000] for row in x]
        scaled_info=[[info[i][j]/([1,1000][i]*[1,1000][j]) for j in range(2)] for i in range(2)]
        _,scaled=glm._covariance(q,scaled_x,y,means,residuals,scaled_info,ids,[3/16]*8,[1e-4]*8)
        self.assertAlmostEqual(scaled[0]/baseline[0],1,delta=1e-12)
        self.assertAlmostEqual(scaled[1]*1e6/baseline[1],1,delta=1e-12)


if __name__=='__main__': unittest.main()
