"""Known recurrences, Lyapunov witnesses and exact recursive variance accounting."""
import copy
from decimal import localcontext,Inexact,ROUND_UP
from fractions import Fraction as F
import json
import unittest
from unittest.mock import patch

from econscope import vector_autoregression as var


def case(stochastic=False):
    variables=['x','y']
    if stochastic:
        values=[[1,0],[0,0],[1,1],[0,0],[1,2],[0,0],[2,1],[0,0],[1,0],[0,0]]
    else:
        values=[[F(0),F(0)]]
        for _ in range(10):
            x,y=values[-1]; values.append([1+x/2+y/4,-1+y/3])
    return dict(schema='societybox-vector-autoregression-v1',variables=variables,
        observations=[dict(period=i,values={name:str(value) for name,value in zip(variables,row)}) for i,row in enumerate(values)],
        period_unit='quarter',estimation_end_period=len(values)-1,
        model=dict(lags=1,intercept=not stochastic,covariance_divisor='RESIDUAL_DF'),
        identification=dict(scheme='RECURSIVE_CHOLESKY',ordering=variables) if stochastic else None,
        units={'x':'output units','y':'price units'},source_declared_sha256='a'*64,
        rights=dict(local_processing='PERMITTED',report_export='PERMITTED'))


class VectorAutoregressionTests(unittest.TestCase):
    def test_var_receipt_is_bound_to_fitted_snapshot_not_later_caller_mutations(self):
        q=case();baseline=var.calculate(copy.deepcopy(q));original=var._fit
        def mutate(spec):
            fitted=original(spec)
            q['observations'][0]['values']['x']='99999'
            return fitted
        with patch.object(var,'_fit',side_effect=mutate):
            observed=var.calculate(q)
        self.assertEqual(observed['lag_matrices_exact'],baseline['lag_matrices_exact'])
        self.assertEqual(observed['input_sha256'],baseline['input_sha256'])

    def test_known_var_coefficients_and_exact_lyapunov_stability(self):
        r=var.calculate(case())
        self.assertEqual(r['lag_matrices_exact'],[[['1/2','1/4'],['0','1/3']]])
        self.assertEqual(r['intercept_exact'],['1','-1'])
        self.assertEqual(r['stability']['status'],'SCHUR_STABLE')
        self.assertEqual(r['stability']['lyapunov_solution_exact'],[['4/3','1/5'],['1/5','201/160']])
        self.assertEqual(r['normal_equations_exact'],[['0','0']]*3)
        self.assertEqual(r['residual_covariance_exact'],[['0','0'],['0','0']])
        self.assertFalse(r['innovation_covariance_positive_definite'])
        self.assertEqual(r['native_execution'],'NOT_RUN')
        self.assertNotEqual(r['coverage']['status'],'PASS')

    def test_forecast_recursion_uses_declared_origin_without_future_leakage(self):
        q=case(); r=var.forecast(q,3)
        last=q['observations'][-1]['values']; x,y=F(last['x']),F(last['y']); expected=[]
        for _ in range(3):
            x,y=1+x/2+y/4,-1+y/3; expected.append([str(x),str(y)])
        self.assertEqual(r['means_exact'],expected)
        self.assertEqual(r['periods'],[11,12,13])
        changed=copy.deepcopy(q); changed['observations'].append(dict(period=11,values={'x':'10000','y':'-10000'}))
        future=var.forecast(changed,3)
        self.assertEqual(future['means_exact'],expected)
        self.assertNotEqual(future['input_sha256'],r['input_sha256'])

    def test_reduced_form_impulses_are_matrix_powers_not_transposed(self):
        r=var.impulse_responses(case(),3)
        self.assertEqual(r['ma_matrices_exact'],[
            [['1','0'],['0','1']], [['1/2','1/4'],['0','1/3']],
            [['1/4','5/24'],['0','1/9']], [['1/8','19/144'],['0','1/27']]])

    def test_residual_df_and_ml_covariance_have_declared_divisors(self):
        q=case(stochastic=True); r=var.calculate(q)
        self.assertEqual(r['lag_matrices_exact'],[[['0','0'],['0','0']]])
        self.assertEqual(r['residual_degrees_of_freedom'],7)
        self.assertEqual(r['residual_covariance_exact'],[['1','5/7'],['5/7','6/7']])
        q['model']['covariance_divisor']='ML'
        self.assertEqual(var.calculate(q)['residual_covariance_exact'],[['7/9','5/9'],['5/9','2/3']])

    def test_recursive_fevd_conserves_forecast_variance_and_records_order(self):
        q=case(stochastic=True); r=var.variance_decomposition(q,3)
        expected=[['1','0'],['25/42','17/42']]
        self.assertEqual(r['shares_exact'],[expected]*3)
        self.assertEqual(r['shock_order'],['x','y'])
        forecast=var.forecast(q,3)
        self.assertEqual(forecast['covariances_exact'],[[['1','5/7'],['5/7','6/7']]]*3)
        for horizon in r['shares_exact']:
            self.assertTrue(all(sum(map(F,row))==1 for row in horizon))
        q['identification']['ordering']=['y','x']
        reversed_order=var.variance_decomposition(q,1)
        self.assertEqual(reversed_order['shares_exact'],[[['25/42','17/42'],['1','0']]])

    def test_recursive_impulses_preserve_algebraic_scale_and_do_not_claim_causality(self):
        r=var.impulse_responses(case(stochastic=True),2,orthogonalized=True)
        self.assertEqual(r['orthogonalized_impulses'][0][1][0]['multiplier_exact'],'5/7')
        self.assertEqual(r['orthogonalized_impulses'][0][1][1]['innovation_variance_exact'],'17/49')
        self.assertEqual(r['structural_causal_identification'],'NOT_ESTABLISHED')
        json.dumps(r,allow_nan=False)

    def test_unstable_or_unit_root_fit_is_reported_but_dynamics_are_blocked(self):
        for unit_root in (False,True):
            q=case(); q['variables']=['x']; q['units']={'x':'units'}
            q['observations']=[dict(period=i,values={'x':str(i+1 if unit_root else 2**i)}) for i in range(10)]
            q['estimation_end_period']=9; q['model']['intercept']=unit_root
            r=var.calculate(q)
            self.assertEqual(r['stability']['status'],'NOT_SCHUR_STABLE')
            with self.assertRaisesRegex(ValueError,'NONSTATIONARY'): var.forecast(q,2)
            with self.assertRaisesRegex(ValueError,'NONSTATIONARY'): var.impulse_responses(q,2)

    def test_missing_periods_rank_failure_and_undeclared_shock_identification_fail(self):
        q=case(); q['observations'].pop(4)
        with self.assertRaisesRegex(ValueError,'CONTIGUOUS'): var.calculate(q)
        q=case(); q['observations']=[dict(period=i,values={'x':str(i),'y':str(i)}) for i in range(11)]
        with self.assertRaisesRegex(ValueError,'RANK'): var.calculate(q)
        with self.assertRaisesRegex(ValueError,'IDENTIFICATION_REQUIRED'): var.variance_decomposition(case(),2)
        q=case(); q['identification']=dict(scheme='RECURSIVE_CHOLESKY',ordering=['x','y'])
        with self.assertRaisesRegex(ValueError,'POSITIVE_DEFINITE'): var.variance_decomposition(q,2)

    def test_schema_rights_covariance_and_horizon_fail_closed(self):
        changes=[('RIGHTS',lambda q:q['rights'].update(local_processing='DENIED')),
                 ('MODEL',lambda q:q['model'].update(lags=True)),
                 ('UNITS',lambda q:q['units'].pop('x')),
                 ('NUMBER',lambda q:q['observations'][0]['values'].update(x='NaN')),
                 ('CUTOFF',lambda q:q.update(estimation_end_period=99))]
        for reason,mutate in changes:
            q=case(); mutate(q)
            with self.subTest(reason=reason),self.assertRaisesRegex(ValueError,reason): var.calculate(q)
        for horizon in (0,True,65):
            with self.assertRaisesRegex(ValueError,'HORIZON'): var.forecast(case(),horizon)

    def test_ar2_lags_and_multivariate_companion_are_not_ar1_shortcuts(self):
        q=case(); q['model']['lags']=2; values=[[F(0),F(1)],[F(1),F(0)]]
        for _ in range(10):
            x,y=values[-1]; px,py=values[-2]
            values.append([1+x/2+px/4,-1+y/4+py/8])
        q['observations']=[dict(period=i,values={name:str(v) for name,v in zip(q['variables'],row)}) for i,row in enumerate(values)]
        q['estimation_end_period']=len(values)-1
        r=var.calculate(q)
        self.assertEqual(r['lag_matrices_exact'],[[['1/2','0'],['0','1/4']],[['1/4','0'],['0','1/8']]])
        self.assertEqual(r['stability']['status'],'SCHUR_STABLE')
        response=var.impulse_responses(q,3)['ma_matrices_exact']
        self.assertEqual(response[2],[['1/2','0'],['0','3/16']])
        self.assertEqual(response[3],[['3/8','0'],['0','5/64']])
        x,y=values[-1]; px,py=values[-2]
        self.assertEqual(var.forecast(q,1)['means_exact'],[[str(1+x/2+px/4),str(-1+y/4+py/8)]])

    def test_nonzero_dynamics_propagate_covariance_and_recursive_shares(self):
        q=case(stochastic=True); q['observations'][-1]['values']={'x':'1','y':'1'}
        r=var.calculate(q)
        self.assertEqual(r['lag_matrices_exact'],[[['6/23','-5/23'],['6/23','-5/23']]])
        self.assertEqual(r['residual_covariance_exact'],[['178/161','132/161'],['132/161','155/161']])
        extra=F(2363,85169)
        expected=[[F(178,161)+extra,F(132,161)+extra],[F(132,161)+extra,F(155,161)+extra]]
        self.assertEqual([[F(v) for v in row] for row in var.forecast(q,2)['covariances_exact'][1]],expected)
        decomposition=var.variance_decomposition(q,2)
        self.assertEqual(decomposition['shares_exact'][0],[['1','0'],['8712/13795','5083/13795']])
        d1,d2=F(178,161),F(5083,14329); b=F(204,2047)
        expected_contribution=[[d1*(1+b*b),d2*F(25,529)],
                               [d1*(F(66,89)**2+b*b),d2*(1+F(25,529))]]
        self.assertEqual([[F(v) for v in row] for row in decomposition['variance_contributions_exact'][1]],expected_contribution)

    def test_scalar_forecast_variance_is_the_ma_variance_sum(self):
        q=case(); q['variables']=['x']; q['units']={'x':'units'}
        q['model']['intercept']=False; q['identification']=dict(scheme='RECURSIVE_CHOLESKY',ordering=['x'])
        q['observations']=[dict(period=i,values={'x':str(y)}) for i,y in enumerate([0,1,1,0,0,0])]
        q['estimation_end_period']=5
        self.assertEqual(var.calculate(q)['lag_matrices_exact'],[[['1/2']]])
        self.assertEqual(var.forecast(q,3)['covariances_exact'],[[['3/8']],[['15/32']],[['63/128']]])
        self.assertEqual(var.variance_decomposition(q,3)['shares_exact'],[[['1']]]*3)

    def test_future_gaps_and_row_permutations_do_not_change_training_fit(self):
        q=case(stochastic=True); original=var.forecast(q,2)
        q['observations'].append(dict(period=20,values={'x':'5000','y':'-5000'})); q['observations'].reverse()
        changed=var.forecast(q,2)
        self.assertEqual(changed['means_exact'],original['means_exact'])
        self.assertEqual(changed['covariances_exact'],original['covariances_exact'])

    def test_receipts_do_not_alias_mutable_input_variable_or_shock_lists(self):
        q=case(stochastic=True); result=var.calculate(q); impulse=var.impulse_responses(q,1,orthogonalized=True)
        before=json.dumps(result,sort_keys=True); before_impulse=json.dumps(impulse,sort_keys=True)
        q['variables'][0]='changed'; q['identification']['ordering'].reverse()
        self.assertEqual(json.dumps(result,sort_keys=True),before)
        self.assertEqual(json.dumps(impulse,sort_keys=True),before_impulse)

    def test_validate_is_not_estimation_and_core_is_discoverable_without_ui_claim(self):
        with patch('econscope.vector_autoregression._fit',side_effect=AssertionError('estimate during review')):
            var.validate(case())
        from econscope.study_capabilities import negotiate_capabilities
        caps=negotiate_capabilities(['exact_var_dynamics'])['var_capabilities']
        self.assertIn('econscope.vector_autoregression.forecast',caps['python'])
        self.assertEqual(caps['chat_ui'],'NOT_CONNECTED')
        self.assertEqual(caps['native_upstream_parity'],'NOT_RUN')

    def test_historical_components_account_for_observations_and_depend_on_recursive_order(self):
        q=case(stochastic=True); r=var.historical_decomposition(q,orthogonalized=True)
        index=r['periods'].index(2)
        self.assertEqual(r['baseline_exact'][index],['0','0'])
        self.assertEqual(r['contributions_exact'][index],[['1','0'],['5/7','2/7']])
        self.assertEqual(r['maximum_accounting_residual_exact'],'0')
        self.assertEqual(r['information_scope'],'EX_POST_FULL_ESTIMATION_SAMPLE')
        self.assertEqual(r['structural_causal_identification'],'NOT_ESTABLISHED')
        q['identification']['ordering']=['y','x']
        reverse=var.historical_decomposition(q,orthogonalized=True)
        self.assertEqual(reverse['contributions_exact'][index],[['5/6','1/6'],['1','0']])
        self.assertEqual(reverse['observed_exact'],r['observed_exact'])

    def test_historical_baseline_and_propagated_shocks_are_exact_for_nonzero_dynamics(self):
        q=case(stochastic=True); q['observations'][-1]['values']={'x':'1','y':'1'}
        r=var.historical_decomposition(q,orthogonalized=True)
        self.assertEqual(r['baseline_exact'][0],['6/23','6/23'])
        self.assertEqual(r['contributions_exact'][0],[['-6/23','0'],['-396/2047','-6/89']])
        for baseline,components,observed in zip(r['baseline_exact'],r['contributions_exact'],r['observed_exact']):
            self.assertEqual([F(b)+sum(map(F,row)) for b,row in zip(baseline,components)],list(map(F,observed)))
        deterministic=var.historical_decomposition(case())
        self.assertEqual(deterministic['baseline_exact'],deterministic['observed_exact'])
        self.assertTrue(all(F(v)==0 for matrix in deterministic['contributions_exact'] for row in matrix for v in row))

    def test_decimal_irf_display_does_not_inherit_callers_rounding_or_traps(self):
        q=case(stochastic=True); before=var.impulse_responses(q,1,orthogonalized=True)
        with localcontext() as context:
            context.prec=4; context.rounding=ROUND_UP; context.traps[Inexact]=True
            after=var.impulse_responses(q,1,orthogonalized=True)
        self.assertEqual(before,after)

    def test_three_variable_var2_lyapunov_witness_matches_finite_power_sum(self):
        q=case(); q['variables']=['x','y','z']; q['units']={name:'units' for name in q['variables']}
        q['model']['lags']=2; values=[[F(1),F(2),F(3)],[F(4),F(5),F(6)]]
        for _ in range(10):
            older=values[-2]; values.append([1+older[1],-1+older[2],F(2)])
        q['observations']=[dict(period=i,values={name:str(v) for name,v in zip(q['variables'],row)}) for i,row in enumerate(values)]
        q['estimation_end_period']=11
        r=var.calculate(q)
        zero=[['0']*3 for _ in range(3)]; jordan=[['0','1','0'],['0','0','1'],['0','0','0']]
        self.assertEqual(r['lag_matrices_exact'],[zero,jordan])
        self.assertEqual(r['intercept_exact'],['1','-1','2'])
        response=var.impulse_responses(q,8)['ma_matrices_exact']
        self.assertEqual(response[2],jordan); self.assertEqual(response[4],[['0','0','1'],['0','0','0'],['0','0','0']])
        self.assertEqual(response[6:],[zero]*3)
        # Nilpotent C**6=0: P=sum_{k=0}^5 (C**k)' C**k, independently of the solver.
        j=[[int(v) for v in row] for row in jordan]
        companion=[[0]*3+row for row in j]+[[int(i==col) for col in range(6)] for i in range(3)]
        power=[[int(i==col) for col in range(6)] for i in range(6)]; witness=[[0]*6 for _ in range(6)]
        for _ in range(6):
            witness=[[witness[i][col]+sum(power[k][i]*power[k][col] for k in range(6)) for col in range(6)] for i in range(6)]
            power=[[sum(power[i][k]*companion[k][col] for k in range(6)) for col in range(6)] for i in range(6)]
        self.assertEqual(r['stability']['lyapunov_solution_exact'],[[str(v) for v in row] for row in witness])

    def test_long_run_multiplier_mean_and_stationary_covariance(self):
        r=var.stationary_moments(case())
        self.assertEqual(r['long_run_multiplier_exact'],[['2','3/4'],['0','3/2']])
        self.assertEqual(r['conditional_steady_mean_exact'],['5/4','-3/2'])
        self.assertEqual(r['unconditional_covariance_exact'],[['0','0'],['0','0']])
        q=case(stochastic=True); q['observations'][-1]['values']={'x':'1','y':'1'}
        result=var.stationary_moments(q)
        extra=F(2363,85008)
        expected=[[F(178,161)+extra,F(132,161)+extra],[F(132,161)+extra,F(155,161)+extra]]
        self.assertEqual([[F(v) for v in row] for row in result['unconditional_covariance_exact']],expected)
        self.assertEqual(result['long_run_multiplier_exact'],[['14/11','-5/22'],['3/11','17/22']])
        self.assertEqual(result['maximum_covariance_identity_residual_exact'],'0')

    def test_stationary_scalar_variance_matches_geometric_innovation_sum(self):
        q=case(); q['variables']=['x']; q['units']={'x':'units'}; q['model']['intercept']=False
        q['observations']=[dict(period=i,values={'x':str(v)}) for i,v in enumerate([0,1,1,0,0,0])]
        q['estimation_end_period']=5
        self.assertEqual(var.stationary_moments(q)['unconditional_covariance_exact'],[['1/2']])
        self.assertEqual(var.stationary_moments(q)['long_run_multiplier_exact'],[['2']])


if __name__=='__main__': unittest.main()
