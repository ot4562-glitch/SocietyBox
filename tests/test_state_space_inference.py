import copy
from decimal import Context,Decimal,localcontext,getcontext,ROUND_DOWN
from fractions import Fraction as F
import json
import math
import unittest
from unittest.mock import patch

from econscope import state_space as ss
from econscope import state_space_inference as inf
from tests.test_state_space import case,iid_case,missing_case,fit_options


def parameter(name,field,row=0,column=0):return dict(name=name,field=field,row=row,column=column)


def iid_at_mle(values=(0,1,2,3,4)):
    q=iid_case(values)
    for step in q['steps']:step['observation_intercept']=[2]
    return q


def perturb(q,parameters,deltas):
    q=copy.deepcopy(q)
    for spec,delta in zip(parameters,deltas):
        for step in q['steps']:
            if step['period']>q['conditioning_end_period']:continue
            field,row,col=spec['field'],spec['row'],spec['column']
            if col is None:step[field][row]=str(F(step[field][row])+delta)
            else:
                value=str(F(step[field][row][col])+delta);step[field][row][col]=value
                if field in ('process_covariance','measurement_covariance'):step[field][col][row]=value
    return q


class StateSpaceInferenceTests(unittest.TestCase):
    def test_wald_iid_reference_and_equivalent_restrictions(self):
        q=iid_at_mle();params=[parameter('d','observation_intercept',column=None),parameter('R','measurement_covariance')]
        info=inf.likelihood_information(q,params)
        self.assertEqual(info['predictable_information_exact'],[['5/2','0'],['0','5/8']])
        self.assertIn('ACTUAL_PRIOR_OBSERVATIONS',info['predictable_information_scope'])
        restriction=dict(matrix=[[1,0]],target=[0])
        r=inf.wald_test(q,params,restriction,[0,1])
        self.assertEqual(r['statistic_exact'],'10');self.assertEqual(r['degrees_of_freedom'],1)
        self.assertAlmostEqual(r['p_value'],math.erfc(math.sqrt(5)),delta=1e-14)
        self.assertEqual(r['null_regularity_status'],'NULL_CHECK_PASSED_NOT_REGULARITY_PROOF')
        self.assertNotEqual(r['coverage']['status'],'PASS');json.dumps(r,allow_nan=False)
        equivalent=inf.wald_test(q,params,dict(matrix=[[2,0]],target=[0]),[0,1])
        self.assertEqual(equivalent['statistic_exact'],r['statistic_exact'])
        self.assertNotEqual(equivalent['input_sha256'],r['input_sha256'])
        two=inf.wald_test(q,params,dict(matrix=[[1,0],[0,1]],target=[0,1]),[0,1])
        self.assertEqual(two['statistic_exact'],'10');self.assertAlmostEqual(two['p_value'],math.exp(-5),delta=1e-14)
        swapped=inf.wald_test(q,params[::-1],dict(matrix=[[0,1]],target=[0]),[1,0])
        self.assertEqual(swapped['statistic_exact'],'10')

    def test_wald_blocks_boundary_and_singular_null_witness(self):
        q=iid_at_mle();restriction=dict(matrix=[[1]],target=[0])
        boundary=inf.wald_test(q,[parameter('R','measurement_covariance')],restriction,[0])
        self.assertEqual(boundary['statistic_exact'],'5/8');self.assertIsNone(boundary['p_value'])
        self.assertEqual(boundary['null_regularity_status'],'BLOCKED_NULL_WITNESS_BOUNDARY_OR_DEGENERATE')
        loading=inf.wald_test(q,[parameter('H','measurement')],restriction,[0])
        self.assertEqual(loading['statistic_exact'],'5/2');self.assertIsNone(loading['p_value'])
        self.assertEqual(loading['null_regularity_status'],'BLOCKED_NULL_PREDICTABLE_INFORMATION_SINGULAR')
        self.assertEqual(loading['null_predictable_information_rank'],0)

    def test_wald_rejects_invalid_restrictions_and_nonstationary_point(self):
        q=iid_at_mle();params=[parameter('d','observation_intercept',column=None),parameter('R','measurement_covariance')]
        with self.assertRaisesRegex(ValueError,'WALD_NULL_WITNESS_CONSTRAINT_MISMATCH'):
            inf.wald_test(q,params,dict(matrix=[[1,0]],target=[0]),[1,1])
        with self.assertRaisesRegex(ValueError,'WALD_RESTRICTION_RANK_DEFICIENT'):
            inf.wald_test(q,params,dict(matrix=[[1,0],[2,0]],target=[0,0]),[0,1])
        with self.assertRaisesRegex(ValueError,'WALD_REQUIRES_STATIONARY_POSITIVE_CURVATURE'):
            inf.wald_test(iid_case(),params,dict(matrix=[[1,0]],target=[0]),[0,1])

    def test_wald_probability_float_range_preserves_exact_statistic(self):
        underflow=inf.wald_test(iid_at_mle(),[parameter('d','observation_intercept',column=None)],
            dict(matrix=[[1]],target=[-1000000]),[-1000000])
        self.assertEqual(underflow['p_value'],0)
        self.assertEqual(underflow['p_value_status'],'BINARY64_UNDERFLOW_LOG_AVAILABLE')
        self.assertTrue(math.isfinite(underflow['log_p_value']))
        q=iid_case((-1e-150,1e-150))
        for step in q['steps']:
            step['transition']=[[0]];step['process_covariance']=[[0]];step['measurement']=[[0]]
            step['measurement_covariance']=[[1e-300]]
        r=inf.wald_test(q,[parameter('d','observation_intercept',column=None)],dict(matrix=[[1]],target=[1000000]),[1000000])
        self.assertEqual(F(r['statistic_exact']),F(2)*10**312)
        self.assertIsNone(r['p_value']);self.assertEqual(r['p_value_status'],'REFERENCE_PROBABILITY_BLOCKED_FLOAT64_RANGE')
        json.dumps(r,allow_nan=False)

    def test_wald_mixed_restriction_and_observation_unit_invariance(self):
        q=iid_at_mle();params=[parameter('d','observation_intercept',column=None),parameter('R','measurement_covariance')]
        mixed=inf.wald_test(q,params,dict(matrix=[[1,1]],target=[0]),[-1,1])
        self.assertEqual(mixed['statistic_exact'],'9/2')
        self.assertEqual(mixed['restriction_covariance_exact'],[['2']])
        baseline=inf.wald_test(q,params,dict(matrix=[[1,0],[0,1]],target=[0,1]),[0,1])
        q['units']['observables']['y']='hundredths_of_index'
        for step in q['steps']:
            step['measurement']=[[100]];step['observation_intercept']=[200]
            step['measurement_covariance']=[[10000]];step['values']['y']*=100
        rescaled=inf.wald_test(q,params,dict(matrix=[[1,0],[0,1]],target=[0,10000]),[0,10000])
        self.assertEqual(rescaled['statistic_exact'],baseline['statistic_exact'])
        self.assertEqual(rescaled['p_value'],baseline['p_value'])

    def test_predictable_information_conditions_on_actual_past(self):
        q=case();params=[parameter('Q','process_covariance')]
        r=inf.likelihood_information(q,params)
        # f1=3, df1=1; f2=8/3, df2=10/9, dmu2=y1/9.
        self.assertEqual(r['predictable_information_exact'],[['127/864']])
        self.assertEqual(r['period_contributions'][0]['predictable_information_exact'],[['1/18']])
        q['steps'][0]['values']['y']=0
        other=inf.likelihood_information(q,params)
        self.assertEqual(F(r['predictable_information_exact'][0][0])-F(other['predictable_information_exact'][0][0]),F(1,216))

    def test_wald_owned_snapshots_and_training_only_null_assignment(self):
        q=iid_at_mle();future=copy.deepcopy(q['steps'][-1]);future.update(period=6,observation_intercept=[123])
        q['steps'].append(future)
        params=[parameter('d','observation_intercept',column=None),parameter('R','measurement_covariance')]
        restriction=dict(matrix=[[1,0]],target=[0]);witness=[0,1]
        expected=inf.wald_test(q,params,restriction,witness);original=inf.likelihood_information;seen=[]
        def mutate(model,chart,tolerance):
            seen.append(copy.deepcopy(model));q['steps'][0]['values']['y']=999
            params[0]['name']='mutated';restriction['target'][0]=999;witness[0]=999
            return original(model,chart,tolerance)
        with patch.object(inf,'likelihood_information',side_effect=mutate):actual=inf.wald_test(q,params,restriction,witness)
        self.assertEqual(actual,expected)
        self.assertEqual(seen[1]['steps'][-1]['observation_intercept'],[123])
        self.assertEqual(seen[1]['steps'][0]['observation_intercept'],['0'])
        self.assertIn('null_regularity',[o['claim_id'] for o in actual['obligations']])

    def test_wald_invalid_shapes_values_and_code_change(self):
        q=iid_at_mle();params=[parameter('d','observation_intercept',column=None)]
        for restriction,witness in ((dict(matrix=[],target=[]),[0]),(dict(matrix=[[1,0]],target=[0]),[0]),
            (dict(matrix=[[1]],target=[]),[0]),(dict(matrix=[[True]],target=[0]),[0]),
            (dict(matrix=[[1]],target=[float('nan')]),[0]),(dict(matrix=[[1]],target=[0]),[])):
            with self.subTest(restriction=restriction,witness=witness),self.assertRaises(ValueError):
                inf.wald_test(q,params,restriction,witness)
        receipt=inf._code_receipt();changed=dict(receipt,**{'state_space_inference.py':'0'*64})
        # Entry, point entry, then point exit: detect source change before result.
        with patch.object(inf,'_code_receipt',side_effect=[receipt,receipt,changed]):
            with self.assertRaisesRegex(ValueError,'CODE_CHANGED_DURING_CALCULATION'):
                inf.wald_test(q,params,dict(matrix=[[1]],target=[0]),[0])


    def test_second_order_rational_arithmetic_retains_zero_value_derivatives(self):
        x=inf._Jet.seed(F(2),0,2);y=inf._Jet.seed(F(3),1,2);r=(x*x*y+1)/(y+2)
        self.assertEqual(r.v,F(13,5));self.assertEqual(r.g,(F(12,5),F(7,25)))
        self.assertEqual(r.h,(F(6,5),F(8,25),F(-14,125)))
        z=inf._Jet.seed(F(0),0,1);self.assertEqual((z*z).h,(F(2),))

    def test_iid_mean_variance_information_and_conditional_standard_errors(self):
        q=iid_at_mle();params=[parameter('d','observation_intercept',column=None),parameter('R','measurement_covariance')]
        r=inf.likelihood_information(q,params)
        self.assertEqual(r['score_exact'],['0','0'])
        self.assertEqual(r['observed_information_exact'],[['5/2','0'],['0','5/8']])
        self.assertEqual(r['inverse_information_exact'],[['2/5','0'],['0','8/5']])
        self.assertEqual(r['reference_status'],'CONDITIONAL_ASYMPTOTIC_REFERENCE_AVAILABLE')
        for actual,expected in zip(r['standard_error_reference_decimal'],[math.sqrt(2/5),math.sqrt(8/5)]):
            self.assertAlmostEqual(float(actual),expected,delta=1e-13)
        self.assertNotEqual(r['coverage']['status'],'PASS');self.assertEqual(r['native_execution'],'NOT_RUN')
        json.dumps(r,allow_nan=False)
        away=inf.likelihood_information(iid_case(),params)
        self.assertEqual(away['inverse_information_exact'],[['2','-8/5'],['-8/5','8/5']])
        self.assertEqual(away['newton_decrement_squared_exact'],'20')
        self.assertIsNone(away['standard_error_reference_decimal'])

    def test_exact_local_level_score_and_hessian_do_not_certify_nonstationary_point(self):
        r=inf.likelihood_information(case(),[parameter('Q','process_covariance')])
        self.assertEqual(r['score_exact'],['-1/8']);self.assertEqual(r['hessian_exact'],[['-3/32']])
        self.assertEqual(r['newton_decrement_squared_exact'],'1/6')
        self.assertEqual(r['reference_status'],'NOT_STATIONARY_AT_DECLARED_TOLERANCE')
        self.assertIsNone(r['standard_error_reference_decimal'])

    def test_fitted_local_level_score_curvature_match_rational_likelihood_derivatives(self):
        fitted=ss.fit_em(case(),['STATE_DYNAMICS'],fit_options(max_iterations=200,tolerance=1e-8),fixed_parameters=['transition','state_intercept'])
        q=fitted['model_request'];value=F(q['steps'][0]['process_covariance'][0][0])
        r=inf.likelihood_information(q,[parameter('Q','process_covariance')])
        score=-(1/(value+1)+1/(value+3)-2/(value+1)**2)/2
        information=(-1/(value+1)**2-1/(value+3)**2+4/(value+1)**3)/2
        self.assertEqual(F(r['score_exact'][0]),score);self.assertEqual(F(r['observed_information_exact'][0][0]),information)
        self.assertEqual(r['reference_status'],'CONDITIONAL_ASYMPTOTIC_REFERENCE_AVAILABLE')
        self.assertAlmostEqual(float(r['standard_error_reference_decimal'][0]),math.sqrt(float(1/information)),delta=1e-12)

    def test_capability_registration_discloses_conditional_curvature_scope(self):
        from econscope.study_capabilities import negotiate_capabilities
        cap=negotiate_capabilities(['state_space_likelihood_information'])['state_space_capabilities']
        self.assertIn('econscope.state_space_inference.likelihood_information',cap['python'])
        self.assertIn('CONDITIONAL',cap['parameter_information'])
        self.assertEqual(cap['native_upstream_parity'],'NOT_RUN')
        wald=negotiate_capabilities(['state_space_wald_test'])
        self.assertEqual(wald['status'],'SUPPORTED')
        self.assertIn('state_space_wald_test',wald['state_space_capabilities']['available'])
        self.assertIn('econscope.state_space_inference.wald_test',cap['python'])
        self.assertIn('NOT_REGULARITY_PROOF',cap['wald_null_check'])

    def test_redundant_loading_variance_chart_has_exact_null_direction(self):
        params=[parameter('d','observation_intercept',column=None),parameter('H','measurement'),parameter('R','measurement_covariance')]
        r=inf.likelihood_information(iid_at_mle(),params)
        self.assertEqual(r['information_rank'],2);self.assertEqual(r['information_nullspace_exact'],[['0','-1/2','1']])
        self.assertIsNone(r['inverse_information_exact']);self.assertIsNone(r['standard_error_reference_decimal'])
        self.assertEqual(r['reference_status'],'SINGULAR_OR_NONCONCAVE_CURVATURE')
        unseen=inf.likelihood_information(missing_case(),[parameter('Ruv','measurement_covariance',0,1)])
        self.assertEqual(unseen['score_exact'],['0']);self.assertEqual(unseen['observed_information_exact'],[['0']])
        self.assertEqual(unseen['information_rank'],0);self.assertIsNone(unseen['standard_error_reference_decimal'])
        minimum=iid_at_mle()
        for step in minimum['steps']:step['measurement']=[[0]]
        curvature=inf.likelihood_information(minimum,[parameter('H','measurement')])
        self.assertEqual(curvature['score_exact'],['0']);self.assertEqual(curvature['observed_information_exact'],[['-5']])
        self.assertEqual(curvature['information_rank'],1);self.assertIsNone(curvature['standard_error_reference_decimal'])

    def test_observation_unit_changes_transform_curvature_and_uncertainty(self):
        params=[parameter('d','observation_intercept',column=None),parameter('R','measurement_covariance')]
        q=iid_at_mle();a=inf.likelihood_information(q,params)
        for step in q['steps']:
            step['measurement']=[[100]];step['observation_intercept']=[200];step['measurement_covariance']=[[10000]]
            step['values']['y']*=100
        q['units']['observables']['y']='hundredths_of_index';b=inf.likelihood_information(q,params);scales=[100,10000]
        for i in range(2):
            for j in range(2):
                self.assertEqual(F(b['observed_information_exact'][i][j])*scales[i]*scales[j],F(a['observed_information_exact'][i][j]))
                self.assertEqual(F(b['inverse_information_exact'][i][j]),F(a['inverse_information_exact'][i][j])*scales[i]*scales[j])
            self.assertAlmostEqual(float(b['standard_error_reference_decimal'][i])/scales[i],float(a['standard_error_reference_decimal'][i]),delta=1e-12)

    def test_missing_iid_and_cutoff_preserve_score_information(self):
        params=[parameter('d','observation_intercept',column=None),parameter('R','measurement_covariance')]
        a=inf.likelihood_information(iid_at_mle(),params)
        b=inf.likelihood_information(iid_at_mle((0,None,1,None,2,None,3,None,4)),params)
        for key in ('score_exact','hessian_exact','observed_information_exact','inverse_information_exact'):self.assertEqual(a[key],b[key])
        q=case();r=inf.likelihood_information(q,[parameter('Q','process_covariance')]);q['steps'][2]['values']['y']=777
        changed=inf.likelihood_information(q,[parameter('Q','process_covariance')])
        self.assertEqual(r['score_exact'],changed['score_exact']);self.assertNotEqual(r['input_sha256'],changed['input_sha256'])

    def test_correlated_missing_derivatives_match_independent_richardson_filter_oracle(self):
        q=missing_case();q['steps'][1]['values']['u']=3
        params=[parameter('F','transition'),parameter('Ruv','measurement_covariance',0,1),parameter('du','observation_intercept',column=None)]
        r=inf.likelihood_information(q,params)
        def value(deltas):return Decimal(ss.filter(perturb(q,params,deltas))['log_likelihood'])
        with localcontext(Context(prec=70)):
            f0=value([F(0)]*3)
            for i in range(3):
                first=[]
                for h in (F(1,1000),F(1,2000)):
                    plus=[F(0)]*3;minus=list(plus);plus[i]=h;minus[i]=-h
                    dh=Decimal(h.numerator)/Decimal(h.denominator)
                    first.append((value(plus)-value(minus))/(2*dh))
                self.assertLess(abs(Decimal(F(r['score_exact'][i]).numerator)/Decimal(F(r['score_exact'][i]).denominator)-(4*first[1]-first[0])/3),Decimal('1e-8'))
                for j in range(i,3):
                    second=[]
                    for h in (F(1,1000),F(1,2000)):
                        dh=Decimal(h.numerator)/Decimal(h.denominator)
                        if i==j:
                            plus=[F(0)]*3;minus=list(plus);plus[i]=h;minus[i]=-h
                            second.append((value(plus)-2*f0+value(minus))/(dh*dh))
                        else:
                            total=Decimal(0)
                            for a,b in ((1,1),(1,-1),(-1,1),(-1,-1)):
                                d=[F(0)]*3;d[i]=a*h;d[j]=b*h;total+=a*b*value(d)
                            second.append(total/(4*dh*dh))
                    expected=F(r['hessian_exact'][i][j]);actual=Decimal(expected.numerator)/Decimal(expected.denominator)
                    self.assertLess(abs(actual-(4*second[1]-second[0])/3),Decimal('1e-8'))

    def test_parameter_chart_and_degenerate_likelihood_fail_explicitly(self):
        q=case()
        for params in ([],[parameter('bad','initial_covariance')],[parameter('Q','process_covariance')]*2,
                       [parameter('d','observation_intercept')],[parameter('Q','process_covariance',True,0)]):
            with self.assertRaisesRegex(ValueError,'PARAMETER_CHART'):inf.likelihood_information(q,params)
        q=case();q['steps'][1]['process_covariance']=[[2]]
        with self.assertRaisesRegex(ValueError,'SHARED_TRAINING_ENTRY'):inf.likelihood_information(q,[parameter('Q','process_covariance')])
        q=case()
        for step in q['steps']:step.update(process_covariance=[[0]],measurement_covariance=[[0]],values={'y':1})
        with self.assertRaisesRegex(ValueError,'COVARIANCE_PARAMETER_INTERIOR'):inf.likelihood_information(q,[parameter('R','measurement_covariance')])
        with self.assertRaisesRegex(ValueError,'NONDEGENERATE_INNOVATION'):inf.likelihood_information(q,[parameter('d','observation_intercept',column=None)])

    def test_observation_permutation_preserves_exact_derivative_chart(self):
        q=missing_case();q['steps'][1]['values']['u']=3
        params=[parameter('F','transition'),parameter('Ruv','measurement_covariance',0,1),parameter('du','observation_intercept',column=None)]
        a=inf.likelihood_information(q,params);q['observables'].reverse()
        for step in q['steps']:
            step['measurement'].reverse();step['observation_intercept'].reverse()
            step['measurement_covariance']=[list(reversed(row)) for row in reversed(step['measurement_covariance'])]
        params[2]['row']=1;b=inf.likelihood_information(q,params)
        for key in ('score_exact','hessian_exact','observed_information_exact','information_rank'):self.assertEqual(a[key],b[key])
        with self.assertRaisesRegex(ValueError,'DUPLICATE_COORDINATE'):
            inf.likelihood_information(q,[parameter('a','measurement_covariance',0,1),parameter('b','measurement_covariance',1,0)])

    def test_input_code_and_decimal_context_bindings_are_preserved(self):
        q=iid_at_mle();params=[parameter('d','observation_intercept',column=None),parameter('R','measurement_covariance')]
        before=inf.likelihood_information(q,params);differentiate=inf._differentiate
        def mutate(*args):
            q['steps'][0]['values']['y']=999;params[0]['name']='changed'
            return differentiate(*args)
        with patch.object(inf,'_differentiate',side_effect=mutate):after=inf.likelihood_information(q,params)
        self.assertEqual(before,after)
        params=[parameter('d','observation_intercept',column=None),parameter('R','measurement_covariance')];ctx=getcontext().copy()
        try:
            getcontext().prec=5;getcontext().rounding=ROUND_DOWN
            self.assertEqual(before,inf.likelihood_information(iid_at_mle(),params))
        finally:
            getcontext().prec=ctx.prec;getcontext().rounding=ctx.rounding
        with patch.object(inf,'_code_receipt',side_effect=[{'code':'before'},{'code':'after'}]):
            with self.assertRaisesRegex(ValueError,'CODE_CHANGED'):inf.likelihood_information(iid_at_mle(),params)

    def test_no_observations_have_zero_information_and_large_work_is_bounded(self):
        params=[parameter('d','observation_intercept',column=None),parameter('R','measurement_covariance')]
        r=inf.likelihood_information(iid_at_mle((None,None)),params)
        self.assertEqual(r['score_exact'],['0','0']);self.assertEqual(r['information_rank'],0);self.assertEqual(r['measurement_count'],0)
        self.assertIsNone(r['standard_error_reference_decimal'])
        q=case();q['states']=[f'x{i}' for i in range(6)];q['observables']=[f'y{i}' for i in range(4)]
        q['units']=dict(states={name:'index' for name in q['states']},observables={name:'index' for name in q['observables']})
        identity=lambda n:[[int(i==j) for j in range(n)] for i in range(n)]
        q['initial'].update(mean=[0]*6,covariance=identity(6));q['conditioning_end_period']=128
        q['steps']=[dict(period=t,transition=identity(6),state_intercept=[0]*6,process_covariance=identity(6),measurement=[[int(i==j) for j in range(6)] for i in range(4)],observation_intercept=[0]*4,measurement_covariance=identity(4),values={name:0 for name in q['observables']}) for t in range(1,129)]
        params=[parameter(str(i),'transition',i//6,i%6) for i in range(8)]
        with self.assertRaisesRegex(ValueError,'RESOURCE_WORK_LIMIT'):inf.likelihood_information(q,params)

    def test_constant_mean_intermediate_still_obeys_rational_resource_limit(self):
        q=case();q['conditioning_end_period']=1;q['initial']['mean']=[str(F(1,2**8190-1))]
        for step in q['steps']:step.update(state_intercept=[str(F(1,2**8190-3))],measurement=[[0]])
        with self.assertRaisesRegex(ValueError,'RESOURCE_LIMIT'):inf.likelihood_information(q,[parameter('R','measurement_covariance')])


if __name__=='__main__':unittest.main()
