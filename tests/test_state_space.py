import copy
from decimal import getcontext,ROUND_DOWN
from fractions import Fraction as F
import json
import math
import unittest
from unittest.mock import patch

from econscope import state_space as ss


def case():
    return dict(schema='societybox-linear-gaussian-state-space-v1',states=['level'],observables=['y'],period_unit='month',
        initial=dict(period=0,mean=[0],covariance=[[1]]),conditioning_end_period=2,
        distribution='GAUSSIAN',noise_assumptions='INDEPENDENT_ACROSS_TIME_INITIAL_STATE_AND_MEASUREMENT',
        steps=[dict(period=t,transition=[[1]],state_intercept=[0],process_covariance=[[1]],
                    measurement=[[1]],observation_intercept=[0],measurement_covariance=[[1]],values={'y':v}) for t,v in ((1,1),(2,2),(3,None))],
        units=dict(states={'level':'index'},observables={'y':'index'}),source_declared_sha256='a'*64,
        rights=dict(local_processing='PERMITTED',report_export='PERMITTED'))


def fractions(matrix): return [[F(v) for v in row] for row in matrix]


def vector_case():
    q=case();q['states']=['a','b'];q['observables']=['u','v']
    q['units']=dict(states={'a':'index','b':'index'},observables={'u':'index','v':'index'})
    q['initial'].update(mean=[1,-1],covariance=[[2,1],[1,3]]);q['conditioning_end_period']=3
    for i,s in enumerate(q['steps']):
        s.update(transition=([[1,1],[0,1]] if i%2==0 else [[1,0],['1/2',1]]),state_intercept=[i,'1/3'],
                 process_covariance=[[1,'1/2'],['1/2',1]],measurement=[[1,1],[1,-1]],
                 observation_intercept=['1/4',i],measurement_covariance=[[2,'1/3'],['1/3',1]],
                 values={'u':2 if i!=2 else None,'v':None if i==0 else i-1})
    return q


def missing_case():
    q=case();q['initial'].update(mean=['1/2'],covariance=[[2]]);q['observables']=['u','v'];q['units']['observables']={'u':'index','v':'index'}
    q['conditioning_end_period']=3
    for step,values in zip(q['steps'],({'u':1,'v':None},{'u':None,'v':2},{'u':None,'v':None})):
        step.update(transition=[['1/2']],state_intercept=['1/7'],process_covariance=[[2]],measurement=[[1],[2]],
            observation_intercept=['1/3','-1/4'],measurement_covariance=[[1,'1/2'],['1/2',1]],values=values)
    return q


def iid_case(values=(0,1,2,3,4)):
    q=case();q['conditioning_end_period']=len(values);q['steps']=[]
    for i,value in enumerate(values,1):
        q['steps'].append(dict(period=i,transition=[[0]],state_intercept=[0],process_covariance=[[1]],
            measurement=[[1]],observation_intercept=[0],measurement_covariance=[[1]],values={'y':value}))
    return q


def fit_options(**updates):
    options=dict(max_iterations=100,tolerance=1e-7,parameter_precision_digits=12);options.update(updates);return options


def inverse(matrix):
    # Batch-conditioning oracle uses pivoted Gauss-Jordan, not Kalman/RTS/LDL.
    n=len(matrix);a=[list(row)+[F(i==j) for j in range(n)] for i,row in enumerate(matrix)]
    for j in range(n):
        pivot=next(i for i in range(j,n) if a[i][j]);a[j],a[pivot]=a[pivot],a[j]
        value=a[j][j];a[j]=[v/value for v in a[j]]
        for i in range(n):
            if i!=j:
                value=a[i][j];a[i]=[v-value*w for v,w in zip(a[i],a[j])]
    return [row[n:] for row in a]


def joint_states(q,end):
    """Unconditional joint moments, without any Kalman conditioning."""
    d=len(q['states']);means=list(map(F,q['initial']['mean']));cov=fractions(q['initial']['covariance']);selected=[]
    for step in q['steps']:
        if step['period']>end:break
        n=len(means);f=fractions(step['transition']);noise=fractions(step['process_covariance'])
        mean=[F(c)+sum(f[i][j]*means[n-d+j] for j in range(d)) for i,c in enumerate(step['state_intercept'])]
        cross=[[sum(cov[a][n-d+j]*f[i][j] for j in range(d)) for i in range(d)] for a in range(n)]
        variance=[[noise[i][j]+sum(f[i][a]*cov[n-d+a][n-d+b]*f[j][b] for a in range(d) for b in range(d)) for j in range(d)] for i in range(d)]
        cov=[row+cross[i] for i,row in enumerate(cov)]+[[cross[a][i] for a in range(n)]+variance[i] for i in range(d)]
        means+=mean;selected.append(step)
    return means,cov,selected


def batch_condition(q,end):
    """Joint unconditional state distribution, then one batch Gaussian update."""
    d=len(q['states']);means,cov,selected=joint_states(q,end);observations=[]
    for t,step in enumerate(selected):
        for j,name in enumerate(q['observables']):
            if step['values'][name] is not None:
                loading=[F(0)]*len(means);loading[(t+1)*d:(t+2)*d]=map(F,step['measurement'][j])
                observations.append((t,j,loading,F(step['values'][name])-F(step['observation_intercept'][j])))
    m=len(observations)
    cross=[[sum(cov[i][k]*o[2][k] for k in range(len(means))) for o in observations] for i in range(len(means))]
    innovation=[[sum(a[2][k]*cross[k][j] for k in range(len(means)))+
                 (F(selected[a[0]]['measurement_covariance'][a[1]][b[1]]) if a[0]==b[0] else 0)
                 for j,b in enumerate(observations)] for a in observations]
    inv=inverse(innovation);error=[o[3]-sum(a*b for a,b in zip(o[2],means)) for o in observations]
    gain=[[sum(cross[i][k]*inv[k][j] for k in range(m)) for j in range(m)] for i in range(len(means))]
    mean=[v+sum(a*b for a,b in zip(g,error)) for v,g in zip(means,gain)]
    covariance=[[cov[i][j]-sum(gain[i][k]*cross[j][k] for k in range(m)) for j in range(len(means))] for i in range(len(means))]
    return mean,covariance


def batch_state_and_emissions(q,end):
    """Joint x and all y, conditioning only the observed y coordinates at once."""
    d=len(q['states']);means,cov,selected=joint_states(q,end);n=len(means);items=[]
    for t,step in enumerate(selected):
        for j,name in enumerate(q['observables']):
            loading=[F(0)]*n;loading[(t+1)*d:(t+2)*d]=map(F,step['measurement'][j])
            items.append((t,j,loading,F(step['observation_intercept'][j]),step['values'][name]))
    cross=[[sum(cov[i][k]*v[2][k] for k in range(n)) for v in items] for i in range(n)]
    yy=[[sum(a[2][k]*cross[k][j] for k in range(n))+
         (F(selected[a[0]]['measurement_covariance'][a[1]][b[1]]) if a[0]==b[0] else 0) for j,b in enumerate(items)] for a in items]
    allmean=means+[v[3]+sum(a*b for a,b in zip(v[2],means)) for v in items]
    allcov=[row+cross[i] for i,row in enumerate(cov)]+[[cross[i][j] for i in range(n)]+yy[j] for j in range(len(items))]
    observed=[n+j for j,v in enumerate(items) if v[4] is not None]
    if not observed:return allmean,allcov,n
    inv=inverse([[allcov[i][j] for j in observed] for i in observed])
    error=[F(items[i-n][4])-allmean[i] for i in observed]
    gain=[[sum(allcov[i][k]*inv[a][j] for a,k in enumerate(observed)) for j in range(len(observed))] for i in range(len(allmean))]
    mean=[v+sum(a*b for a,b in zip(row,error)) for v,row in zip(allmean,gain)]
    covariance=[[allcov[i][j]-sum(gain[i][a]*allcov[k][j] for a,k in enumerate(observed)) for j in range(len(allmean))] for i in range(len(allmean))]
    return mean,covariance,n


class StateSpaceTests(unittest.TestCase):
    def test_fixed_dynamics_noise_m_step_uses_full_residual_second_moment(self):
        q=case();fixed=['transition','state_intercept','measurement','observation_intercept']
        r=ss.em_step(q,['STATE_DYNAMICS','OBSERVATION_DYNAMICS'],fixed_parameters=fixed)
        model=r['candidate_request']['steps'][0]
        self.assertEqual(model['process_covariance'],[['7/8']]);self.assertEqual(model['measurement_covariance'],[['11/16']])
        for old,new in zip(q['steps'],r['candidate_request']['steps']):
            for key in fixed:self.assertEqual(old[key],new[key])
        self.assertEqual(r['sufficient_statistics']['STATE_DYNAMICS']['free_coefficient_columns'],[])
        q['initial']['covariance']=[[0]];q['conditioning_end_period']=1
        r=ss.em_step(q,['STATE_DYNAMICS','OBSERVATION_DYNAMICS'],fixed_parameters=fixed)
        self.assertEqual(r['candidate_request']['steps'][0]['process_covariance'],[['3/4']])
        self.assertEqual(r['candidate_request']['steps'][0]['measurement_covariance'],[['3/4']])

    def test_fixed_intercepts_or_matrices_have_correct_conditional_regressions(self):
        q=case();blocks=['STATE_DYNAMICS','OBSERVATION_DYNAMICS']
        a=ss.em_step(q,blocks,fixed_parameters=['state_intercept','observation_intercept'])['candidate_request']['steps'][0]
        for key,value in [('transition','20/19'),('process_covariance','265/304'),('measurement','32/35'),('measurement_covariance','47/70')]:
            self.assertEqual(a[key],[[value]])
        b=ss.em_step(q,blocks,fixed_parameters=['transition','measurement'])['candidate_request']['steps'][0]
        self.assertEqual(b['state_intercept'],['1/2']);self.assertEqual(b['observation_intercept'],['1/4'])
        self.assertEqual(b['process_covariance'],[['5/8']]);self.assertEqual(b['measurement_covariance'],[['5/8']])
        c=ss.em_step(q,blocks,fixed_parameters=['process_covariance','measurement_covariance'])['candidate_request']['steps'][0]
        self.assertEqual(c['transition'],[['1/2']]);self.assertEqual(c['state_intercept'],['7/8'])
        self.assertEqual(c['measurement'],[['1/5']]);self.assertEqual(c['observation_intercept'],['5/4'])
        self.assertEqual(c['process_covariance'],[[1]]);self.assertEqual(c['measurement_covariance'],[[1]])

    def test_fixed_rational_loading_is_not_quantized_during_iterated_fit(self):
        q=iid_case()
        for step in q['steps']:step.update(process_covariance=[[9]],measurement=[['1/3']])
        r=ss.fit_em(q,['OBSERVATION_DYNAMICS'],fit_options(),fixed_parameters=['measurement'])
        self.assertEqual(r['status'],'CONVERGED_BY_DECLARED_CHANGE_CRITERION')
        for step in r['model_request']['steps']:self.assertEqual(step['measurement'],[['1/3']])
        self.assertAlmostEqual(float(F(r['model_request']['steps'][0]['observation_intercept'][0])),2,delta=2e-6)
        self.assertAlmostEqual(float(F(r['model_request']['steps'][0]['measurement_covariance'][0][0])),1,delta=2e-6)

    def test_fixed_local_level_process_variance_matches_scalar_likelihood_oracle(self):
        q=case();r=ss.fit_em(q,['STATE_DYNAMICS'],fit_options(max_iterations=200,tolerance=1e-8),fixed_parameters=['transition','state_intercept'])
        self.assertEqual(r['status'],'CONVERGED_BY_DECLARED_CHANGE_CRITERION')
        estimate=float(F(r['model_request']['steps'][0]['process_covariance'][0][0]))
        self.assertAlmostEqual(estimate,math.sqrt(2)-1,delta=3e-7)
        expected=-.5*(2*math.log(2*math.pi)+math.log(estimate+1)+math.log(estimate+3)+2/(estimate+1))
        self.assertAlmostEqual(float(r['log_likelihood']),expected,delta=1e-12)

    def test_fixed_parameter_policy_validation_is_explicit(self):
        for fixed in (['unknown'],['measurement','measurement'],['transition'],'measurement'):
            with self.assertRaisesRegex(ValueError,'EM_FIXED_PARAMETERS'):ss.em_step(case(),['OBSERVATION_DYNAMICS'],fixed_parameters=fixed)
        with self.assertRaisesRegex(ValueError,'EM_NO_FREE_PARAMETERS'):
            ss.em_step(case(),['OBSERVATION_DYNAMICS'],fixed_parameters=['measurement','observation_intercept','measurement_covariance'])

    def test_missing_correlated_fixed_intercepts_match_batch_conditional_regression(self):
        q=missing_case();mean,covariance,start=batch_state_and_emissions(q,3)
        count=3;zz=[[F(0)]*2 for _ in range(2)];yz=[[F(0)]*2 for _ in range(2)];yy=[[F(0)]*2 for _ in range(2)]
        for t in range(1,4):
            y=start+(t-1)*2;zm=[mean[t],F(1)];ym=mean[y:y+2]
            for i in range(2):
                for j in range(2):zz[i][j]+=zm[i]*zm[j]+(covariance[t][t] if i==j==0 else 0)
            for i in range(2):
                for j in range(2):yz[i][j]+=ym[i]*zm[j]+(covariance[y+i][t] if j==0 else 0)
                for j in range(2):yy[i][j]+=ym[i]*ym[j]+covariance[y+i][y+j]
        d=list(map(F,q['steps'][0]['observation_intercept']))
        b=[[(row[0]-intercept*zz[1][0])/zz[0][0],intercept] for row,intercept in zip(yz,d)]
        noise=[[(yy[i][j]-sum(b[i][k]*yz[j][k]+yz[i][k]*b[j][k] for k in range(2))+
                 sum(b[i][a]*zz[a][c]*b[j][c] for a in range(2) for c in range(2)))/count for j in range(2)] for i in range(2)]
        r=ss.em_step(q,['OBSERVATION_DYNAMICS'],fixed_parameters=['observation_intercept']);model=r['candidate_request']['steps'][0]
        self.assertEqual(fractions(model['measurement']),[row[:1] for row in b]);self.assertEqual(fractions(model['measurement_covariance']),noise)
        self.assertEqual(model['observation_intercept'],q['steps'][0]['observation_intercept'])
        self.assertTrue(any(row[1]-sum(a*c for a,c in zip(coeff,[zz[0][1],zz[1][1]]))!=0 for row,coeff in zip(yz,b)))
        self.assertEqual(r['sufficient_statistics']['OBSERVATION_DYNAMICS']['free_coefficient_columns'],[0])

    def test_fixed_policy_snapshot_digest_and_backtracking_preserve_declarations(self):
        q=iid_case();fixed=['measurement'];before=ss.em_step(q,['OBSERVATION_DYNAMICS'],fixed_parameters=fixed);smoother=ss._run_smoother
        def mutate(*args,**kwargs):
            fixed.clear()
            return smoother(*args,**kwargs)
        with patch.object(ss,'_run_smoother',side_effect=mutate):after=ss.em_step(q,['OBSERVATION_DYNAMICS'],fixed_parameters=fixed)
        self.assertEqual(before,after)
        self.assertNotEqual(before['input_sha256'],ss.em_step(q,['OBSERVATION_DYNAMICS'])['input_sha256'])
        for step in q['steps']:step.update(process_covariance=[[9]],measurement=[['1/3']])
        quantize=ss._quantized_request
        def overshoot(*args):
            target=quantize(*args)
            for step in target['steps']:step['observation_intercept']=['4']
            return target
        with patch.object(ss,'_quantized_request',side_effect=overshoot):r=ss.fit_em(q,['OBSERVATION_DYNAMICS'],fit_options(max_iterations=1),fixed_parameters=['measurement'])
        self.assertEqual(r['iteration_trace'][0]['step_fraction'],'1/2')
        for step in r['model_request']['steps']:self.assertEqual(step['measurement'],[['1/3']])

    def test_iterated_em_matches_iid_normal_mle_and_exact_em_recurrence(self):
        from decimal import Decimal
        q=iid_case();r=ss.fit_em(q,['OBSERVATION_DYNAMICS'],fit_options())
        self.assertEqual(r['status'],'CONVERGED_BY_DECLARED_CHANGE_CRITERION')
        model=r['model_request']['steps'][0];h=F(model['measurement'][0][0]);noise=F(model['measurement_covariance'][0][0]);offset=F(model['observation_intercept'][0])
        self.assertAlmostEqual(float(offset),2,delta=2e-6)
        self.assertAlmostEqual(float(h*h+noise),2,delta=2e-6)
        expected=-2.5*(math.log(2*math.pi)+math.log(2)+1)
        self.assertAlmostEqual(float(r['log_likelihood']),expected,delta=1e-10)
        self.assertEqual(r['iteration_trace'][0]['parameter_distance_scaled'],'1')
        self.assertEqual(r['iteration_trace'][0]['step_fraction'],'1')
        for row in r['iteration_trace']:
            self.assertGreaterEqual(Decimal(row['log_likelihood_after']),Decimal(row['log_likelihood_before']))
            self.assertGreaterEqual(Decimal(row['expected_complete_loglik_after']),Decimal(row['expected_complete_loglik_before']))
        self.assertEqual(r['parameter_identification'],'NOT_ESTABLISHED_LATENT_COORDINATE_EQUIVALENCE')
        ss.validate(r['model_request']);json.dumps(r,allow_nan=False)

    def test_iterated_em_limit_boundary_and_bad_options_have_explicit_status(self):
        options=fit_options();options['max_iterations']=1
        r=ss.fit_em(iid_case(),['OBSERVATION_DYNAMICS'],options)
        self.assertEqual(r['status'],'ITERATION_LIMIT');self.assertEqual(r['iterations'],1)
        self.assertEqual(F(r['model_request']['steps'][0]['observation_intercept'][0]),1)
        r=ss.fit_em(iid_case((1,)),['OBSERVATION_DYNAMICS'],fit_options())
        self.assertEqual(r['status'],'M_STEP_COVARIANCE_BOUNDARY')
        self.assertEqual(r['accepted_updates'],0)
        for field,value in [('max_iterations',0),('parameter_precision_digits',7),('tolerance',0)]:
            options=fit_options();options[field]=value
            with self.assertRaisesRegex(ValueError,'FIT_OPTIMIZER'):ss.fit_em(iid_case(),['OBSERVATION_DYNAMICS'],options)

    def test_iterated_em_missing_data_reaches_observed_iid_mle(self):
        q=iid_case((0,None,1,None,2,None,3,None,4));r=ss.fit_em(q,['OBSERVATION_DYNAMICS'],fit_options())
        self.assertEqual(r['status'],'CONVERGED_BY_DECLARED_CHANGE_CRITERION')
        model=r['model_request']['steps'][0];h=F(model['measurement'][0][0]);noise=F(model['measurement_covariance'][0][0])
        self.assertAlmostEqual(float(F(model['observation_intercept'][0])),2,delta=5e-6)
        self.assertAlmostEqual(float(h*h+noise),2,delta=5e-6)
        self.assertEqual(r['likelihood_scope'],'OBSERVED_TRAINING_COMPONENTS_MISSING_INTEGRATED_OUT')

    def test_iterated_em_quantization_does_not_claim_unresolved_convergence(self):
        q=iid_case((0,1,3))
        for step in q['steps']:step['measurement_covariance']=[['5/9']]
        options=fit_options();options.update(tolerance=1e-12,parameter_precision_digits=8)
        r=ss.fit_em(q,['OBSERVATION_DYNAMICS'],options)
        self.assertNotEqual(r['status'],'CONVERGED_BY_DECLARED_CHANGE_CRITERION')
        self.assertIn(r['status'],('PARAMETER_QUANTIZATION_LIMIT','LIKELIHOOD_LINE_SEARCH_FAILED'))
        self.assertGreater(float(r['last_unquantized_m_step_distance_scaled']),1e-12)

    def test_covariance_factor_quantization_keeps_positive_definiteness(self):
        matrix=fractions([['1','999999999999/1000000000000'],['999999999999/1000000000000','1']])
        rounded=ss._quantized_covariance(matrix,8)
        self.assertTrue(all(v>0 for v in ss._factor(rounded)[1]))
        self.assertNotEqual(rounded,[[F(1),F(1)],[F(1),F(1)]])

    def test_iterated_em_no_data_preserves_input_and_reports_fixed_point(self):
        q=iid_case((None,None));original=copy.deepcopy(q);r=ss.fit_em(q,['OBSERVATION_DYNAMICS'],fit_options())
        self.assertEqual(r['status'],'CONVERGED_BY_DECLARED_CHANGE_CRITERION')
        self.assertEqual(r['accepted_updates'],0);self.assertEqual(r['log_likelihood'],'0')
        self.assertEqual(q,original);self.assertEqual(r['model_request'],q)

    def test_fit_resource_stop_keeps_last_accepted_model(self):
        proposal=ss._em_proposal;calls=0
        def stop_after_one(*args,**kwargs):
            nonlocal calls
            calls+=1
            if calls==2:raise ValueError('STATE_SPACE_EXACT_ARITHMETIC_RESOURCE_LIMIT')
            return proposal(*args,**kwargs)
        with patch.object(ss,'_em_proposal',side_effect=stop_after_one):r=ss.fit_em(iid_case(),['OBSERVATION_DYNAMICS'],fit_options())
        self.assertEqual(r['status'],'RESOURCE_LIMIT');self.assertEqual(r['accepted_updates'],1)
        self.assertEqual(r['model_request']['steps'][0]['observation_intercept'],['1'])

    def test_fit_backtracking_checks_expected_complete_and_observed_objectives(self):
        quantize=ss._quantized_request
        def overshoot(*args):
            result=quantize(*args)
            for step in result['steps']:step['observation_intercept']=['4']
            return result
        with patch.object(ss,'_quantized_request',side_effect=overshoot):r=ss.fit_em(iid_case(),['OBSERVATION_DYNAMICS'],fit_options(max_iterations=1))
        self.assertEqual(r['iteration_trace'][0]['step_fraction'],'1/2')
        self.assertEqual(r['model_request']['steps'][0]['observation_intercept'],['2'])

    def test_fit_snapshots_request_blocks_optimizer_and_rejects_code_change(self):
        q=iid_case();blocks=['OBSERVATION_DYNAMICS'];options=fit_options(max_iterations=1);before=ss.fit_em(q,blocks,options);proposal=ss._em_proposal
        def mutate(*args,**kwargs):
            q['steps'][0]['values']['y']=999;blocks.clear();options.clear()
            return proposal(*args,**kwargs)
        with patch.object(ss,'_em_proposal',side_effect=mutate):after=ss.fit_em(q,blocks,options)
        self.assertEqual(after,before)
        with patch.object(ss,'_code_receipt',side_effect=[{'code':'before'},{'code':'after'}]):
            with self.assertRaisesRegex(ValueError,'CODE_CHANGED'):ss.fit_em(iid_case(),['OBSERVATION_DYNAMICS'],fit_options(max_iterations=1))

    def test_fit_forward_reuse_is_call_local_and_input_digest_checked(self):
        from econscope.workspace import canonical,digest
        run_filter=ss._run_filter
        with patch.object(ss,'_run_filter',wraps=run_filter) as calls:
            r=ss.fit_em(iid_case(),['OBSERVATION_DYNAMICS'],fit_options(max_iterations=4))
        self.assertEqual(r['accepted_updates'],4)
        self.assertEqual(calls.call_count,5)
        q=iid_case();cached=(digest(canonical(q)),run_filter(q));q['units']['states']['level']='changed'
        with self.assertRaisesRegex(ValueError,'FORWARD_CACHE_INPUT_MISMATCH'):ss._run_smoother(q,cached)

    def test_scalar_filter_smoother_forecast_and_log_density_match_closed_form(self):
        q=case();r=ss.smooth(q)
        self.assertEqual([F(v['mean'][0]) for v in r['filtered']],[F(2,3),F(3,2)])
        self.assertEqual([F(v['covariance'][0][0]) for v in r['filtered']],[F(2,3),F(5,8)])
        self.assertEqual([F(v['mean'][0]) for v in r['smoothed']],[F(1),F(3,2)])
        self.assertEqual([F(v['covariance'][0][0]) for v in r['smoothed']],[F(1,2),F(5,8)])
        self.assertEqual(F(r['smoothed_initial']['mean'][0]),F(1,2))
        self.assertEqual(F(r['smoothed_initial']['covariance'][0][0]),F(5,8))
        self.assertEqual(F(r['lag_one_covariances'][0]['covariance'][0][0]),F(1,4))
        self.assertAlmostEqual(float(r['log_likelihood']),-.5*(2*math.log(2*math.pi)+math.log(8)+1),delta=1e-13)
        forecast=ss.forecast(q,1)['forecasts'][0]
        self.assertEqual(F(forecast['state']['mean'][0]),F(3,2))
        self.assertEqual(F(forecast['state']['covariance'][0][0]),F(13,8))
        self.assertEqual(F(forecast['observation_covariance'][0][0]),F(21,8))
        self.assertEqual(r['native_execution'],'NOT_RUN');self.assertNotEqual(r['coverage']['status'],'PASS')
        json.dumps(r,allow_nan=False)

    def test_multivariate_time_varying_smoothing_and_lag_covariance_match_batch_conditioning(self):
        q=vector_case()
        expected_mean,expected_cov=batch_condition(q,3);r=ss.smooth(q)
        cells=[r['smoothed_initial']]+r['smoothed']
        for t,cell in enumerate(cells):
            self.assertEqual(list(map(F,cell['mean'])),expected_mean[t*2:t*2+2])
            self.assertEqual(fractions(cell['covariance']),[row[t*2:t*2+2] for row in expected_cov[t*2:t*2+2]])
        for t,cell in enumerate(r['lag_one_covariances']):
            self.assertEqual(fractions(cell['covariance']),[row[(t+1)*2:(t+2)*2] for row in expected_cov[t*2:t*2+2]])
        for i,cell in enumerate(r['filtered'],1):
            m,p=batch_condition(q,i)
            self.assertEqual(list(map(F,cell['mean'])),m[-2:])
            self.assertEqual(fractions(cell['covariance']),[row[-2:] for row in p[-2:]])

    def test_partial_and_all_missing_measurements_use_only_observed_submatrix(self):
        q=case();q['conditioning_end_period']=1;q['observables']=['u','v'];q['units']['observables']={'u':'index','v':'index'}
        for s in q['steps']:
            s.update(process_covariance=[[0]],measurement=[[1],[1]],observation_intercept=[0,0],
                     measurement_covariance=[[1,'1/2'],['1/2',1]],values={'u':2,'v':None})
        r=ss.filter(q);self.assertEqual(F(r['filtered'][0]['mean'][0]),1)
        self.assertEqual(F(r['filtered'][0]['covariance'][0][0]),F(1,2))
        q['steps'][0]['values']['v']=2;r=ss.filter(q)
        self.assertEqual(F(r['filtered'][0]['mean'][0]),F(8,7))
        self.assertEqual(F(r['filtered'][0]['covariance'][0][0]),F(3,7))
        q['steps'][0]['values']={'u':None,'v':None};r=ss.filter(q)
        self.assertEqual(r['priors'][0]['mean'],r['filtered'][0]['mean'])
        self.assertEqual(r['log_likelihood'],'0');self.assertEqual(r['measurement_count'],0)

    def test_singular_measurements_and_deterministic_rts_are_supported_without_fake_density(self):
        q=case();q['observables']=['u','v'];q['units']['observables']={'u':'index','v':'index'}
        for s in q['steps']:
            s.update(process_covariance=[[0]],measurement=[[1],[2]],observation_intercept=[0,0],
                     measurement_covariance=[[0,0],[0,0]],values={'u':1,'v':2})
        r=ss.smooth(q);self.assertEqual(r['innovations'][0]['rank'],1);self.assertEqual(r['innovations'][1]['rank'],0)
        self.assertIsNone(r['log_likelihood']);self.assertIn('SINGULAR',r['likelihood_status'])
        self.assertEqual(F(r['smoothed_initial']['mean'][0]),1)
        self.assertEqual(F(r['smoothed_initial']['covariance'][0][0]),0)
        q['steps'][1]['values']['v']=3
        with self.assertRaisesRegex(ValueError,'OUTSIDE_DEGENERATE_GAUSSIAN_SUPPORT'):ss.filter(q)

    def test_cutoff_blocks_future_measurements_and_smoothing_is_retrospective(self):
        q=case();before=ss.smooth(q);forecast=ss.forecast(q,1)
        q['steps'][2]['values']['y']=9999
        after=ss.smooth(q);changed=ss.forecast(q,1)
        for key in ('filtered','smoothed','innovations'):self.assertEqual(before[key],after[key])
        self.assertEqual(forecast['forecasts'],changed['forecasts'])
        self.assertNotEqual(before['input_sha256'],after['input_sha256'])
        self.assertFalse(before['filtered'][0]['uses_future_measurements'])
        self.assertTrue(before['smoothed'][0]['uses_future_measurements'])
        q['conditioning_end_period']=1;prefix=ss.smooth(q)
        self.assertEqual(prefix['filtered'],before['filtered'][:1])
        self.assertNotEqual(prefix['smoothed'],before['smoothed'][:1])

    def test_invalid_covariance_noise_timing_rights_and_future_dynamics_fail(self):
        q=case();q['initial']['covariance']=[[-1]]
        with self.assertRaisesRegex(ValueError,'POSITIVE_SEMIDEFINITE'):ss.validate(q)
        q=case();q['steps'][1]['period']=3
        with self.assertRaisesRegex(ValueError,'CONTIGUOUS'):ss.validate(q)
        q=case();q['noise_assumptions']='CORRELATED'
        with self.assertRaisesRegex(ValueError,'NOISE_ASSUMPTIONS'):ss.validate(q)
        q=case();q['rights']['local_processing']='UNKNOWN'
        with self.assertRaisesRegex(ValueError,'RIGHTS'):ss.filter(q)
        with self.assertRaisesRegex(ValueError,'FUTURE_DYNAMICS'):ss.forecast(case(),2)

    def test_fixed_decimal_context_no_input_alias_and_capability_status(self):
        q=case();original=copy.deepcopy(q);before=ss.smooth(q);em_before=ss.em_step(q,['STATE_DYNAMICS']);ctx=getcontext().copy()
        try:
            getcontext().prec=5;getcontext().rounding=ROUND_DOWN
            self.assertEqual(ss.smooth(q),before)
            self.assertEqual(ss.em_step(q,['STATE_DYNAMICS']),em_before)
        finally:
            getcontext().prec=ctx.prec;getcontext().rounding=ctx.rounding
        before['filtered'][0]['mean'][0]='changed';self.assertEqual(q,original)
        from econscope.study_capabilities import negotiate_capabilities
        cap=negotiate_capabilities(['linear_gaussian_state_space'])['state_space_capabilities']
        self.assertEqual(cap['implementation_status'],'INDEPENDENT_IMPLEMENTED')
        self.assertEqual(cap['native_upstream_parity'],'NOT_RUN')
        self.assertIn('econscope.state_space.em_step',cap['python'])
        self.assertEqual(cap['em_convergence'],'DECLARED_CHANGE_CRITERIA_NOT_GLOBAL_OR_EXACT_MLE_CERTIFICATION')
        self.assertIn('econscope.state_space.fit_em',cap['python'])
        self.assertIn('PARTIAL_MISSING',cap['parameter_estimation'])
        self.assertIn('CORRELATED_R',cap['em_missing_data'])
        self.assertEqual(set(cap['em_fixed_parameter_groups']),{'transition','state_intercept','process_covariance','measurement','observation_intercept','measurement_covariance'})
        self.assertIn('EM_ELEMENTWISE_COVARIANCE_MASKS',cap['unsupported'])

    def test_receipt_binds_entry_snapshot_if_caller_mutates_during_calculation(self):
        q=case();before=ss.smooth(q);predict=ss._predict
        def mutate_then_predict(*args):
            q['steps'][0]['values']['y']=777
            q['units']['states']['level']='changed'
            return predict(*args)
        with patch.object(ss,'_predict',side_effect=mutate_then_predict):after=ss.smooth(q)
        self.assertEqual(after,before)

    def test_invertible_state_basis_preserves_observation_law_and_transforms_moments(self):
        q=vector_case();q['conditioning_end_period']=2;original=ss.smooth(q);forecast=ss.forecast(q,1)
        def left(a):return [list(map(F,[a[0][j]+a[1][j] for j in range(len(a[0]))])),list(map(F,a[1]))]
        def right_inverse(a):return [[F(row[0]),F(row[1])-F(row[0])] for row in a]
        def covariance(a):
            b=left(fractions(a));return [[b[i][0]+b[i][1],b[i][1]] for i in range(2)]
        def vector(a):return [F(a[0])+F(a[1]),F(a[1])]
        def strings(a):return [[str(v) for v in row] for row in a]
        q['initial']['mean']=list(map(str,vector(q['initial']['mean'])))
        q['initial']['covariance']=strings(covariance(q['initial']['covariance']))
        for s in q['steps']:
            s['transition']=strings(right_inverse(left(fractions(s['transition']))))
            s['state_intercept']=list(map(str,vector(s['state_intercept'])))
            s['process_covariance']=strings(covariance(s['process_covariance']))
            s['measurement']=strings(right_inverse(fractions(s['measurement'])))
        changed=ss.smooth(q);new_forecast=ss.forecast(q,1)
        for old,new in zip(original['smoothed'],changed['smoothed']):
            self.assertEqual(list(map(F,new['mean'])),vector(old['mean']))
            self.assertEqual(fractions(new['covariance']),covariance(old['covariance']))
        for old,new in zip(forecast['forecasts'],new_forecast['forecasts']):
            self.assertEqual(old['observation_mean'],new['observation_mean'])
            self.assertEqual(old['observation_covariance'],new['observation_covariance'])

    def test_observation_permutation_preserves_posterior_and_likelihood(self):
        from decimal import Decimal
        q=vector_case();original=ss.smooth(q);q['observables'].reverse()
        for s in q['steps']:
            s['measurement'].reverse();s['observation_intercept'].reverse()
            s['measurement_covariance']=[list(reversed(row)) for row in reversed(s['measurement_covariance'])]
        changed=ss.smooth(q)
        self.assertEqual(original['filtered'],changed['filtered'])
        self.assertEqual(original['smoothed'],changed['smoothed'])
        self.assertLess(abs(Decimal(original['log_likelihood'])-Decimal(changed['log_likelihood'])),Decimal('1e-70'))

    def test_zero_transition_leaves_initial_prior_uninformed_by_deterministic_future(self):
        q=case();q['conditioning_end_period']=1
        q['steps'][0].update(transition=[[0]],process_covariance=[[0]],measurement_covariance=[[0]],values={'y':0})
        r=ss.smooth(q)
        self.assertEqual(r['smoothed_initial']['mean'],['0'])
        self.assertEqual(r['smoothed_initial']['covariance'],[['1']])
        self.assertEqual(r['lag_one_covariances'][0]['covariance'],[['0']])

    def test_psd_zero_pivot_nonzero_column_and_asymmetry_are_rejected(self):
        q=vector_case();q['initial']['covariance']=[[0,1],[1,0]]
        with self.assertRaisesRegex(ValueError,'POSITIVE_SEMIDEFINITE'):ss.validate(q)
        q=vector_case();q['steps'][0]['process_covariance']=[[1,0],[1,1]]
        with self.assertRaisesRegex(ValueError,'SYMMETRY'):ss.validate(q)

    def test_exact_fraction_growth_is_bounded_without_silent_rounding(self):
        q=case();template=q['steps'][0];q['steps']=[]
        for i in range(128):
            s=copy.deepcopy(template);s.update(period=i+1,process_covariance=[['1/9223372036854775783']],
                measurement_covariance=[['1/9223372036854775643']]);q['steps'].append(s)
        q['conditioning_end_period']=128
        with self.assertRaisesRegex(ValueError,'EXACT_ARITHMETIC_RESOURCE_LIMIT'):ss.filter(q)

    def test_six_state_four_observable_128_step_limit_preserves_unobserved_modes(self):
        q=case();q['states']=[f'x{i}' for i in range(6)];q['observables']=[f'y{i}' for i in range(4)]
        q['units']=dict(states={name:'index' for name in q['states']},observables={name:'index' for name in q['observables']})
        identity=lambda n:[[int(i==j) for j in range(n)] for i in range(n)]
        q['initial'].update(mean=[0]*6,covariance=identity(6));q['conditioning_end_period']=128
        q['steps']=[dict(period=t,transition=identity(6),state_intercept=[0]*6,process_covariance=[[0]*6 for _ in range(6)],
            measurement=[[int(i==j) for j in range(6)] for i in range(4)],observation_intercept=[0]*4,
            measurement_covariance=identity(4),values={name:0 for name in q['observables']}) for t in range(1,129)]
        r=ss.smooth(q)
        expected=[[str(F(1,129) if i==j and i<4 else F(i==j)) for j in range(6)] for i in range(6)]
        self.assertEqual(r['smoothed_initial']['covariance'],expected)
        self.assertEqual(r['smoothed'][-1]['covariance'],expected)
        self.assertEqual(r['measurement_count'],512)

    def test_one_em_step_matches_exact_augmented_regression_oracle(self):
        from decimal import Decimal
        q=case();r=ss.em_step(q,['STATE_DYNAMICS','OBSERVATION_DYNAMICS']);model=r['candidate_request']['steps'][0]
        self.assertEqual(fractions(model['transition']),[[F(1,2)]])
        self.assertEqual(list(map(F,model['state_intercept'])),[F(7,8)])
        self.assertEqual(fractions(model['process_covariance']),[[F(15,32)]])
        self.assertEqual(fractions(model['measurement']),[[F(1,5)]])
        self.assertEqual(list(map(F,model['observation_intercept'])),[F(5,4)])
        self.assertEqual(fractions(model['measurement_covariance']),[[F(9,40)]])
        self.assertEqual(r['candidate_request']['initial'],q['initial'])
        self.assertEqual(r['parameter_estimation'],'ONE_EM_STEP_NOT_CONVERGENCE_OR_GLOBAL_MAXIMUM')
        self.assertGreater(Decimal(r['candidate_log_likelihood']),Decimal(r['original_log_likelihood']))
        self.assertTrue(all(v['normal_equations_exact'] for v in r['sufficient_statistics'].values()))
        ss.validate(r['candidate_request']);json.dumps(r,allow_nan=False)

    def test_em_selected_blocks_and_future_values_do_not_change_other_parameters(self):
        q=case();original=copy.deepcopy(q);a=ss.em_step(q,['STATE_DYNAMICS'])
        for old,new in zip(q['steps'],a['candidate_request']['steps']):
            for key in ('measurement','observation_intercept','measurement_covariance','values'):self.assertEqual(old[key],new[key])
        q['steps'][2]['values']['y']=999;b=ss.em_step(q,['STATE_DYNAMICS'])
        self.assertEqual(a['sufficient_statistics'],b['sufficient_statistics'])
        self.assertEqual(a['candidate_log_likelihood'],b['candidate_log_likelihood'])
        self.assertNotEqual(a['input_sha256'],b['input_sha256'])
        self.assertEqual(original['initial'],a['candidate_request']['initial'])

    def test_em_rejects_time_varying_singular_noise_and_unidentified_moments(self):
        q=case();q['steps'][1]['transition']=[[2]]
        with self.assertRaisesRegex(ValueError,'EM_TIME_INVARIANT'):ss.em_step(q,['STATE_DYNAMICS'])
        q=case();q['steps'][0]['process_covariance']=[[0]]
        for step in q['steps']:step['process_covariance']=[[0]]
        with self.assertRaisesRegex(ValueError,'EM_POSITIVE_DEFINITE_NOISE'):ss.em_step(q,['STATE_DYNAMICS'])
        q=case();q['conditioning_end_period']=1;q['initial']['covariance']=[[0]]
        with self.assertRaisesRegex(ValueError,'EM_UNIDENTIFIED_SECOND_MOMENTS'):ss.em_step(q,['STATE_DYNAMICS'])
        for blocks in ([],['UNKNOWN'],['STATE_DYNAMICS']*2,'STATE_DYNAMICS'):
            with self.assertRaisesRegex(ValueError,'EM_BLOCKS'):ss.em_step(case(),blocks)

    def test_missing_em_keeps_conditional_second_moments_instead_of_mean_imputation(self):
        q=case();q['steps'][1]['values']['y']=None
        r=ss.em_step(q,['STATE_DYNAMICS','OBSERVATION_DYNAMICS']);model=r['candidate_request']['steps'][0]
        expected=dict(transition='18/25',state_intercept='23/75',process_covariance='121/150',
                      measurement='5/7',observation_intercept='5/14',measurement_covariance='193/252')
        for key,value in expected.items():
            actual=model[key][0] if key.endswith('intercept') else model[key][0][0]
            self.assertEqual(F(actual),F(value),key)
        self.assertEqual(r['observed_measurement_count'],1);self.assertEqual(r['missing_measurement_count'],1)
        self.assertEqual(r['sufficient_statistics']['OBSERVATION_DYNAMICS']['second_target_sum'],[['37/9']])
        self.assertEqual(r['likelihood_scope'],'OBSERVED_TRAINING_COMPONENTS_MISSING_INTEGRATED_OUT')

    def test_no_observations_em_preserves_parameters_and_does_not_invent_information(self):
        q=case()
        for step in q['steps']:step['values']['y']=None
        r=ss.em_step(q,['STATE_DYNAMICS','OBSERVATION_DYNAMICS']);model=r['candidate_request']['steps'][0]
        for key in ('transition','process_covariance','measurement','measurement_covariance'):self.assertEqual(model[key],[['1']])
        self.assertEqual(model['state_intercept'],['0']);self.assertEqual(model['observation_intercept'],['0'])
        self.assertEqual(r['original_log_likelihood'],'0');self.assertEqual(r['candidate_log_likelihood'],'0')
        self.assertEqual(r['observed_measurement_count'],0);self.assertEqual(r['missing_measurement_count'],2)
        q=missing_case()
        for step in q['steps']:step['values']={'u':None,'v':None}
        r=ss.em_step(q,['STATE_DYNAMICS','OBSERVATION_DYNAMICS']);old=q['steps'][0];new=r['candidate_request']['steps'][0]
        for key in ('transition','process_covariance','measurement','measurement_covariance'):self.assertEqual(fractions(old[key]),fractions(new[key]))
        for key in ('state_intercept','observation_intercept'):self.assertEqual(list(map(F,old[key])),list(map(F,new[key])))
        self.assertEqual(r['original_log_likelihood'],'0');self.assertEqual(r['candidate_log_likelihood'],'0')

    def test_correlated_missing_em_matches_joint_latent_and_emission_batch_oracle(self):
        from decimal import Decimal
        q=missing_case()
        mean,covariance,emission_start=batch_state_and_emissions(q,3)
        result=ss.em_step(q,['STATE_DYNAMICS','OBSERVATION_DYNAMICS']);model=result['candidate_request']['steps'][0]
        for kind in ('STATE_DYNAMICS','OBSERVATION_DYNAMICS'):
            width=1 if kind=='STATE_DYNAMICS' else 2;zz=[[F(0)]*2 for _ in range(2)];yz=[[F(0)]*2 for _ in range(width)];yy=[[F(0)]*width for _ in range(width)]
            for t in range(1,4):
                x=t-1 if kind=='STATE_DYNAMICS' else t;y=t if kind=='STATE_DYNAMICS' else emission_start+(t-1)*2
                zm=[mean[x],F(1)];ym=mean[y:y+width]
                for i in range(2):
                    for j in range(2):zz[i][j]+=zm[i]*zm[j]+(covariance[x][x] if i==0 and j==0 else 0)
                for i in range(width):
                    for j in range(2):yz[i][j]+=ym[i]*zm[j]+(covariance[y+i][x] if j==0 else 0)
                    for j in range(width):yy[i][j]+=ym[i]*ym[j]+covariance[y+i][y+j]
            inv=inverse(zz);b=[[sum(yz[i][k]*inv[k][j] for k in range(2)) for j in range(2)] for i in range(width)]
            noise=[[(yy[i][j]-sum(b[i][k]*yz[j][k]+yz[i][k]*b[j][k] for k in range(2))+
                     sum(b[i][a]*zz[a][c]*b[j][c] for a in range(2) for c in range(2)))/3 for j in range(width)] for i in range(width)]
            matrix,intercept,variance=(('transition','state_intercept','process_covariance') if kind=='STATE_DYNAMICS' else ('measurement','observation_intercept','measurement_covariance'))
            self.assertEqual(fractions(model[matrix]),[row[:1] for row in b]);self.assertEqual(list(map(F,model[intercept])),[row[-1] for row in b])
            self.assertEqual(fractions(model[variance]),noise)
            for name,expected in [('second_predictor_sum',zz),('target_predictor_sum',yz),('second_target_sum',yy)]:
                self.assertEqual(fractions(result['sufficient_statistics'][kind][name]),expected)
        self.assertGreaterEqual(Decimal(result['candidate_log_likelihood']),Decimal(result['original_log_likelihood']))
        self.assertEqual(result['observed_measurement_count'],2);self.assertEqual(result['missing_measurement_count'],4)

    def test_missing_em_observation_permutation_preserves_equivalent_updates(self):
        from decimal import Decimal
        q=missing_case();q['steps'][1]['values']['u']=3
        a=ss.em_step(q,['STATE_DYNAMICS','OBSERVATION_DYNAMICS']);q['observables'].reverse()
        for step in q['steps']:
            step['measurement'].reverse();step['observation_intercept'].reverse()
            step['measurement_covariance']=[list(reversed(row)) for row in reversed(step['measurement_covariance'])]
        b=ss.em_step(q,['STATE_DYNAMICS','OBSERVATION_DYNAMICS'])
        x=a['candidate_request']['steps'][0];y=b['candidate_request']['steps'][0]
        for key in ('transition','state_intercept','process_covariance'):self.assertEqual(x[key],y[key])
        self.assertEqual(list(reversed(x['measurement'])),y['measurement'])
        self.assertEqual(list(reversed(x['observation_intercept'])),y['observation_intercept'])
        self.assertEqual([list(reversed(row)) for row in reversed(x['measurement_covariance'])],y['measurement_covariance'])
        self.assertLess(abs(Decimal(a['candidate_log_likelihood'])-Decimal(b['candidate_log_likelihood'])),Decimal('1e-70'))

    def test_fully_unobserved_stream_is_not_identified_by_an_em_update(self):
        q=missing_case()
        for i,step in enumerate(q['steps']):step['values']={'u':i+1,'v':None}
        a=ss.em_step(q,['STATE_DYNAMICS','OBSERVATION_DYNAMICS'])
        for step in q['steps']:step.update(measurement=[[1],[-5]],observation_intercept=['1/3',3],measurement_covariance=[[1,'1/4'],['1/4',4]])
        b=ss.em_step(q,['STATE_DYNAMICS','OBSERVATION_DYNAMICS'])
        self.assertEqual(a['original_log_likelihood'],b['original_log_likelihood'])
        self.assertEqual(a['candidate_log_likelihood'],b['candidate_log_likelihood'])
        self.assertNotEqual(a['candidate_request']['steps'][0]['measurement'][1],b['candidate_request']['steps'][0]['measurement'][1])
        self.assertEqual(a['unobserved_observables'],['v'])
        self.assertEqual(a['observation_counts'],{'u':3,'v':0})
        self.assertEqual(a['parameter_identification'],'NOT_ESTABLISHED_LATENT_COORDINATE_EQUIVALENCE')

    def test_em_covariance_boundary_has_no_fabricated_likelihood_comparison(self):
        q=case();q['conditioning_end_period']=1
        r=ss.em_step(q,['OBSERVATION_DYNAMICS'])
        self.assertEqual(r['candidate_request']['steps'][0]['measurement_covariance'],[['0']])
        self.assertIsNone(r['candidate_log_likelihood'])
        self.assertEqual(r['likelihood_comparison_status'],'SINGULAR_CANDIDATE_NO_COMMON_FULL_DIMENSIONAL_DENSITY')

    def test_em_candidate_retains_exact_rationals_beyond_64_bits(self):
        q=case();q['conditioning_end_period']=2
        for i,s in enumerate(q['steps']):s['values']['y']=str(F(1,1000000007+i))
        r=ss.em_step(q,['STATE_DYNAMICS','OBSERVATION_DYNAMICS']);candidate=r['candidate_request']
        self.assertTrue(any(F(v).denominator.bit_length()>64 for key in ('transition','process_covariance','measurement_covariance') for row in candidate['steps'][0][key] for v in row))
        ss.validate(candidate);ss.filter(candidate)

    def test_multivariate_em_blocks_match_independent_batch_second_moments(self):
        q=vector_case();template=copy.deepcopy(q['steps'][0])
        for i,step in enumerate(q['steps']):
            for key,value in template.items():
                if key not in ('period','values'):step[key]=copy.deepcopy(value)
            step['values']={'u':i+2,'v':i-1}
        means,covariance=batch_condition(q,3);r=ss.em_step(q,['STATE_DYNAMICS','OBSERVATION_DYNAMICS']);n=3;p=2
        for kind in ('STATE_DYNAMICS','OBSERVATION_DYNAMICS'):
            zz=[[F(0)]*3 for _ in range(3)];yz=[[F(0)]*3 for _ in range(2)];yy=[[F(0)]*2 for _ in range(2)]
            for t,step in enumerate(q['steps'],1):
                prev=(t-1)*p if kind=='STATE_DYNAMICS' else t*p
                xm=means[prev:prev+p]+[F(1)]
                ym=(means[t*p:(t+1)*p] if kind=='STATE_DYNAMICS' else list(map(F,step['values'].values())))
                for i in range(3):
                    for j in range(3):zz[i][j]+=xm[i]*xm[j]+(covariance[prev+i][prev+j] if i<p and j<p else 0)
                for i in range(2):
                    for j in range(3):yz[i][j]+=ym[i]*xm[j]+(covariance[t*p+i][prev+j] if kind=='STATE_DYNAMICS' and j<p else 0)
                    for j in range(2):yy[i][j]+=ym[i]*ym[j]+(covariance[t*p+i][t*p+j] if kind=='STATE_DYNAMICS' else 0)
            inv=inverse(zz);b=[[sum(yz[i][k]*inv[k][j] for k in range(3)) for j in range(3)] for i in range(2)]
            noise=[[(yy[i][j]-sum(b[i][k]*yz[j][k]+yz[i][k]*b[j][k] for k in range(3))+
                     sum(b[i][a]*zz[a][c]*b[j][c] for a in range(3) for c in range(3)))/n for j in range(2)] for i in range(2)]
            model=r['candidate_request']['steps'][0];matrix='transition' if kind=='STATE_DYNAMICS' else 'measurement'
            intercept='state_intercept' if kind=='STATE_DYNAMICS' else 'observation_intercept'
            variance='process_covariance' if kind=='STATE_DYNAMICS' else 'measurement_covariance'
            self.assertEqual(fractions(model[matrix]),[row[:2] for row in b])
            self.assertEqual(list(map(F,model[intercept])),[row[-1] for row in b])
            self.assertEqual(fractions(model[variance]),noise)

    def test_em_block_order_is_simultaneous_and_entry_snapshots_include_block_list(self):
        q=case();blocks=['STATE_DYNAMICS','OBSERVATION_DYNAMICS'];a=ss.em_step(q,blocks)
        b=ss.em_step(q,list(reversed(blocks)));self.assertEqual(a['candidate_request'],b['candidate_request'])
        self.assertNotEqual(a['input_sha256'],b['input_sha256'])
        smoother=ss._run_smoother
        def mutate(*args):
            q['steps'][0]['values']['y']=999;blocks.clear()
            return smoother(*args)
        with patch.object(ss,'_run_smoother',side_effect=mutate):changed=ss.em_step(q,blocks)
        self.assertEqual(changed,a)


if __name__=='__main__':unittest.main()
