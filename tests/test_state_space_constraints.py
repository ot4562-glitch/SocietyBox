import copy
from decimal import Decimal
from fractions import Fraction as F
import unittest
from unittest.mock import patch

from econscope import state_space as ss
from tests.test_state_space import case,missing_case,iid_case,fit_options,batch_state_and_emissions,fractions,vector_case,inverse


def entry(field,row=0,column=0):return dict(field=field,row=row,column=column)


def batch_statistics(q):
    mean,covariance,start=batch_state_and_emissions(q,3)
    zz=[[F(0)]*2 for _ in range(2)];yz=[[F(0)]*2 for _ in range(2)];yy=[[F(0)]*2 for _ in range(2)]
    for t in range(1,4):
        y=start+(t-1)*2;z=[mean[t],F(1)];target=mean[y:y+2]
        for i in range(2):
            for j in range(2):
                zz[i][j]+=z[i]*z[j]+(covariance[t][t] if i==j==0 else 0)
                yz[i][j]+=target[i]*z[j]+(covariance[y+i][t] if j==0 else 0)
                yy[i][j]+=target[i]*target[j]+covariance[y+i][y+j]
    return zz,yz,yy


class StateSpaceConstraintTests(unittest.TestCase):
    def test_masked_correlated_regression_exact_two_by_two_oracle(self):
        zz=[[F(2),F(1)],[F(1),F(3)]];yz=[[F(4),F(2)],[F(1),F(6)]];yy=[[F(20),F(0)],[F(0),F(20)]]
        base=[[F(1),F(0)],[F(0),F(2)]];noise=[[F(2),F(1)],[F(1),F(3)]]
        b,r,proof=ss._em_masked_regression(zz,yz,yy,5,base,{(0,0),(1,1)},noise,True)
        # Restricted precision-weighted equations: [[9,-1],[-1,4]] [d0,h1] = [3,-4].
        self.assertEqual(b,[[F(1),F(8,35)],[F(-33,35),F(2)]])
        self.assertEqual(r,noise);self.assertTrue(proof['normal_equations_exact'])
        self.assertNotEqual(b[1][0],(yz[1][0]-base[1][1]*zz[1][0])/zz[0][0])

    def test_correlated_missing_public_step_matches_independent_batch_system(self):
        q=missing_case();zz,yz,yy=batch_statistics(q)
        mask=[entry('measurement',0,0),entry('observation_intercept',1,None)]
        result=ss.em_step(q,['OBSERVATION_DYNAMICS'],fixed_entries=mask);model=result['candidate_request']['steps'][0]
        base=[[F(1),F('1/3')],[F(2),F('-1/4')]];precision=[[F(4,3),F(-2,3)],[F(-2,3),F(4,3)]]
        fixed={(0,0),(1,1)};free=[(0,1),(1,0)]
        a=[[precision[i][k]*zz[j][l] for k,l in free] for i,j in free]
        rhs=[sum(precision[i][k]*(yz[k][j]-sum(base[k][l]*zz[l][j] for row,l in fixed if row==k)) for k in range(2)) for i,j in free]
        det=a[0][0]*a[1][1]-a[0][1]*a[1][0]
        values=[(rhs[0]*a[1][1]-rhs[1]*a[0][1])/det,(a[0][0]*rhs[1]-a[1][0]*rhs[0])/det]
        b=copy.deepcopy(base)
        for (i,j),value in zip(free,values):b[i][j]=value
        self.assertEqual(fractions(model['measurement']),[row[:1] for row in b])
        self.assertEqual(list(map(F,model['observation_intercept'])),[row[1] for row in b])
        noise=[[(yy[i][j]-sum(b[i][k]*yz[j][k]+yz[i][k]*b[j][k] for k in range(2))+sum(b[i][a]*zz[a][c]*b[j][c] for a in range(2) for c in range(2)))/3 for j in range(2)] for i in range(2)]
        self.assertEqual(fractions(model['measurement_covariance']),noise)
        self.assertGreaterEqual(Decimal(result['candidate_log_likelihood']),Decimal(result['original_log_likelihood']))
        stats=result['sufficient_statistics']['OBSERVATION_DYNAMICS']
        before=ss._expected_complete_loglik({'OBSERVATION_DYNAMICS':stats},q,['OBSERVATION_DYNAMICS'])
        after=ss._expected_complete_loglik({'OBSERVATION_DYNAMICS':stats},result['candidate_request'],['OBSERVATION_DYNAMICS'])
        self.assertGreaterEqual(Decimal(after),Decimal(before))
        self.assertIn('CONDITIONAL',stats['coefficient_maximization'])

    def test_uniform_mask_agrees_with_whole_group_fixed_fit(self):
        q=missing_case();mask=[entry('observation_intercept',i,None) for i in range(2)]
        a=ss.em_step(q,['OBSERVATION_DYNAMICS'],fixed_entries=mask)
        b=ss.em_step(q,['OBSERVATION_DYNAMICS'],fixed_parameters=['observation_intercept'])
        self.assertEqual(a['candidate_request'],b['candidate_request'])

    def test_iterated_masks_preserve_values_through_quantization_and_mix(self):
        q=iid_case()
        for step in q['steps']:step.update(process_covariance=[[9]],measurement=[['1/3']])
        mask=[entry('measurement')]
        r=ss.fit_em(q,['OBSERVATION_DYNAMICS'],fit_options(),fixed_entries=mask)
        self.assertEqual(r['status'],'CONVERGED_BY_DECLARED_CHANGE_CRITERION')
        for step in r['model_request']['steps']:self.assertEqual(step['measurement'],[['1/3']])
        self.assertAlmostEqual(float(F(r['model_request']['steps'][0]['observation_intercept'][0])),2,delta=2e-6)
        quantize=ss._quantized_request
        def overshoot(*args):
            candidate=quantize(*args)
            for step in candidate['steps']:step['observation_intercept']=['4']
            return candidate
        with patch.object(ss,'_quantized_request',side_effect=overshoot):
            mixed=ss.fit_em(q,['OBSERVATION_DYNAMICS'],fit_options(max_iterations=1),fixed_entries=mask)
        self.assertEqual(mixed['iteration_trace'][0]['step_fraction'],'1/2')
        for step in mixed['model_request']['steps']:self.assertEqual(step['measurement'],[['1/3']])

    def test_masks_snapshot_and_digest_bindings(self):
        q=missing_case();original=copy.deepcopy(q);mask=[entry('measurement',0,0)]
        before=ss.em_step(q,['OBSERVATION_DYNAMICS'],fixed_entries=mask);smoother=ss._run_smoother
        def mutate(*args,**kwargs):mask.clear();return smoother(*args,**kwargs)
        with patch.object(ss,'_run_smoother',side_effect=mutate):after=ss.em_step(q,['OBSERVATION_DYNAMICS'],fixed_entries=mask)
        self.assertEqual(before,after);self.assertEqual(q,original)
        self.assertNotEqual(before['input_sha256'],ss.em_step(q,['OBSERVATION_DYNAMICS'])['input_sha256'])

    def test_invalid_and_unsupported_entry_policies(self):
        q=missing_case()
        for mask in ([entry('measurement_covariance')],[entry('transition')],[entry('measurement',2,0)],
            [entry('measurement',0,None)],[entry('measurement'),entry('measurement')],{'measurement':[0,0]},[entry('measurement',True,0)]):
            with self.subTest(mask=mask),self.assertRaisesRegex(ValueError,'EM_FIXED_ENTRIES'):
                ss.em_step(q,['OBSERVATION_DYNAMICS'],fixed_entries=mask)
        with self.assertRaisesRegex(ValueError,'EM_FIXED_ENTRIES'):
            ss.em_step(q,['OBSERVATION_DYNAMICS'],fixed_parameters=['measurement'],fixed_entries=[entry('measurement')])
        with self.assertRaisesRegex(ValueError,'EM_NO_FREE_PARAMETERS'):
            ss.em_step(case(),['OBSERVATION_DYNAMICS'],fixed_parameters=['measurement_covariance'],fixed_entries=[entry('measurement'),entry('observation_intercept',0,None)])

    def test_all_coefficient_entries_fixed_allows_noise_only_without_design_rank(self):
        q=case();q['initial']['covariance']=[[0]];q['conditioning_end_period']=1
        for step in q['steps']:step['transition']=[[0]]
        mask=[entry('transition'),entry('state_intercept',0,None)]
        r=ss.em_step(q,['STATE_DYNAMICS'],fixed_entries=mask)
        for step in r['candidate_request']['steps']:
            self.assertEqual(step['transition'],[[0]]);self.assertEqual(step['state_intercept'],[0])
        self.assertEqual(r['sufficient_statistics']['STATE_DYNAMICS']['free_coefficient_coordinates'],[])

    def test_transition_entry_masks_match_multistate_batch_gls(self):
        q=vector_case();first=copy.deepcopy(q['steps'][0])
        for step in q['steps']:
            for field in ('transition','state_intercept','process_covariance','measurement','observation_intercept','measurement_covariance'):
                step[field]=copy.deepcopy(first[field])
        mean,covariance,_=batch_state_and_emissions(q,3);zz=[[F(0)]*3 for _ in range(3)];yz=[[F(0)]*3 for _ in range(2)]
        for t in range(1,4):
            prior=(t-1)*2;current=t*2;z=mean[prior:prior+2]+[F(1)]
            for i in range(3):
                for j in range(3):zz[i][j]+=z[i]*z[j]+(covariance[prior+i][prior+j] if i<2 and j<2 else 0)
            for i in range(2):
                for j in range(3):yz[i][j]+=mean[current+i]*z[j]+(covariance[current+i][prior+j] if j<2 else 0)
        base=[list(map(F,row))+[F(d)] for row,d in zip(first['transition'],first['state_intercept'])]
        fixed={(0,1),(1,2)};free=[(i,j) for i in range(2) for j in range(3) if (i,j) not in fixed]
        precision=inverse(fractions(first['process_covariance']))
        matrix=[[precision[i][k]*zz[j][l] for k,l in free] for i,j in free]
        rhs=[sum(precision[i][k]*(yz[k][j]-sum(base[k][l]*zz[l][j] for row,l in fixed if row==k)) for k in range(2)) for i,j in free]
        solved=[sum(a*b for a,b in zip(row,rhs)) for row in inverse(matrix)]
        expected=copy.deepcopy(base)
        for (i,j),value in zip(free,solved):expected[i][j]=value
        mask=[entry('transition',0,1),entry('state_intercept',1,None)]
        r=ss.em_step(q,['STATE_DYNAMICS'],fixed_entries=mask);model=r['candidate_request']['steps'][0]
        self.assertEqual(fractions(model['transition']),[row[:2] for row in expected])
        self.assertEqual(list(map(F,model['state_intercept'])),[row[2] for row in expected])
        self.assertGreaterEqual(Decimal(r['candidate_log_likelihood']),Decimal(r['original_log_likelihood']))

    def test_masked_step_code_integrity_and_capability_scope(self):
        code=ss._code_receipt();changed=dict(code,**{'state_space.py':'0'*64})
        with patch.object(ss,'_code_receipt',side_effect=[code,changed]):
            with self.assertRaisesRegex(ValueError,'CODE_CHANGED_DURING_CALCULATION'):
                ss.em_step(case(),['STATE_DYNAMICS'],fixed_entries=[entry('transition')])
        from econscope.study_capabilities import negotiate_capabilities
        capabilities=negotiate_capabilities(['state_space_constrained_em'])
        self.assertEqual(capabilities['status'],'SUPPORTED')
        cap=capabilities['state_space_capabilities']
        self.assertIn('PRECISION_WEIGHTED',cap['em_fixed_entry_semantics'])
        self.assertIn('EM_ELEMENTWISE_COVARIANCE_MASKS',cap['unsupported'])


if __name__=='__main__':unittest.main()
