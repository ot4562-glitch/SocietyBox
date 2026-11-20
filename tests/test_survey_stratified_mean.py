"""Independent exact survey-stratum PSU Taylor mean-variance oracles."""
from copy import deepcopy
from fractions import Fraction as F
from random import Random
import threading
import time
import unittest

from econscope.execution_control import deadline_budget
from econscope.survey_stratified_mean import stratified_psu_mean, validate


def case(rows=None):
    if rows is None:
        rows = [
            dict(unit='u1', stratum='s1', psu='A', value='0', population_weight='1', in_domain=True),
            dict(unit='u2', stratum='s1', psu='B', value='10', population_weight='1', in_domain=True),
            dict(unit='u3', stratum='s2', psu='A', value='2', population_weight='1', in_domain=True),
            dict(unit='u4', stratum='s2', psu='B', value='8', population_weight='1', in_domain=True),
        ]
    return dict(schema='societybox-survey-stratified-psu-mean-v1',
                data_kind='SYNTHETIC',source_declared_sha256='a'*64,
                rights=dict(local_processing='PERMITTED',report_export='PERMITTED'),
                outcome_unit='USD',
                psu_sampling='WITH_REPLACEMENT_FIRST_STAGE_APPROXIMATION_DECLARED',
                rows=rows)


class StratifiedSurveyMeanTests(unittest.TestCase):
    def test_exact_two_stratum_psu_variance_hand_oracle(self):
        value=stratified_psu_mean(case())
        self.assertEqual(value['weighted_mean_exact'],'5')
        self.assertEqual(value['variance_exact'],'17/2')
        self.assertEqual(value['degrees_of_freedom'],2)
        self.assertEqual(value['strata_count'],2)
        self.assertEqual(value['total_psus'],4)
        self.assertEqual(value['stratum_variance_components'],[
            dict(stratum='s1',psu_count=2,variance_component_exact='25/4'),
            dict(stratum='s2',psu_count=2,variance_component_exact='9/4')])
        self.assertEqual(value['design_authentication'],'NOT_ESTABLISHED')
        self.assertEqual(value['confidence_interval'],'NOT_CALCULATED')
        self.assertEqual(value['original_upstream_parity'],'NOT_RUN')
        self.assertNotIn('u1',repr(value))
        self.assertIsNone(validate(case()))

    def test_domain_retains_out_of_domain_psu_contributions(self):
        q=case()
        for row in q['rows']:
            row['in_domain']=(row['unit'] in ('u1','u3'))
            if not row['in_domain']:
                row['value']='1000'
            elif row['unit']=='u1':
                row['value']='4'
            else:
                row['value']='8'
        value=stratified_psu_mean(q)
        # Domain point=6; PSU residuals (-2,0) and (+2,0) in two strata.
        # Each stratum contributes (2)^2 / (W=2)^2 = 1.
        self.assertEqual(value['weighted_mean_exact'],'6')
        self.assertEqual(value['variance_exact'],'2')
        self.assertEqual(value['domain_observations'],2)
        self.assertEqual(value['total_psus'],4)
        self.assertEqual(value['stratum_variance_components'][0]['variance_component_exact'],'1')

    def test_within_psu_multiunit_linearization_and_weights(self):
        q=case(rows=[
            dict(unit='a1',stratum='h',psu='a',value='1',population_weight='1',in_domain=True),
            dict(unit='a2',stratum='h',psu='a',value='4',population_weight='2',in_domain=True),
            dict(unit='b1',stratum='h',psu='b',value='7',population_weight='1',in_domain=True),
            dict(unit='b2',stratum='h',psu='b',value='2',population_weight='1',in_domain=True),
        ])
        value=stratified_psu_mean(q)
        self.assertEqual(value['weighted_mean_exact'],'18/5')
        # Residual PSU totals: 1+8 -3*(18/5) = -9/5;
        # 7+2 -2*(18/5) = 9/5.  Two-PSU WR variance = (18/5)^2/5^2.
        self.assertEqual(value['variance_exact'],'324/625')

    def test_permutation_weight_rescale_and_outcome_shift_invariance(self):
        original=stratified_psu_mean(case())
        q=case();q['rows']=list(reversed(q['rows']))
        revised=stratified_psu_mean(q)
        self.assertEqual(revised['variance_exact'],original['variance_exact'])
        self.assertEqual(revised['weighted_mean_exact'],original['weighted_mean_exact'])
        for row in q['rows']:
            row['value']=str(int(row['value'])+100)
            row['population_weight']='100'
        after=stratified_psu_mean(q)
        self.assertEqual(after['variance_exact'],original['variance_exact'])
        self.assertEqual(after['weighted_mean_exact'],'105')
        self.assertEqual(after['domain_population_weight_exact'],'400')

    def test_sampled_stratum_without_domain_still_retained(self):
        q=case()
        for row in q['rows']:
            if row['stratum']=='s2':
                row['in_domain']=False
        value=stratified_psu_mean(q)
        self.assertEqual(value['domain_observations'],2)
        self.assertEqual(value['weighted_mean_exact'],'5')
        self.assertEqual(value['variance_exact'],'25')
        self.assertEqual(value['strata_count'],2)
        self.assertEqual(value['degrees_of_freedom'],2)
        self.assertEqual(value['stratum_variance_components'][1]['variance_component_exact'],'0')

    def test_explicit_lack_of_domain_support_or_lonely_psu_rejected(self):
        q=case()
        for row in q['rows'][1:]:
            row['in_domain']=False
        with self.assertRaisesRegex(ValueError,'DOMAIN_PSU_SUPPORT_INSUFFICIENT'):
            stratified_psu_mean(q)
        q=case()
        for row in q['rows']:
            row['in_domain']=False
        with self.assertRaisesRegex(ValueError,'EMPTY_DOMAIN'):
            stratified_psu_mean(q)
        q=case();q['rows'][1]['psu']='A'
        with self.assertRaisesRegex(ValueError,'LONELY_PSU'):
            stratified_psu_mean(q)

    def test_input_schema_rejections(self):
        variants=[]
        def alter(fn, reason):
            q=case();fn(q);variants.append((q,reason))
        alter(lambda q:q['rows'][0].update(value='nan'),'NUMBER_INVALID')
        alter(lambda q:q['rows'][0].update(value=True),'NUMBER_INVALID')
        alter(lambda q:q['rows'][0].update(value='1e5'),'NUMBER_INVALID')
        alter(lambda q:q['rows'][0].update(population_weight='0'),'POSITIVE_WEIGHT_REQUIRED')
        alter(lambda q:q['rows'][0].update(population_weight='-2'),'POSITIVE_WEIGHT_REQUIRED')
        alter(lambda q:q['rows'][0].update(in_domain='true'),'DOMAIN_INDICATOR_INVALID')
        alter(lambda q:q['rows'][0].update(unit='u2'),'DUPLICATE_UNIT')
        alter(lambda q:q['rows'][0].update(psu='A/B'),'IDENTIFIER_INVALID')
        alter(lambda q:q['rows'].pop(),'ROW_BUDGET')
        alter(lambda q:q.update(source_declared_sha256='invalid'),'SOURCE_DIGEST_REQUIRED')
        alter(lambda q:q.update(rights={}),'RIGHTS_REQUIRED')
        alter(lambda q:q.update(psu_sampling='SRS'),'DESIGN_DECLARATION_REQUIRED')
        alter(lambda q:q.update(schema='wrong'),'SCHEMA_INVALID')
        alter(lambda q:q.update(outcome_unit='oops\n'),'UNIT_INVALID')
        alter(lambda q:q.update(extra='unknown'),'SCHEMA_INVALID')
        for q, reason in variants:
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError,reason):
                stratified_psu_mean(q)

    def test_240_randomized_independent_pairwise_psu_oracles(self):
        # Separate derivation: for PSU linearized totals e_i,
        # sum_i(e_i-mean(e))^2 = sum_{i<j}(e_i-e_j)^2 / m.
        # Thus V_h = sum_{i<j}(e_i-e_j)^2 / [(m-1)*W_domain^2].
        rng=Random(20261120)
        for example in range(240):
            rows=[]
            mstrata=rng.randint(1,4)
            for h in range(mstrata):
                for psu in range(rng.randint(2,5)):
                    for obs in range(rng.randint(1,4)):
                        rows.append(dict(unit=f'u{h}_{psu}_{obs}',stratum=f'h{h}',
                                         psu=f'p{psu}',value=str(rng.randint(-80,150)),
                                         population_weight=str(rng.randint(1,20)),
                                         in_domain=rng.randrange(4)!=0))
            domain={ (r['stratum'],r['psu']) for r in rows if r['in_domain'] }
            if len(domain)<2:
                continue
            q=case(rows=rows)
            out=stratified_psu_mean(q)
            W=sum((F(r['population_weight']) for r in rows if r['in_domain']),F(0))
            point=sum((F(r['value'])*F(r['population_weight']) for r in rows
                       if r['in_domain']),F(0))/W
            variance=F(0)
            for h in sorted(set(r['stratum'] for r in rows)):
                psus=sorted(set(r['psu'] for r in rows if r['stratum']==h))
                e=[]
                for p in psus:
                    e.append(sum((F(r['population_weight'])*(F(r['value'])-point)
                                  for r in rows if r['stratum']==h and r['psu']==p
                                  and r['in_domain']),F(0)))
                variance+=sum(((e[i]-e[j])**2 for i in range(len(e))
                               for j in range(i+1,len(e))),F(0)) / ((len(e)-1)*W*W)
            self.assertEqual(out['weighted_mean_exact'],str(point),msg=str(example))
            self.assertEqual(out['variance_exact'],str(variance),msg=str(example))
            self.assertEqual(out['degrees_of_freedom'],len({(r['stratum'],r['psu']) for r in rows})-mstrata)

    def test_cancel_and_deadline_production_checkpoint_contract(self):
        event=threading.Event();event.set()
        with self.assertRaisesRegex(ValueError,'OPERATION_CANCELLED'):
            with deadline_budget(None,cancel=event):
                stratified_psu_mean(case())
        with self.assertRaisesRegex(ValueError,'OPERATION_DEADLINE_EXCEEDED'):
            with deadline_budget(time.monotonic()-1):
                stratified_psu_mean(case())


if __name__=='__main__':
    unittest.main()
