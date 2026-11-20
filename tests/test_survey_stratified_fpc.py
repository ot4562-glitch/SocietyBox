"""Independent SRSWOR first-stage FPC and declared-design rejection oracles."""
from copy import deepcopy
from fractions import Fraction as F
from random import Random
import hashlib
import json
import threading
import time
import unittest

from econscope.execution_control import deadline_budget
from econscope.survey_stratified_fpc import (
    DESIGN, SCHEMA, stratified_psu_fpc_mean,
)
from econscope.survey_stratified_mean import stratified_psu_mean


def example():
    rows=[
        dict(unit='u1',stratum='s1',psu='a',value='0',population_weight='1',in_domain=True),
        dict(unit='u2',stratum='s1',psu='b',value='10',population_weight='1',in_domain=True),
        dict(unit='u3',stratum='s2',psu='a',value='2',population_weight='1',in_domain=True),
        dict(unit='u4',stratum='s2',psu='b',value='8',population_weight='1',in_domain=True),
    ]
    return dict(schema=SCHEMA, data_kind='SYNTHETIC',
                source_declared_sha256='a'*64,
                rights=dict(local_processing='PERMITTED',report_export='PERMITTED'),
                outcome_unit='USD',psu_sampling=DESIGN,
                population_psus_by_stratum=dict(s1=4,s2=6), rows=rows)


def wr_spec(spec):
    copy=deepcopy(spec)
    del copy['population_psus_by_stratum']
    copy['schema']='societybox-survey-stratified-psu-mean-v1'
    copy['psu_sampling']='WITH_REPLACEMENT_FIRST_STAGE_APPROXIMATION_DECLARED'
    return copy


class SRSWORPSUFinitePopulationCorrectionTests(unittest.TestCase):
    def test_two_stratum_independent_hand_calculation(self):
        result=stratified_psu_fpc_mean(example())
        self.assertEqual(result['weighted_mean_exact'],'5')
        self.assertEqual(result['uncorrected_variance_exact'],'17/2')
        self.assertEqual(result['fpc_variance_exact'],'37/8')
        self.assertEqual(result['degrees_of_freedom'],2)
        self.assertEqual(result['stratum_corrections'],[
            dict(stratum='s1',sampled_psus=2,population_psus=4,
                 first_stage_sampling_fraction_exact='1/2',
                 fpc_multiplier_exact='1/2',
                 uncorrected_variance_component_exact='25/4',
                 corrected_variance_component_exact='25/8'),
            dict(stratum='s2',sampled_psus=2,population_psus=6,
                 first_stage_sampling_fraction_exact='1/3',
                 fpc_multiplier_exact='2/3',
                 uncorrected_variance_component_exact='9/4',
                 corrected_variance_component_exact='3/2'),
        ])
        self.assertEqual(result['confidence_interval'],'NOT_CALCULATED')
        self.assertEqual(result['design_authentication'],'NOT_ESTABLISHED')
        self.assertEqual(result['original_upstream_parity'],'NOT_RUN')
        self.assertEqual(result['second_stage'],'DECLARED_PSU_CENSUS_NOT_AUTHENTICATED')

    def test_total_census_of_declared_psus_zero_first_stage_variance(self):
        spec=example()
        spec['population_psus_by_stratum']={'s1':2,'s2':2}
        v=stratified_psu_fpc_mean(spec)
        self.assertEqual(v['fpc_variance_exact'],'0')
        self.assertTrue(all(row['fpc_multiplier_exact']=='0' for row in v['stratum_corrections']))
        self.assertEqual(v['uncorrected_variance_exact'],'17/2')

    def test_one_census_stratum_does_not_erase_other_variance(self):
        spec=example()
        spec['population_psus_by_stratum']={'s1':2,'s2':6}
        v=stratified_psu_fpc_mean(spec)
        self.assertEqual(v['fpc_variance_exact'],'3/2')
        self.assertEqual([x['corrected_variance_component_exact'] for x in v['stratum_corrections']],['0','3/2'])

    def test_domain_psus_outside_domain_stay_in_fpc_counts(self):
        spec=example()
        for row in spec['rows']:
            row['in_domain']=(row['psu']=='a')
            row['value']='4' if row['stratum']=='s1' and row['in_domain'] else (
                '8' if row['stratum']=='s2' and row['in_domain'] else '9999')
        spec['population_psus_by_stratum']={'s1':4,'s2':2}
        v=stratified_psu_fpc_mean(spec)
        self.assertEqual(v['weighted_mean_exact'],'6')
        self.assertEqual(v['uncorrected_variance_exact'],'2')
        self.assertEqual(v['fpc_variance_exact'],'1/2')
        self.assertEqual([x['sampled_psus'] for x in v['stratum_corrections']],[2,2])
        self.assertEqual(v['domain_observations'],2)

    def test_read_only_data_and_full_spec_digest_changes_with_population_frame(self):
        spec=example();before=deepcopy(spec)
        a=stratified_psu_fpc_mean(spec)
        self.assertEqual(spec,before)
        expected=hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',', ':'),
                                          ensure_ascii=False,allow_nan=False).encode()).hexdigest()
        self.assertEqual(a['input_sha256'],expected)
        self.assertEqual(a['uncorrected_input_sha256'],stratified_psu_mean(wr_spec(spec))['input_sha256'])
        spec['population_psus_by_stratum']['s1']=5
        b=stratified_psu_fpc_mean(spec)
        self.assertEqual(a['weighted_mean_exact'],b['weighted_mean_exact'])
        self.assertNotEqual(a['input_sha256'],b['input_sha256'])
        self.assertNotEqual(a['fpc_variance_exact'],b['fpc_variance_exact'])
        self.assertEqual(a['uncorrected_input_sha256'],b['uncorrected_input_sha256'])

    def test_permutation_price_weight_shift_invariance(self):
        base=example()
        b=stratified_psu_fpc_mean(base)
        altered=deepcopy(base)
        altered['rows'].reverse()
        for item in altered['rows']:
            item['population_weight']='10'
            item['value']=str(int(item['value'])+777)
        a=stratified_psu_fpc_mean(altered)
        self.assertEqual(a['weighted_mean_exact'],'782')
        self.assertEqual(a['fpc_variance_exact'],b['fpc_variance_exact'])

    def test_block_unavailable_or_invalid_psu_population_frame(self):
        for counts, reason in [
            ({},'STRATUM_POPULATION_COUNTS_REQUIRED'),
            ({'s1':4},'POPULATION_STRATA_NOT_EXACTLY_MATCHED'),
            ({'s1':4,'s2':6,'extra':2},'POPULATION_STRATA_NOT_EXACTLY_MATCHED'),
            ({'s1':1,'s2':6},'POPULATION_PSU_COUNT_INVALID'),
            ({'s1':True,'s2':6},'POPULATION_PSU_COUNT_INVALID'),
            ({'s1':2.5,'s2':6},'POPULATION_PSU_COUNT_INVALID'),
            ({'s1':'4','s2':6},'POPULATION_PSU_COUNT_INVALID'),
            ({'s1':1000001,'s2':6},'POPULATION_PSU_COUNT_INVALID'),
        ]:
            with self.subTest(counts=counts),self.assertRaisesRegex(ValueError,reason):
                spec=example();spec['population_psus_by_stratum']=counts
                stratified_psu_fpc_mean(spec)
        for invalid,reason in [
            ('WR','SRSWOR_CENSUS_DESIGN_DECLARATION_REQUIRED'),
            ('PPS_SRSWOR','SRSWOR_CENSUS_DESIGN_DECLARATION_REQUIRED'),
            (1,'SRSWOR_CENSUS_DESIGN_DECLARATION_REQUIRED'),
        ]:
            with self.subTest(design=invalid),self.assertRaisesRegex(ValueError,reason):
                spec=example();spec['psu_sampling']=invalid
                stratified_psu_fpc_mean(spec)

    def test_block_population_sample_larger_than_census_and_unverified_survey_input(self):
        spec=example();spec['population_psus_by_stratum']['s1']=1
        # Invalid even before m<=N; uses stronger bound.
        with self.assertRaisesRegex(ValueError,'POPULATION_PSU_COUNT_INVALID'):
            stratified_psu_fpc_mean(spec)
        spec=example()
        spec['rows'] += [dict(unit='u5',stratum='s1',psu='c',value='4',population_weight='1',in_domain=True)]
        spec['population_psus_by_stratum']['s1']=2
        with self.assertRaisesRegex(ValueError,'POPULATION_PSU_COUNT_SMALLER_THAN_SAMPLED'):
            stratified_psu_fpc_mean(spec)
        spec=example();spec['rights']={}
        with self.assertRaisesRegex(ValueError,'RIGHTS_REQUIRED'):
            stratified_psu_fpc_mean(spec)
        spec=example();spec['rows'][0]['in_domain']='yes'
        with self.assertRaisesRegex(ValueError,'DOMAIN_INDICATOR_INVALID'):
            stratified_psu_fpc_mean(spec)

    def test_220_seeded_independent_pairwise_psu_fpc_oracles(self):
        rng=Random(20261120)
        tested=0
        for trial in range(220):
            records=[]; population={}
            for h in range(rng.randint(1,4)):
                stratum=f'h{h}'
                m=rng.randint(2,5)
                population[stratum]=m+rng.randint(0,12)
                for i in range(m):
                    for j in range(rng.randint(2,3)):
                        records.append(dict(unit=f'u{h}_{i}_{j}',stratum=stratum,psu=f'p{i}',
                                            value=str(rng.randint(-80,150)),
                                            population_weight=str(rng.randint(1,20)),
                                            in_domain=(rng.randrange(4)!=0)))
            domain_psus={(r['stratum'],r['psu']) for r in records if r['in_domain']}
            if len(domain_psus)<2:
                records[0]['in_domain']=True
                second=next(r for r in records if (r['stratum'],r['psu']) !=
                            (records[0]['stratum'],records[0]['psu']))
                second['in_domain']=True
            spec=example()
            spec['rows']=records
            spec['population_psus_by_stratum']=population
            got=stratified_psu_fpc_mean(spec)
            W=sum((F(r['population_weight']) for r in records if r['in_domain']),F(0))
            theta=sum((F(r['value'])*F(r['population_weight']) for r in records
                       if r['in_domain']),F(0))/W
            independent=F(0)
            for stratum,N in sorted(population.items()):
                psus=sorted({r['psu'] for r in records if r['stratum']==stratum})
                e=[]
                for name in psus:
                    e.append(sum((F(r['population_weight'])*(F(r['value'])-theta)
                                  for r in records if r['stratum']==stratum and r['psu']==name
                                  and r['in_domain']),F(0)))
                m=len(e)
                # Separate O(m^2) pairwise variance formula, not the production
                # centered-square algorithm. Verify FPC on every stratum.
                pairs=sum(((e[i]-e[j])**2 for i in range(m) for j in range(i+1,m)),F(0))
                independent += F(N-m,N)*pairs/F((m-1),1)/W**2
            self.assertEqual(got['weighted_mean_exact'],str(theta),msg=str(trial))
            self.assertEqual(got['fpc_variance_exact'],str(independent),msg=str(trial))
            self.assertLessEqual(F(got['fpc_variance_exact']),F(got['uncorrected_variance_exact']))
            tested+=1
        self.assertEqual(tested,220)

    def test_declared_maximum_design_budget_2048_rows_512_psus(self):
        spec=example()
        spec['rows']=[]
        spec['population_psus_by_stratum']={f'h{h}':9 for h in range(64)}
        for h in range(64):
            for psu in range(8):
                for j in range(4):
                    spec['rows'].append(dict(unit=f'u{h}_{psu}_{j}',stratum=f'h{h}',
                                             psu=f'p{psu}',value=str((h+7*psu+j)%19),
                                             population_weight='1',in_domain=True))
        got=stratified_psu_fpc_mean(spec)
        self.assertEqual(got['observations'],2048)
        self.assertEqual(got['total_psus'],512)
        self.assertEqual(got['strata_count'],64)
        self.assertEqual(got['degrees_of_freedom'],448)
        self.assertEqual(len(got['stratum_corrections']),64)
        self.assertEqual(F(got['fpc_variance_exact']),F(got['uncorrected_variance_exact'])/9)
        self.assertTrue(all(item['fpc_multiplier_exact']=='1/9'
                            for item in got['stratum_corrections']))

    def test_real_checkpoint_cancellation_and_deadline(self):
        stop=threading.Event();stop.set()
        with self.assertRaisesRegex(ValueError,'OPERATION_CANCELLED'):
            with deadline_budget(None,cancel=stop):
                stratified_psu_fpc_mean(example())
        with self.assertRaisesRegex(ValueError,'OPERATION_DEADLINE_EXCEEDED'):
            with deadline_budget(time.monotonic()-1):
                stratified_psu_fpc_mean(example())


if __name__=='__main__':
    unittest.main()
