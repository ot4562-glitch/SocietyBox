"""Exact independent placebo DiD differences and fail-closed support tests."""
from copy import deepcopy
from fractions import Fraction as F
from random import Random
import unittest

from econscope.did_pretrends import pretrend_placebos


def make_spec(groups=None, periods=range(5), anticipation=0, comparison='NOT_YET_TREATED'):
    if groups is None:
        groups=[('g2a',2,0),('g2b',2,10),('g3a',3,20),('g3b',3,30),('never',None,40)]
    rows=[]
    for unit, cohort, intercept in groups:
        for t in periods:
            effect=100 if cohort==2 and t>=2 else (5 if cohort==3 and t>=3 else 0)
            rows.append(dict(unit=unit, period=t, first_treated_period=cohort,
                             treated=cohort is not None and t>=cohort,
                             outcome=str(intercept+3*t+effect)))
    return dict(schema='societybox-staggered-did-v1',data_kind='SYNTHETIC',
                source_declared_sha256='a'*64,
                rights=dict(local_processing='PERMITTED',report_export='PERMITTED'),
                outcome_unit='USD',period_unit='year',comparison=comparison,
                anticipation_periods=anticipation,rows=rows)


class DiDPretrendPlaceboTests(unittest.TestCase):
    def test_zero_untreated_pretrends_not_automatic_identification(self):
        result=pretrend_placebos(make_spec())
        self.assertEqual(result['supported_cells'],3)
        self.assertEqual(result['blocked_cells'],0)
        self.assertEqual([(r['cohort'],r['end_period'],r['placebo_contrast_exact'])
                          for r in result['cohort_placebo_cells']],
                         [(2,1,'0'),(3,1,'0'),(3,2,'0')])
        self.assertEqual(result['parallel_trends'],'NOT_VERIFIED_EVEN_IF_PLACEBOS_ARE_ZERO')
        self.assertEqual(result['p_values'],'NOT_COMPUTED')
        self.assertEqual(result['original_upstream_parity'],'NOT_RUN')
        self.assertEqual(result['lead_time_aggregates'],[
            dict(lead_to_treatment=1,placebo_contrast_exact='0',cohorts=[2,3],
                 cohort_weights_exact=['1/2','1/2'],treated_units_in_support=4),
            dict(lead_to_treatment=2,placebo_contrast_exact='0',cohorts=[3],
                 cohort_weights_exact=['1'],treated_units_in_support=2)])

    def test_artificial_cohort_pretrend_violation_and_time_sensitivity(self):
        spec=make_spec()
        for row in spec['rows']:
            if row['first_treated_period']==2 and row['period']==1:
                row['outcome']=str(int(row['outcome'])+5)
        output=pretrend_placebos(spec)
        early=next(x for x in output['cohort_placebo_cells'] if (x['cohort'],x['end_period'])==(2,1))
        self.assertEqual(early['treated_change_exact'],'8')
        self.assertEqual(early['control_change_exact'],'3')
        self.assertEqual(early['placebo_contrast_exact'],'5')

    def test_never_and_not_yet_contamination_exclusion(self):
        a=pretrend_placebos(make_spec())
        g3t2=next(x for x in a['cohort_placebo_cells'] if (x['cohort'],x['end_period'])==(3,2))
        self.assertEqual(g3t2['control_units'],1)
        self.assertEqual(g3t2['placebo_contrast_exact'],'0')
        b=pretrend_placebos(make_spec(comparison='NEVER_TREATED'))
        for cell in b['cohort_placebo_cells']:
            self.assertEqual(cell['control_units'],1)
            self.assertEqual(cell['placebo_contrast_exact'],'0')

    def test_anticipation_excludes_near_treated_controls(self):
        spec=make_spec(groups=[('g1',1,0),('g3',3,10),('g4',4,20),('never',None,30)],
                       anticipation=1)
        # Earliest treated g=1 is invalid when A=1 and time starts at 0.
        with self.assertRaisesRegex(ValueError,'UNTREATED_BASELINE_MISSING'):
            pretrend_placebos(spec)
        spec=make_spec(groups=[('g3',3,0),('g4',4,10),('never',None,20)],anticipation=1)
        out=pretrend_placebos(spec)
        c=next(x for x in out['cohort_placebo_cells'] if (x['cohort'],x['end_period'])==(3,1))
        self.assertEqual(c['lead_to_treatment'],2)
        self.assertEqual(c['control_units'],2)
        self.assertEqual(c['placebo_contrast_exact'],'0')

    def test_unsupported_later_cohort_placebo_is_blocked(self):
        spec=make_spec(groups=[('g2',2,0),('g3',3,20)])
        out=pretrend_placebos(spec)
        self.assertEqual(out['supported_cells'],2)
        self.assertEqual(out['blocked_cells'],1)
        self.assertEqual(out['status'],'PARTIAL_COMPUTATION_ASSUMPTIONS_UNVERIFIED')
        blocked=next(x for x in out['cohort_placebo_cells'] if x['status']=='BLOCKED')
        self.assertEqual((blocked['cohort'],blocked['end_period']),(3,2))
        self.assertEqual(blocked['blocker'],'NO_UNEXPOSED_CONTROLS')
        self.assertIsNone(blocked['placebo_contrast_exact'])

    def test_no_available_pretrend_pairs_never_report_zero(self):
        out=pretrend_placebos(make_spec(groups=[('treated',1,0),('never',None,10)],periods=range(3)))
        self.assertEqual(out['cohort_placebo_cells'],[])
        self.assertEqual(out['lead_time_aggregates'],[])
        self.assertEqual(out['status'],'NOT_EVALUABLE_NO_CLEAN_PREPERIOD_TRANSITIONS')
        self.assertEqual(out['supported_cells'],0)

    def test_shared_additive_time_path_and_row_permutation_invariance(self):
        spec=make_spec();baseline=pretrend_placebos(spec)
        altered=deepcopy(spec)
        altered['rows']=list(reversed(altered['rows']))
        for row in altered['rows']:
            row['outcome']=str(int(row['outcome'])+13*row['period']+100)
        new=pretrend_placebos(altered)
        self.assertEqual([r['placebo_contrast_exact'] for r in new['cohort_placebo_cells']],
                         [r['placebo_contrast_exact'] for r in baseline['cohort_placebo_cells']])

    def test_randomized_independent_group_change_oracle(self):
        rng=Random(20261120)
        for trial in range(130):
            groups=[(f'g{g}_{k}',g,rng.randint(-20,20)) for g in (2,3,4,None)
                    for k in range(1+rng.randrange(1,4))]
            spec=make_spec(groups=groups,periods=range(6))
            for row in spec['rows']:
                row['outcome']=str(rng.randint(-100,100))
            raw={(r['unit'],r['period']):F(r['outcome']) for r in spec['rows']}
            by_unit={r['unit']:r['first_treated_period'] for r in spec['rows']}
            output=pretrend_placebos(spec)
            for cell in output['cohort_placebo_cells']:
                g,end=cell['cohort'],cell['end_period']
                tr=[u for u,c in by_unit.items() if c==g]
                co=[u for u,c in by_unit.items() if c!=g and (c is None or c>end)]
                before=end-1
                mean=lambda population:sum((raw[u,end]-raw[u,before] for u in population),F(0))/len(population)
                self.assertEqual(cell['treated_change_exact'],str(mean(tr)),msg=str(trial))
                self.assertEqual(cell['control_change_exact'],str(mean(co)),msg=str(trial))
                self.assertEqual(cell['placebo_contrast_exact'],str(mean(tr)-mean(co)),msg=str(trial))

    def test_invalid_panel_contract_propagates(self):
        spec=make_spec();spec['rows'][1]['treated']=True
        with self.assertRaisesRegex(ValueError,'TREATMENT_REVERSAL'):
            pretrend_placebos(spec)


if __name__=='__main__':
    unittest.main()
