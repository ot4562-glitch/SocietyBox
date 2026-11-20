"""Independent hand-computed ATT(g,t) and support/contract tests."""
from copy import deepcopy
from fractions import Fraction as F
import unittest

from econscope.staggered_did import cohort_time_att


def make_spec(groups=None, periods=range(5), comparison='NOT_YET_TREATED', anticipation=0):
    if groups is None:
        groups=[('early1',2,0),('early2',2,10),('late1',3,20),('late2',3,30),
                ('never1',None,40),('never2',None,50)]
    rows=[]
    for unit,cohort,intercept in groups:
        for t in periods:
            untreated=intercept+t
            effect = ((2*(t-1)) if cohort==2 and t>=2 else
                      (5 if cohort==3 and t>=3 else 0))
            rows.append(dict(unit=unit,period=t,first_treated_period=cohort,
                             treated=(cohort is not None and t>=cohort),outcome=str(untreated+effect)))
    return dict(schema='societybox-staggered-did-v1',data_kind='SYNTHETIC',
                source_declared_sha256='0'*64,
                rights=dict(local_processing='PERMITTED',report_export='PERMITTED'),
                outcome_unit='USD',period_unit='year',comparison=comparison,
                anticipation_periods=anticipation,rows=rows)


class StaggeredDifferenceInDifferencesTests(unittest.TestCase):
    def test_staggered_cohort_time_and_dynamic_exposure_aggregation_exact(self):
        r=cohort_time_att(make_spec())
        observed={(x['cohort'],x['period']):x['att_exact'] for x in r['cohort_time']}
        # Independent hand-computation from common 1-per-period untreated trend.
        self.assertEqual(observed,{(2,2):'2',(2,3):'4',(2,4):'6',(3,3):'5',(3,4):'5'})
        self.assertEqual(r['status'],'COMPUTED_ASSUMPTIONS_UNVERIFIED')
        self.assertEqual([(x['event_time'],x['att_exact']) for x in r['event_time']],
                         [(0,'7/2'),(1,'9/2'),(2,'6')])
        self.assertEqual(r['overall_cohort_mean_att_exact'],'9/2')
        self.assertEqual(r['cohort_means'][0]['mean_att_exact'],'4')
        self.assertEqual(r['parallel_trends'],'NOT_VERIFIED')
        self.assertEqual(r['standard_errors'],'NOT_IMPLEMENTED')
        self.assertEqual(r['original_upstream_parity'],'NOT_RUN')

    def test_classic_two_by_two_exact_contrast(self):
        rows=[]
        for name,cohort,y0 in [('a',1,4),('b',1,10),('c',None,20),('d',None,6)]:
            for t in range(2):
                rows.append(dict(unit=name,period=t,first_treated_period=cohort,
                                treated=(cohort is not None and t>=cohort),
                                outcome=str(y0 + 2*t + (3 if cohort==1 and t==1 else 0))))
        spec=make_spec()
        spec['rows']=rows
        r=cohort_time_att(spec)
        self.assertEqual(r['cohort_time'][0]['treated_change_exact'],'5')
        self.assertEqual(r['cohort_time'][0]['control_change_exact'],'2')
        self.assertEqual(r['cohort_time'][0]['att_exact'],'3')
        self.assertEqual(r['overall_cohort_mean_att_exact'],'3')

    def test_never_vs_not_yet_controls_selected_with_correct_timing(self):
        spec=make_spec()
        # Extra pre-adoption outcome jump among the late cohort changes the
        # not-yet-treated comparison at t=2 but never-treated is unaffected.
        for row in spec['rows']:
            if row['first_treated_period']==3 and row['period']==2:
                row['outcome']=str(int(row['outcome'])+7)
        not_yet=cohort_time_att(spec)
        spec['comparison']='NEVER_TREATED'
        never=cohort_time_att(spec)
        a=next(x for x in not_yet['cohort_time'] if (x['cohort'],x['period'])==(2,2))
        b=next(x for x in never['cohort_time'] if (x['cohort'],x['period'])==(2,2))
        self.assertEqual(a['control_units'],4)
        self.assertEqual(b['control_units'],2)
        self.assertEqual(a['att_exact'],str(F(2)-F(7,2)))
        self.assertEqual(b['att_exact'],'2')
        self.assertEqual(a['baseline_period'],1)

    def test_no_never_treated_reports_later_unsupported_cells(self):
        groups=[('a',2,0),('b',2,10),('c',3,20),('d',3,30)]
        spec=make_spec(groups=groups)
        result=cohort_time_att(spec)
        self.assertEqual(result['supported_cells'],1)
        self.assertEqual(result['blocked_cells'],4)
        self.assertEqual(result['status'],'PARTIAL_COMPUTATION_ASSUMPTIONS_UNVERIFIED')
        blocked=[r for r in result['cohort_time'] if r['att_exact'] is None]
        self.assertTrue(all(x['blocker']=='NO_UNEXPOSED_CONTROLS_AT_PERIOD' for x in blocked))
        self.assertEqual(result['event_time'][0]['contributing_cohorts'],[2])
        # A single treated cohort with no controls cannot be estimated.
        with self.assertRaisesRegex(ValueError,'NO_SUPPORTED_COHORT_TIME'):
            cohort_time_att(make_spec(groups=groups[:2]))

    def test_anticipation_baseline_and_future_control_exclusion(self):
        # g=3, A=1 gives baseline 1 and excludes cohort 4 at t=3.
        groups=[('a',3,0),('b',4,10),('c',None,20)]
        spec=make_spec(groups=groups,periods=range(5),anticipation=1)
        r=cohort_time_att(spec)
        entry=next(x for x in r['cohort_time'] if x['cohort']==3 and x['period']==3)
        self.assertEqual(entry['baseline_period'],1)
        self.assertEqual(entry['control_units'],1)
        self.assertEqual(r['treatment_anticipation_assumption'],'DECLARED_NOT_VERIFIED')

    def test_permutation_and_additive_common_trend_invariance(self):
        spec=make_spec(); ref=cohort_time_att(spec)
        modified=deepcopy(spec)
        modified['rows']=list(reversed(modified['rows']))
        for row in modified['rows']:
            row['outcome']=str(int(row['outcome'])+1234+17*row['period'])
        new=cohort_time_att(modified)
        self.assertEqual([v['att_exact'] for v in ref['cohort_time']],
                         [v['att_exact'] for v in new['cohort_time']])
        self.assertEqual(ref['event_time'],new['event_time'])

    def test_invalid_panel_treatment_reversal_earliest_and_missing(self):
        initial=make_spec()
        changes=[
            ('TREATMENT_REVERSAL',lambda s:s['rows'][0].update(treated=True)),
            ('INCONSISTENT_COHORT',lambda s:s['rows'][0].update(first_treated_period=4)),
            ('DUPLICATE_UNIT_PERIOD',lambda s:s['rows'].append(deepcopy(s['rows'][0]))),
            ('UNBALANCED_PANEL',lambda s:s['rows'].pop()),
            ('UNTREATED_BASELINE',lambda s:[r.update(first_treated_period=0,treated=True) for r in s['rows'] if r['unit']=='early1']),
            ('OUTCOME_DECIMAL',lambda s:s['rows'][0].update(outcome='NaN')),
            ('RIGHTS_REQUIRED',lambda s:s.update(rights={})),
            ('ANTICIPATION_INVALID',lambda s:s.update(anticipation_periods=True)),
        ]
        for error,modify in changes:
            with self.subTest(error=error):
                s=deepcopy(initial)
                modify(s)
                with self.assertRaisesRegex(ValueError,error):
                    cohort_time_att(s)

    def test_randomized_independent_group_mean_oracle(self):
        from random import Random
        rng=Random(20261120)
        cohorts=(2,3,4,None)
        for iteration in range(120):
            s=make_spec()
            s['rows']=[]
            groups={}
            for g in cohorts:
                for k in range(1+rng.randrange(1,4)):
                    unit=f'u_{g}_{k}'
                    outcomes={t:rng.randint(-100,100) for t in range(6)}
                    groups[unit]=(g,outcomes)
                    for t in range(6):
                        s['rows'].append(dict(unit=unit,period=t,first_treated_period=g,
                                            treated=g is not None and t>=g,outcome=str(outcomes[t])))
            calculated=cohort_time_att(s)
            for cell in calculated['cohort_time']:
                g,t=cell['cohort'],cell['period']
                baseline=g-1
                treated=[outcomes[t]-outcomes[baseline] for cohort,outcomes in groups.values() if cohort==g]
                controls=[outcomes[t]-outcomes[baseline] for cohort,outcomes in groups.values()
                          if cohort is None or cohort>t]
                expected=F(sum(treated),len(treated))-F(sum(controls),len(controls))
                self.assertEqual(cell['att_exact'],str(expected),msg=f'{iteration} {g} {t}')

    def test_nonconsecutive_panel_and_untrusted_inference(self):
        s=make_spec(periods=(0,1,3,4))
        with self.assertRaisesRegex(ValueError,'NONCONSECUTIVE_PERIODS'):
            cohort_time_att(s)
        self.assertEqual(cohort_time_att(make_spec())['source_authentication'],'NOT_ESTABLISHED')
        # A giant gap is rejected without allocating one million period slots.
        extreme=make_spec(periods=(0,1000000))
        with self.assertRaisesRegex(ValueError,'NONCONSECUTIVE_PERIODS'):
            cohort_time_att(extreme)


if __name__=='__main__':
    unittest.main()
