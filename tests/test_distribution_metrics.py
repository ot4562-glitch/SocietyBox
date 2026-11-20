"""Independent pairwise weighted-Gini and FGT poverty accounting oracles."""
from copy import deepcopy
from fractions import Fraction as F
from random import Random
import unittest

from econscope.distribution_metrics import distribution_metrics


def make_spec(values=(0,1,2,3), weights=None, poverty_line='2'):
    if weights is None: weights=(1,)*len(values)
    return dict(schema='societybox-distribution-metrics-v1',data_kind='SYNTHETIC',
                source_declared_sha256='f'*64,
                rights=dict(local_processing='PERMITTED',report_export='PERMITTED'),
                income_unit='USD',poverty_line=str(poverty_line),
                rows=[dict(unit=f'i_{i}',income=str(value),population_weight=str(weight))
                      for i,(value,weight) in enumerate(zip(values,weights))])


class DistributionMetricTests(unittest.TestCase):
    def test_four_unit_analytic_gini_lorenz_poverty_oracles(self):
        result=distribution_metrics(make_spec())
        self.assertEqual(result['weighted_mean_exact'],'3/2')
        self.assertEqual(result['gini_exact'],'5/12')
        self.assertEqual(result['weighted_median_exact'],'1')
        self.assertEqual(result['fgt_0_headcount_exact'],'1/2')
        self.assertEqual(result['fgt_1_gap_exact'],'3/8')
        self.assertEqual(result['fgt_2_severity_exact'],'5/16')
        self.assertEqual(result['lorenz'],[
            dict(population_share_exact='0',income_share_exact='0'),
            dict(population_share_exact='1/4',income_share_exact='0'),
            dict(population_share_exact='1/2',income_share_exact='1/6'),
            dict(population_share_exact='3/4',income_share_exact='1/2'),
            dict(population_share_exact='1',income_share_exact='1')])
        self.assertEqual(result['survey_sampling_uncertainty'],'NOT_COMPUTED')

    def test_unequal_weight_exact_gini_and_poverty(self):
        r=distribution_metrics(make_spec((1,3),(1,3),'2'))
        self.assertEqual(r['total_population_weight_exact'],'4')
        self.assertEqual(r['weighted_mean_exact'],'5/2')
        self.assertEqual(r['weighted_median_exact'],'3')
        self.assertEqual(r['gini_exact'],'3/20')
        self.assertEqual(r['fgt_0_headcount_exact'],'1/4')
        self.assertEqual(r['fgt_1_gap_exact'],'1/8')
        self.assertEqual(r['fgt_2_severity_exact'],'1/16')

    def test_zero_mean_blocks_gini_but_not_poverty(self):
        r=distribution_metrics(make_spec((0,0),(1,2),'10'))
        self.assertIsNone(r['gini_exact'])
        self.assertEqual(r['lorenz'],[])
        self.assertEqual(r['gini_status'],'UNDEFINED_ZERO_MEAN')
        self.assertEqual(r['fgt_0_headcount_exact'],'1')
        self.assertEqual(r['fgt_1_gap_exact'],'1')
        self.assertEqual(r['fgt_2_severity_exact'],'1')
        self.assertEqual(r['weighted_median_exact'],'0')

    def test_equal_income_gini_zero_and_strict_poverty_boundary(self):
        r=distribution_metrics(make_spec((5,5,5),(2,1,3),'5'))
        self.assertEqual(r['gini_exact'],'0')
        self.assertEqual(r['fgt_0_headcount_exact'],'0')
        self.assertEqual(r['fgt_1_gap_exact'],'0')
        self.assertEqual(r['fgt_2_severity_exact'],'0')
        self.assertEqual(r['weighted_mean_exact'],'5')
        for point in r['lorenz']:
            self.assertEqual(point['population_share_exact'],point['income_share_exact'])

    def test_permutation_price_weight_scaling_invariance(self):
        baseline=distribution_metrics(make_spec((0,3,7,14),(1,3,2,6),'10'))
        altered=make_spec((0,3,7,14),(1,3,2,6),'10')
        altered['rows']=list(reversed(altered['rows']))
        self.assertEqual(distribution_metrics(altered),baseline)
        for row in altered['rows']:
            row['income']=str(int(row['income'])*10)
            row['population_weight']=str(int(row['population_weight'])*100)
        altered['poverty_line']='100'
        changed=distribution_metrics(altered)
        for key in ('gini_exact','fgt_0_headcount_exact','fgt_1_gap_exact','fgt_2_severity_exact'):
            self.assertEqual(changed[key],baseline[key])
        self.assertEqual(F(changed['weighted_mean_exact']),10*F(baseline['weighted_mean_exact']))

    def test_reject_invalid_weights_income_poverty_line_and_duplicates(self):
        cases=[
            ('POSITIVE_POPULATION_WEIGHT',lambda s:s['rows'][0].update(population_weight='0')),
            ('NONNEGATIVE_DECIMAL',lambda s:s['rows'][0].update(population_weight='-1')),
            ('NONNEGATIVE_DECIMAL',lambda s:s['rows'][0].update(income='-3')),
            ('NONNEGATIVE_DECIMAL',lambda s:s['rows'][0].update(income='NaN')),
            ('POSITIVE_POVERTY_LINE',lambda s:s.update(poverty_line='0')),
            ('UNIT_ID_DUPLICATE',lambda s:s['rows'][1].update(unit=s['rows'][0]['unit'])),
            ('SCHEMA_INVALID',lambda s:s.update(schema='wrong')),
            ('RIGHTS_REQUIRED',lambda s:s.update(rights={})),
            ('ROW_BUDGET',lambda s:s.update(rows=[])),
        ]
        for reason,edit in cases:
            with self.subTest(reason=reason):
                s=make_spec();edit(s)
                with self.assertRaisesRegex(ValueError,reason):
                    distribution_metrics(s)

    def test_250_independent_pairwise_gini_fgt_oracles(self):
        rng=Random(20261120)
        for case in range(250):
            n=rng.randint(2,17)
            incomes=[rng.randint(0,90) for _ in range(n)]
            weights=[rng.randint(1,20) for _ in range(n)]
            line=rng.randint(1,100)
            r=distribution_metrics(make_spec(incomes,weights,line))
            total=sum(weights)
            income_total=sum(x*w for x,w in zip(incomes,weights))
            # Deliberately independent O(n^2) pairwise Gini definition.
            double=sum(weights[i]*weights[j]*abs(incomes[i]-incomes[j])
                       for i in range(n) for j in range(n))
            expected_gini=F(double,2*total*income_total) if income_total else None
            self.assertEqual(r['gini_exact'],None if expected_gini is None else str(expected_gini),msg=str(case))
            for order,key in ((0,'fgt_0_headcount_exact'),(1,'fgt_1_gap_exact'),(2,'fgt_2_severity_exact')):
                expected=sum((F(w)*F(line-x,line)**order for x,w in zip(incomes,weights) if x<line),F(0))/total
                self.assertEqual(r[key],str(expected),msg=f'{case} {key}')


if __name__=='__main__': unittest.main()
