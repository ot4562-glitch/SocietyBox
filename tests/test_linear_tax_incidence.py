"""Independent analytically solvable partial-equilibrium incidence oracles."""
from fractions import Fraction as F
from random import Random
import unittest

from econscope.linear_tax_incidence import linear_tax_incidence


def case(a=10,b=1,c=0,d=1,tau=2):
    return dict(schema='societybox-linear-tax-incidence-v1',data_kind='SYNTHETIC',
                source_declared_sha256='a'*64,
                rights=dict(local_processing='PERMITTED',report_export='PERMITTED'),
                quantity_unit='kg',price_unit='USD_per_kg',
                demand=dict(quantity_intercept=str(a),price_slope=str(b)),
                supply=dict(quantity_intercept=str(c),price_slope=str(d)),
                unit_tax=str(tau))


class LinearTaxIncidenceTests(unittest.TestCase):
    def test_symmetric_linear_market_exact_welfare_oracle(self):
        r=linear_tax_incidence(case())
        self.assertEqual(r['baseline'],dict(price_exact='5',quantity_exact='5',
                                            consumer_surplus_exact='25/2',producer_surplus_exact='25/2'))
        self.assertEqual(r['after_tax'],dict(consumer_price_exact='6',producer_price_exact='4',
                                             quantity_exact='4',consumer_surplus_exact='8',
                                             producer_surplus_exact='8'))
        self.assertEqual(r['incidence']['consumer_burden_per_unit_exact'],'1')
        self.assertEqual(r['incidence']['producer_burden_per_unit_exact'],'1')
        self.assertEqual(r['welfare']['tax_revenue_exact'],'8')
        self.assertEqual(r['welfare']['deadweight_loss_exact'],'1')
        self.assertEqual(r['welfare']['total_before_exact'],'25')
        self.assertEqual(r['welfare']['total_after_including_revenue_exact'],'24')

    def test_asymmetric_slopes_exact_tax_incidence(self):
        r=linear_tax_incidence(case(20,2,2,1,3))
        self.assertEqual(r['baseline']['price_exact'],'6')
        self.assertEqual(r['baseline']['quantity_exact'],'8')
        self.assertEqual(r['after_tax']['consumer_price_exact'],'7')
        self.assertEqual(r['after_tax']['producer_price_exact'],'4')
        self.assertEqual(r['after_tax']['quantity_exact'],'6')
        self.assertEqual(r['incidence']['consumer_price_pass_through_exact'],'1/3')
        self.assertEqual(r['incidence']['consumer_burden_per_unit_exact'],'1')
        self.assertEqual(r['incidence']['producer_burden_per_unit_exact'],'2')
        self.assertEqual(r['welfare']['tax_revenue_exact'],'18')
        self.assertEqual(r['welfare']['deadweight_loss_exact'],'3')
        self.assertEqual(r['welfare']['total_before_exact'],'48')
        self.assertEqual(r['welfare']['total_after_including_revenue_exact'],'45')

    def test_zero_tax_and_supply_intercept_below_zero(self):
        r=linear_tax_incidence(case(10,1,-2,2,0))
        self.assertEqual(r['welfare']['tax_revenue_exact'],'0')
        self.assertEqual(r['welfare']['deadweight_loss_exact'],'0')
        self.assertEqual(r['incidence']['consumer_burden_per_unit_exact'],'0')
        self.assertEqual(r['incidence']['producer_burden_per_unit_exact'],'0')
        self.assertEqual(r['after_tax']['quantity_exact'],r['baseline']['quantity_exact'])

    def test_quantity_scaling_preserves_prices_and_multiplies_welfare(self):
        original=linear_tax_incidence(case(20,2,2,1,3))
        enlarged=linear_tax_incidence(case(200,20,20,10,3))
        self.assertEqual(enlarged['baseline']['price_exact'],original['baseline']['price_exact'])
        self.assertEqual(enlarged['after_tax']['consumer_price_exact'],original['after_tax']['consumer_price_exact'])
        self.assertEqual(F(enlarged['after_tax']['quantity_exact']),10*F(original['after_tax']['quantity_exact']))
        for key in ('tax_revenue_exact','deadweight_loss_exact'):
            self.assertEqual(F(enlarged['welfare'][key]),10*F(original['welfare'][key]))

    def test_invalid_domain_and_market_shutdown_fail_closed(self):
        changes=[
            ('POSITIVE_DEMAND',case(10,0,0,1,2)),
            ('POSITIVE_DEMAND',case(10,1,0,0,2)),
            ('NONNEGATIVE_UNIT_TAX',case(10,1,0,1,-1)),
            ('AFTER_TAX_INTERIOR',case(10,1,0,1,10)),
            ('BASELINE_INTERIOR',case(5,1,10,1,0)),
        ]
        for reason,s in changes:
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError,reason):
                linear_tax_incidence(s)
        for key,bad in [('unit_tax','NaN'),('unit_tax','inf'),('unit_tax',1.2)]:
            s=case();s[key]=bad
            with self.assertRaisesRegex(ValueError,'DECIMAL_STRING'):
                linear_tax_incidence(s)
        s=case();s['rights']={}
        with self.assertRaisesRegex(ValueError,'RIGHTS_REQUIRED'):
            linear_tax_incidence(s)

    def test_250_random_independent_market_clearing_and_welfare_oracles(self):
        rng=Random(20261120)
        for iteration in range(250):
            b=rng.randint(1,20); d=rng.randint(1,20)
            p0=rng.randint(5,100)
            q0=rng.randint(50,500)
            # Set curve intercepts so market (p0,q0) clears exactly.
            a=q0+b*p0; c=q0-d*p0
            tau=F(rng.randint(0,3),4)
            # The public input contract accepts finite decimal strings, not fractions.
            r=linear_tax_incidence(case(a,b,c,d,str(float(tau))))
            q1=F(q0)-F(b*d,b+d)*tau
            producer_price=F(p0)-F(b,b+d)*tau
            consumer_price=F(p0)+F(d,b+d)*tau
            self.assertEqual(F(r['baseline']['price_exact']),p0,msg=str(iteration))
            self.assertEqual(F(r['after_tax']['quantity_exact']),q1,msg=str(iteration))
            self.assertEqual(F(r['after_tax']['producer_price_exact']),producer_price,msg=str(iteration))
            self.assertEqual(F(r['after_tax']['consumer_price_exact']),consumer_price,msg=str(iteration))
            self.assertEqual(F(r['welfare']['tax_revenue_exact']),tau*q1,msg=str(iteration))
            self.assertEqual(F(r['welfare']['deadweight_loss_exact']),
                             F(b*d,2*(b+d))*tau*tau,msg=str(iteration))


if __name__=='__main__': unittest.main()
