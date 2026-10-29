"""Analytic and polynomial probability oracles independent of continued fractions."""
from decimal import Decimal, localcontext
from fractions import Fraction
import math
import unittest

from econscope import inference_distributions as dist


class InferenceDistributionTests(unittest.TestCase):
    def test_f_survival_exact_analytic_families_and_reciprocity(self):
        for x in (1e-20,1e-6,.1,1,10,1e6,1e20,1e308):
            expected=2*math.atan(1/math.sqrt(x))/math.pi
            self.assertAlmostEqual(dist.f_survival(x,1,1)/expected,1,delta=3e-12)
        for df2 in (1,2,5,20,200):
            for x in (.01,.5,1,10,100):
                expected=math.exp(-df2/2*math.log1p(2*x/df2))
                self.assertAlmostEqual(dist.f_survival(x,2,df2)/expected,1,delta=3e-11)
                self.assertAlmostEqual(dist.f_survival(x,df2,2)+dist.f_survival(1/x,2,df2),1,delta=3e-13)

    def test_even_df_f_survival_matches_exact_beta_polynomial(self):
        with localcontext() as context:
            context.prec=90
            for a,b in ((1,1),(2,3),(7,4),(10,15)):
                for raw in ('0.01','0.5','2','20','1000'):
                    statistic=Decimal(raw); x=Decimal(a)/(Decimal(a)+Decimal(b)*statistic)
                    degree=a+b-1
                    expected=sum(Decimal(math.comb(degree,j))*x**j*(1-x)**(degree-j)
                                 for j in range(a,degree+1))
                    actual=dist.f_survival(float(statistic),2*b,2*a)
                    self.assertAlmostEqual(actual/float(expected),1,delta=3e-11)

    def test_f_extreme_tail_keeps_log_probability(self):
        self.assertAlmostEqual(dist.f_survival(1e300,10000,1)/(math.sqrt(2/math.pi)/1e150),1,delta=3e-5)
        # F(2,2000) SF=(1+x/1000)^(-1000); float SF underflows.
        self.assertEqual(dist.f_survival(1e8,2,2000),0)
        self.assertAlmostEqual(dist.log_f_survival(1e8,2,2000),-1000*math.log1p(1e5),delta=1e-9)

    def test_chi_square_survival_matches_even_and_odd_closed_forms(self):
        for x in (1e-6,.1,1,4,20,100):
            for df in (1,2,3,4,5,10,21):
                u=x/2
                if df%2:
                    expected=math.erfc(math.sqrt(u))
                    a=.5
                    while a<df/2:
                        expected+=math.exp(a*math.log(u)-u-math.lgamma(a+1)); a+=1
                else:
                    expected=math.exp(-u)*sum(u**j/math.factorial(j) for j in range(df//2))
                self.assertAlmostEqual(dist.chi_square_survival(x,df)/expected,1,delta=3e-12)
        self.assertEqual(dist.chi_square_survival(2000,2),0)
        self.assertEqual(dist.log_chi_square_survival(2000,2),-1000)

    def test_f_inverse_matches_analytic_quantiles_and_brackets_tail(self):
        for alpha in (1e-10,.001,.05,.5,.95,1-1e-10):
            for df2 in (1,2,6,100):
                expected=df2/2*math.expm1(-2/df2*math.log(alpha))
                record=dist.f_critical_value(alpha,2,df2)
                self.assertAlmostEqual(record['critical_value']/expected,1,delta=5e-11)
                self.assertLessEqual(record['lower'],record['critical_value'])
                self.assertGreaterEqual(record['upper'],record['critical_value'])
                self.assertGreaterEqual(dist.f_survival(record['lower'],2,df2),alpha*(1-1e-10))
                self.assertLessEqual(dist.f_survival(record['upper'],2,df2),alpha*(1+1e-10))
                self.assertLess(record['tail_relative_error'],1e-9)

    def test_probability_boundaries_and_invalid_domains(self):
        self.assertEqual(dist.f_survival(0,3,7),1)
        self.assertEqual(dist.chi_square_survival(0,3),1)
        self.assertEqual(dist.f_survival(float('inf'),3,7),0)
        self.assertEqual(dist.chi_square_survival(float('inf'),3),0)
        for x in (-1,True,float('nan'),'1'):
            for fn,args in ((dist.f_survival,(x,2,3)),(dist.chi_square_survival,(x,2))):
                with self.subTest(x=x), self.assertRaisesRegex(ValueError,'STATISTIC'):
                    fn(*args)
        for df in (0,-1,True,2.5,10001):
            with self.assertRaisesRegex(ValueError,'DEGREES_OF_FREEDOM'):
                dist.f_survival(1,df,3)
        for alpha in (0,1,True,1e-20,float('nan')):
            with self.assertRaisesRegex(ValueError,'TAIL_PROBABILITY'):
                dist.f_critical_value(alpha,2,3)

    def test_subnormal_statistic_does_not_erase_representable_log_tail(self):
        tiny=math.ulp(0.0)
        # F(1,2) SF=1-sqrt(x/(x+2)); chi-square(1) SF=erfc(sqrt(x/2)).
        expected_f=math.log1p(-math.sqrt(tiny)/math.sqrt(2))
        expected_chi=-math.sqrt(2/math.pi)*math.sqrt(tiny)
        self.assertAlmostEqual(dist.log_f_survival(tiny,1,2)/expected_f,1,delta=2e-12)
        self.assertAlmostEqual(dist.log_chi_square_survival(tiny,1)/expected_chi,1,delta=2e-12)

    def test_large_df_symmetry_and_boundary_quantile_calibration(self):
        for df in (1,20,1000,10000):
            self.assertAlmostEqual(dist.f_survival(1,df,df),.5,delta=3e-11)
        for d1,d2 in ((1,10000),(10000,1),(10000,10000)):
            for alpha in (1e-12,.05,1-1e-12):
                r=dist.f_critical_value(alpha,d1,d2)
                self.assertLess(r['tail_relative_error'],5e-10)

    def test_exact_rational_statistics_keep_log_tails_outside_float_range(self):
        tiny=Fraction(1,10**400); huge=Fraction(10**800)
        expected_small=-2/math.pi*1e-200
        self.assertAlmostEqual(dist.log_f_survival(tiny,1,1)/expected_small,1,delta=3e-12)
        self.assertAlmostEqual(dist.log_chi_square_survival(tiny,1)/(-math.sqrt(2/math.pi)*1e-200),1,delta=3e-12)
        self.assertAlmostEqual(dist.log_f_survival(huge,1,1),math.log(2/math.pi)-400*math.log(10),delta=3e-12)
        self.assertEqual(dist.f_survival(huge,1,1),0)


if __name__=='__main__': unittest.main()
