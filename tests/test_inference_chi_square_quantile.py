"""Independent analytic critical-value checks for central chi-square inference.

No third-party numerical package is needed for this regression test. SciPy
comparisons are an optional separate numerical-parity exercise, not a claim
of original upstream execution by the project.
"""
from fractions import Fraction
import math
import unittest

from econscope import inference_distributions as dist


class ChiSquareCriticalValueTests(unittest.TestCase):
    def test_df2_exponential_analytic_critical_values(self):
        # Chi-square(2) SF(x) = exp(-x/2), so its upper-tail quantile is
        # exactly -2 log(alpha).
        for alpha in (1e-12, 1e-9, .001, .05, .5, .95, 1-1e-12):
            with self.subTest(alpha=alpha):
                result = dist.chi_square_critical_value(alpha,2)
                expected = -2*math.log(alpha)
                self.assertAlmostEqual(result['critical_value']/expected,1,delta=3e-9)
                self.assertLessEqual(result['lower'],result['critical_value'])
                self.assertGreaterEqual(result['upper'],result['critical_value'])
                self.assertLess(result['tail_relative_error'],5e-10)

    def test_df1_and_df4_independent_probability_formulas(self):
        # No use of the implementation's gamma fractions for expected SF.
        for df in (1,4):
            for alpha in (1e-12,1e-6,.05,.5,.95,1-1e-12):
                with self.subTest(df=df,alpha=alpha):
                    got=dist.chi_square_critical_value(alpha,df)
                    x=got['critical_value']
                    expected=(math.erfc(math.sqrt(x/2)) if df==1
                              else math.exp(-x/2)*(1+x/2))
                    self.assertAlmostEqual(expected/alpha,1,delta=5e-10)

    def test_large_df_and_monotonic_quantile_brackets(self):
        for df in (1,20,1000,10000):
            values=[]
            for alpha in (1e-12,.001,.05,.5,.95,1-1e-12):
                with self.subTest(df=df,alpha=alpha):
                    rec=dist.chi_square_critical_value(alpha,df)
                    values.append(rec['critical_value'])
                    self.assertLessEqual(rec['lower'],rec['critical_value'])
                    self.assertGreaterEqual(rec['upper'],rec['critical_value'])
                    self.assertLess(rec['tail_relative_error'],5e-10)
                    self.assertEqual(rec['degrees_of_freedom'],df)
            self.assertEqual(values,sorted(values,reverse=True))

    def test_domain_rejections_and_fraction_input(self):
        for alpha in (True,0,1,-.1,1e-13,math.nan,math.inf,'0.05'):
            with self.subTest(alpha=alpha),self.assertRaisesRegex(ValueError,'TAIL_PROBABILITY'):
                dist.chi_square_critical_value(alpha,2)
        for df in (True,0,-3,1.5,10001):
            with self.subTest(df=df),self.assertRaisesRegex(ValueError,'DEGREES_OF_FREEDOM'):
                dist.chi_square_critical_value(.05,df)
        value=dist.chi_square_critical_value(Fraction(1,20),2)['critical_value']
        self.assertAlmostEqual(value,-2*math.log(.05),places=11)


if __name__=='__main__': unittest.main()
