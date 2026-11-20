"""Opt-in ORIGINAL SciPy 1.17.0 distribution-API parity for inference tails.

Run only in a separately provisioned, license-reviewed SciPy reference runtime.
A source archive, package availability or skipped test is NOT original execution
or parity evidence. This test is optional, never part of base dependencies.
"""
import math
import unittest

from econscope import inference_distributions as local

try:
    import scipy
    from scipy.stats import f as reference_f, chi2 as reference_chi2
except ImportError:
    scipy = None

REFERENCE_VERSION = '1.17.0'


@unittest.skipUnless(scipy is not None and scipy.__version__ == REFERENCE_VERSION,
                     'REFERENCE_NOT_RUN: require separately installed scipy==1.17.0')
class PinnedOriginalScipyInferenceTests(unittest.TestCase):
    def test_chi_square_log_survival_against_original_scipy(self):
        for df in (1,2,3,10,100,1000,10000):
            for statistic in (1e-6,.3,1,3,20,200,2000):
                with self.subTest(df=df,statistic=statistic):
                    expected = float(reference_chi2.logsf(statistic,df))
                    actual = local.log_chi_square_survival(statistic,df)
                    if math.isfinite(expected):
                        self.assertAlmostEqual(actual,expected,delta=1e-8+1e-11*abs(expected))

    def test_f_log_survival_against_original_scipy(self):
        for df1,df2 in ((1,1),(2,3),(3,20),(20,100),(1000,10000),(10000,10000)):
            for statistic in (1e-6,.3,1,3,20,200,2000):
                with self.subTest(df1=df1,df2=df2,statistic=statistic):
                    expected = float(reference_f.logsf(statistic,df1,df2))
                    actual = local.log_f_survival(statistic,df1,df2)
                    if math.isfinite(expected):
                        self.assertAlmostEqual(actual,expected,delta=1e-8+1e-11*abs(expected))

    def test_f_quantile_cross_checks_reference_survival_not_isf_rounding(self):
        # SciPy's isf can lose relative precision at alpha=1e-12; evaluating
        # independent scipy.stats.f.sf on our critical x avoids mistaking
        # an inverse-roundoff discrepancy for a true tail mismatch.
        for alpha in (1e-12,1e-8,.001,.05,.5,.95,1-1e-12):
            for d1,d2 in ((1,1),(2,3),(3,20),(20,100),(1000,10000),(10000,10000)):
                with self.subTest(alpha=alpha,d1=d1,d2=d2):
                    result=local.f_critical_value(alpha,d1,d2)
                    reference_tail=float(reference_f.sf(result['critical_value'],d1,d2))
                    self.assertAlmostEqual(reference_tail/alpha,1,delta=5e-8)
                    self.assertLess(result['tail_relative_error'],5e-10)


if __name__ == '__main__': unittest.main()
