"""Independent piecewise-line and fractional HC0 oracles for sharp RDD."""
from fractions import Fraction as F
import math
import unittest

from econscope.regression_discontinuity import sharp_local_linear


def _fraction_side(xs, ys):
    """Hand-derived unweighted OLS intercept and HC0 influence oracle."""
    n = F(len(xs)); total_x = sum(xs); total_y = sum(ys)
    total_xx = sum(x*x for x in xs); total_xy = sum(x*y for x,y in zip(xs,ys))
    determinant = n*total_xx-total_x**2
    b0 = (total_xx*total_y-total_x*total_xy)/determinant
    b1 = (n*total_xy-total_x*total_y)/determinant
    variance = sum(((total_xx-total_x*x)/determinant * (y-b0-b1*x))**2
                   for x,y in zip(xs,ys))
    return b0, variance


class SharpLocalLinearTests(unittest.TestCase):
    def test_piecewise_exact_lines_and_cutoff_assignment(self):
        x=[-.9,-.7,-.4,-.2,-.1,0,.1,.2,.4,.7,.9]
        y=[1+2*v if v < 0 else 4-v for v in x]
        for kernel in ('uniform','triangular'):
            with self.subTest(kernel=kernel):
                result=sharp_local_linear(x,y,bandwidth=1,kernel=kernel)
                self.assertAlmostEqual(result['jump'],3,places=12)
                self.assertAlmostEqual(result['left']['intercept'],1,places=12)
                self.assertAlmostEqual(result['right']['intercept'],4,places=12)
                self.assertAlmostEqual(result['variance_hc0'],0,delta=1e-27)
                self.assertEqual(result['right']['observations'],6)
                self.assertEqual(result['causal_identification'],'NOT_ESTABLISHED')
                self.assertEqual(result['p_value'],'NOT_CALCULATED')

    def test_independent_fractional_HC0_intercept_oracle(self):
        x=[F(-4,10),F(-3,10),F(-2,10),F(-1,10),
           F(1,10),F(2,10),F(3,10),F(4,10)]
        y=[F(0),F(2),F(1),F(3),F(2),F(3),F(4),F(6)]
        result=sharp_local_linear(list(map(float,x)),list(map(float,y)),
                                  bandwidth=1,kernel='uniform')
        a,va=_fraction_side(x[:4],y[:4])
        b,vb=_fraction_side(x[4:],y[4:])
        self.assertAlmostEqual(result['jump'],float(b-a),places=12)
        self.assertAlmostEqual(result['variance_hc0'],float(va+vb),places=12)
        self.assertAlmostEqual(result['left']['intercept'],float(a),places=12)
        self.assertAlmostEqual(result['right']['intercept'],float(b),places=12)

    def test_out_of_band_observations_do_not_affect_fit(self):
        x=[-.8,-.6,-.2,.1,.3,.7]
        y=[-.8,-.6,-.2,3.1,3.3,3.7]
        base=sharp_local_linear(x,y,bandwidth=1)
        extended=sharp_local_linear(x+[-9,10],y+[1e100,-1e100],bandwidth=1)
        self.assertAlmostEqual(base['jump'],extended['jump'],places=12)
        self.assertEqual(base['left']['observations'],extended['left']['observations'])

    def test_fixed_bandwidth_and_continuity_are_not_claimed(self):
        x=[-.8,-.6,-.2,.1,.3,.7]
        y=[-.8,-.6,-.2,3.1,3.3,3.7]
        result=sharp_local_linear(x,y,bandwidth=1)
        self.assertEqual(result['bandwidth_selection'],'RESEARCHER_DECLARED_NOT_OPTIMIZED')
        self.assertEqual(result['confidence_interval'],'NOT_CALCULATED_BIAS_AND_DESIGN_UNVERIFIED')
        self.assertIn('NOT_VERIFIED',result['covariance_assumptions'])

    def test_rank_support_type_and_numeric_errors_fail_closed(self):
        x=[-.8,-.6,-.2,.1,.3,.7]
        y=[-.8,-.6,-.2,3.1,3.3,3.7]
        cases=[
            ((x,y),dict(bandwidth=0),'BANDWIDTH'),
            ((x,y),dict(bandwidth=float('inf')),'FINITE'),
            ((x,y),dict(bandwidth=1,kernel='epanechnikov'),'KERNEL_UNSUPPORTED'),
            ((x+[True],y+[0]),dict(bandwidth=1),'FINITE'),
            ((x, y[:5]),dict(bandwidth=1),'DIMENSIONS'),
            (([-.8,-.6,-.2,.1,.1,.1],y),dict(bandwidth=1),'RANK'),
            ((x,y),dict(bandwidth=.1),'THREE_SUPPORTED'),
            ((x,y),dict(bandwidth=1,cutoff=float('nan')),'FINITE'),
            (([1e308]+x[1:],y),dict(bandwidth=1,cutoff=-1e308),'RUNNING_DISTANCE_RANGE'),
        ]
        for (data,options,reason) in cases:
            with self.subTest(reason=reason),self.assertRaisesRegex(ValueError,reason):
                sharp_local_linear(*data,**options)


if __name__ == '__main__': unittest.main()
