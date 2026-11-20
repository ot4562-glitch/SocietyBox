import math
import unittest

from econscope import numeric_linear as linear


class NumericLinearTests(unittest.TestCase):
    def test_householder_fit_and_information_inverse_match_exact_small_oracle(self):
        r=linear.least_squares([[1,0],[1,1],[1,2],[1,3]],[1,2,2,4])
        for actual,expected in zip(r['coefficients'],[.9,.9]): self.assertAlmostEqual(actual,expected,places=13)
        for row,expected in zip(r['inverse_information'],[[.7,-.3],[-.3,.2]]):
            for a,b in zip(row,expected): self.assertAlmostEqual(a,b,places=13)
        self.assertAlmostEqual(r['weighted_sse'],.7,places=13)

    def test_weighted_fit_scaling_and_permutation_preserve_units(self):
        x=[[1,0],[1,1],[1,2]]; y=[1,3,2]; w=[1,2,1]
        r=linear.least_squares(x,y,weights=w)
        self.assertAlmostEqual(r['coefficients'][0],1.75,places=13)
        self.assertAlmostEqual(r['coefficients'][1],.5,places=13)
        self.assertAlmostEqual(r['inverse_information'][0][0],.75,places=13)
        scaled=linear.least_squares([[row[1]*1e8,row[0]] for row in x],y,weights=w)
        self.assertAlmostEqual(scaled['coefficients'][0]*1e8,.5,places=12)
        self.assertAlmostEqual(scaled['coefficients'][1],1.75,places=12)
        self.assertAlmostEqual(scaled['inverse_information'][0][0]*1e16,.5,places=12)
        self.assertAlmostEqual(scaled['inverse_information'][0][1]*1e8,-.5,places=12)

    def test_extreme_uniform_weight_scale_remains_finite_and_preserves_units(self):
        # Exact two-equation QR: an unscaled sqrt(w) * y would overflow
        # binary64 even though the fitted coefficients and X'WX inverse fit.
        design=[[1,0],[0,1]]; target=[2e155,3e155]
        ordinary=linear.least_squares(design,target,weights=[1,1])
        extreme=linear.least_squares(design,target,weights=[1e308,1e308])
        self.assertEqual(extreme['coefficients'],ordinary['coefficients'])
        self.assertEqual(extreme['residuals'],[0.0,0.0])
        self.assertEqual(extreme['weighted_sse'],0.0)
        self.assertEqual(extreme['inverse_information'],[[1e-308,0.0],[0.0,1e-308]])

    def test_extreme_weight_ratio_does_not_overflow_information_inverse(self):
        # Both original inverse-information diagonal entries are representable,
        # despite 616 orders of magnitude between the two weights. Taking the
        # Gram product of a max-weight-normalized inverse first would overflow.
        result=linear.least_squares([[1,0],[0,1]],[1,2],weights=[1e-308,1e308])
        self.assertEqual(result['coefficients'],[1.0,2.0])
        self.assertEqual(result['inverse_information'],[[1e308,0.0],[0.0,1e-308]])

    def test_weight_rescaling_keeps_wls_fit_and_scales_information_and_sse(self):
        x=[[1],[1]]; y=[1,2]
        base=linear.least_squares(x,y,weights=[1,1])
        scaled=linear.least_squares(x,y,weights=[1e308,1e308])
        self.assertAlmostEqual(scaled['coefficients'][0],base['coefficients'][0],places=13)
        self.assertAlmostEqual(scaled['inverse_information'][0][0]/base['inverse_information'][0][0],1e-308,delta=1e-321)
        self.assertAlmostEqual(scaled['weighted_sse']/1e308,base['weighted_sse'],places=13)

    def test_heterogeneous_weight_scaling_matches_independent_exact_wls(self):
        # Hand-derived normal equations (only used as an independent oracle):
        # X'WX = [[10,20],[20,50]], X'Wy = [27,64], det = 100.
        # beta = [7/10,1], inv(X'WX) = [[1/2,-1/5],[-1/5,1/10]].
        # weighted SSE = 21/10. The actual solver remains pivoted QR.
        x=[[1,0],[1,1],[1,2],[1,3]]; y=[1,2,2,4]; w=[1,2,3,4]
        for factor in [1.0,1e250,1e-250]:
            with self.subTest(factor=factor):
                observed=linear.least_squares(x,y,weights=[value*factor for value in w])
                for actual,expected in zip(observed['coefficients'],[.7,1.0]):
                    self.assertAlmostEqual(actual,expected,places=12)
                for row,reference in zip(observed['inverse_information'],[[.5,-.2],[-.2,.1]]):
                    for value,expected in zip(row,reference):
                        self.assertAlmostEqual(value*factor,expected,places=12)
                self.assertAlmostEqual(observed['weighted_sse']/factor,2.1,places=12)

    def test_nontrivial_column_pivoting_keeps_original_coefficient_order(self):
        x=[[v,v+e/100,1] for v,e in zip([-2,-1,0,1,3],[1,-1,2,0,-2])]
        y=[2*row[0]-row[1]+3 for row in x]
        r=linear.least_squares(x,y)
        for a,b in zip(r['coefficients'],[2,-1,3]): self.assertAlmostEqual(a,b,delta=2e-12)
        self.assertEqual(sorted(r['column_permutation']),[0,1,2])

    def test_rank_invalid_weights_and_nonfinite_data_fail(self):
        with self.assertRaisesRegex(ValueError,'RANK'): linear.least_squares([[1,1],[2,2],[3,3]],[1,2,3])
        with self.assertRaisesRegex(ValueError,'WEIGHT'): linear.least_squares([[1],[2]],[1,2],weights=[1,0])
        with self.assertRaisesRegex(ValueError,'FINITE'): linear.least_squares([[1],[math.nan]],[1,2])
        with self.assertRaisesRegex(ValueError,'DIMENSION'): linear.least_squares([[1,2],[1]],[1,2])


if __name__=='__main__': unittest.main()
