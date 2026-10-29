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
