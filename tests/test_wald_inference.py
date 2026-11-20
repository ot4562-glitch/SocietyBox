"""Independent analytic and rejection oracles for exact linear Wald tests."""
from fractions import Fraction as F
import math
import unittest

from econscope.wald_inference import linear_wald_test


class LinearWaldTests(unittest.TestCase):
    def test_one_parameter_chi_square1_closed_form(self):
        v=linear_wald_test([2],[[1]],[[1]],[0])
        self.assertEqual(v['wald_statistic_exact'],'4')
        self.assertEqual(v['degrees_of_freedom'],1)
        self.assertAlmostEqual(v['chi_square_upper_tail'],math.erfc(math.sqrt(2)),places=13)
        self.assertEqual(v['status'],'COMPUTED_ASSUMPTIONS_UNVERIFIED')

    def test_two_restrictions_chi_square2_exponential_oracle(self):
        v=linear_wald_test([1,4],[[1,0],[0,4]],[[1,0],[0,1]],[0,0])
        self.assertEqual(v['wald_statistic_exact'],'5')
        self.assertEqual(v['restriction_covariance_exact'],[['1','0'],['0','4']])
        self.assertAlmostEqual(v['chi_square_upper_tail'],math.exp(-2.5),places=13)

    def test_linear_contrast_and_singular_psd_covariance_restriction(self):
        v=linear_wald_test([2,3],[[1,1],[1,1]],[[1,1]],[0])
        self.assertEqual(v['wald_statistic_exact'],str(F(25,4)))
        self.assertAlmostEqual(v['chi_square_upper_tail'],math.erfc(math.sqrt(25/8)),places=12)
        x=linear_wald_test([2,3],[[2,1],[1,2]],[[1,1]],[1])
        self.assertEqual(x['wald_statistic_exact'],str(F(16,6)))

    def test_exact_multivariate_independent_2x2_inverse_oracle(self):
        beta=[F(3,2),F(-1,2)]; cov=[[F(2),F(1,2)],[F(1,2),F(3)]]
        r=[[F(2),F(1)], [F(1),F(-1)]]; null=[F(1),F(0)]
        y=[sum(r[i][k]*beta[k] for k in range(2))-null[i] for i in range(2)]
        a=[[sum(r[i][k]*cov[k][l]*r[j][l] for k in range(2) for l in range(2)) for j in range(2)] for i in range(2)]
        det=a[0][0]*a[1][1]-a[0][1]**2
        statistic=(a[1][1]*y[0]**2-2*a[0][1]*y[0]*y[1]+a[0][0]*y[1]**2)/det
        v=linear_wald_test(beta,cov,r,null)
        self.assertEqual(v['wald_statistic_exact'],str(statistic))
        self.assertAlmostEqual(v['chi_square_upper_tail'],math.exp(-float(statistic)/2),places=12)

    def test_fail_closed_invalid_or_degenerate_inputs(self):
        cases=[
            (([1],[[1]],[[1]],[]),'NULL_DIMENSIONS'),
            (([1],[[1,2]],[[1]],[0]),'COVARIANCE_DIMENSIONS'),
            (([True],[[1]],[[1]],[0]),'NUMBER_INVALID'),
            (([1],[[float('nan')]],[[1]],[0]),'NUMBER_INVALID'),
            (([1,2],[[1,2],[2,1]],[[1,0]],[0]),'COVARIANCE_NOT_POSITIVE_SEMIDEFINITE'),
            (([1,2],[[1,2],[0,1]],[[1,0]],[0]),'COVARIANCE_NOT_SYMMETRIC'),
            (([1,2],[[1,1],[1,1]],[[1,-1]],[0]),'RESTRICTION_COVARIANCE_SINGULAR'),
            (([1,2],[[1,0],[0,1]],[[1,0],[2,0]],[0,0]),'RESTRICTION_COVARIANCE_SINGULAR'),
        ]
        for args,code in cases:
            with self.subTest(code=code),self.assertRaisesRegex(ValueError,code):
                linear_wald_test(*args)


if __name__=='__main__': unittest.main()
