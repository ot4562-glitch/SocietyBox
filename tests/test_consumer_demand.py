"""Exact budget, Hicksian duality, and independently derived Slutsky oracles."""
from fractions import Fraction as F
import math
import random
import unittest

from econscope.consumer_demand import marshallian_demand, hicksian_demand


class StoneGearyDemandTests(unittest.TestCase):
    def test_cobb_douglas_two_goods_closed_form(self):
        r=marshallian_demand([2,3],[.4,.6],[0,0],120)
        for actual,expected in zip(r['quantities'],[24,24]):
            self.assertAlmostEqual(actual,expected,places=12)
        u=math.exp(.4*math.log(24)+.6*math.log(24))
        self.assertAlmostEqual(r['utility'],u,places=11)
        h=hicksian_demand([2,3],[.4,.6],[0,0],u)
        self.assertAlmostEqual(h['expenditure'],120,places=11)
        for actual,expected in zip(h['quantities'],[24,24]):
            self.assertAlmostEqual(actual,expected,places=11)

    def test_stone_geary_exact_rational_quantity_and_slutsky_oracle(self):
        # Two shares 1/4 and 3/4, p=(2,4), gamma=(1,2), income=50.
        # D=40, demands (6,19/2). For two goods, compensated effects
        # S_11= -D*a1*a2/p1^2 = -15/8, S_12= D*a1*a2/(p1*p2)=15/16,
        # S_22 = -D*a1*a2/p2^2 = -15/32.
        r=marshallian_demand([2,4],[.25,.75],[1,2],50)
        self.assertAlmostEqual(r['quantities'][0],6,places=12)
        self.assertAlmostEqual(r['quantities'][1],9.5,places=12)
        expected=[[F(-15,8),F(15,16)],[F(15,16),F(-15,32)]]
        for row,reference in zip(r['slutsky_matrix'],expected):
            for actual,want in zip(row,reference):
                self.assertAlmostEqual(actual,float(want),places=12)
        h=hicksian_demand([2,4],[.25,.75],[1,2],r['utility'])
        self.assertAlmostEqual(h['expenditure'],50,places=11)
        self.assertAlmostEqual(h['quantities'][0],6,places=11)
        self.assertAlmostEqual(h['quantities'][1],9.5,places=11)

    def test_price_income_homogeneity_and_negative_semidefinite_slutsky(self):
        r=marshallian_demand([2,5],[.3,.7],[1,.5],40)
        z=marshallian_demand([20,50],[.3,.7],[1,.5],400)
        for x,y in zip(r['quantities'],z['quantities']):
            self.assertAlmostEqual(x,y,places=12)
        self.assertAlmostEqual(r['price_index']*10,z['price_index'],places=11)
        s=r['slutsky_matrix']
        self.assertLessEqual(s[0][0],0)
        self.assertLessEqual(s[1][1],0)
        self.assertAlmostEqual(s[0][1],s[1][0],places=12)
        self.assertAlmostEqual(s[0][0]*s[1][1]-s[0][1]*s[1][0],0,places=12)

    def test_zero_supernumerary_income_is_feasible(self):
        r=marshallian_demand([2,4],[.25,.75],[1,2],10)
        self.assertEqual(r['quantities'],[1.0,2.0])
        self.assertEqual(r['utility'],0)
        self.assertTrue(all(v==0 for row in r['slutsky_matrix'] for v in row))
        h=hicksian_demand([2,4],[.25,.75],[1,2],0)
        self.assertEqual(h['quantities'],[1.0,2.0])
        self.assertEqual(h['expenditure'],10)

    def test_random_duality_budget_and_euler_conditions(self):
        rng=random.Random(2047)
        for _ in range(150):
            prices=[rng.uniform(.3,6) for i in range(3)]
            raw=[rng.uniform(.2,2) for i in range(3)]
            shares=[x/sum(raw) for x in raw]
            gamma=[rng.uniform(0,3) for i in range(3)]
            income=sum(p*g for p,g in zip(prices,gamma))+rng.uniform(10,100)
            m=marshallian_demand(prices,shares,gamma,income)
            h=hicksian_demand(prices,shares,gamma,m['utility'])
            self.assertAlmostEqual(m['spending'],income,delta=1e-9)
            self.assertAlmostEqual(h['expenditure'],income,delta=1e-9)
            for i in range(3):
                self.assertAlmostEqual(h['quantities'][i],m['quantities'][i],delta=1e-9)
                self.assertAlmostEqual(shares[i]/(m['quantities'][i]-gamma[i])/prices[i],
                                       1/m['supernumerary_income'],delta=1e-12)

    def test_invalid_inputs_fail_closed(self):
        cases=[
            (([2],[1],[0],4),'DIMENSIONS_INVALID'),
            (([0,4],[.5,.5],[0,0],4),'POSITIVE_PRICES'),
            (([2,4],[.5,-.5],[0,0],4),'POSITIVE_SHARES'),
            (([2,4],[.5,.2],[0,0],4),'SHARES_SUM_ONE'),
            (([2,4],[.5,.5],[-1,0],4),'NONNEGATIVE_SUBSISTENCE'),
            (([2,4],[.5,.5],[1,2],9),'INCOME_BELOW_SUBSISTENCE'),
            (([2,4],[.5,.5],[1,2],True),'FINITE_NUMBER_REQUIRED'),
            (([2,float('nan')],[.5,.5],[1,2],10),'FINITE_NUMBER_REQUIRED'),
        ]
        for args,pattern in cases:
            with self.subTest(pattern=pattern),self.assertRaisesRegex(ValueError,pattern):
                marshallian_demand(*args)
        with self.assertRaisesRegex(ValueError,'NONNEGATIVE_UTILITY_REQUIRED'):
            hicksian_demand([2,4],[.5,.5],[1,2],-1)


if __name__=='__main__': unittest.main()
