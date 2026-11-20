"""Two-way clustering: exact independent pairwise influence oracles."""
from fractions import Fraction as F
import unittest

from econscope.multiway_cluster import two_way_cluster_covariance


class TwoWayClusterTests(unittest.TestCase):
    def setUp(self):
        self.scores=[[1,0],[1,1],[0,1],[1,0],[1,2],[0,3]]
        self.a=['A','A','B','B','C','C']
        self.b=['x','y','x','y','x','y']
        self.bread=[[2,1],[1,2]]

    def test_two_way_exact_cgm_inclusion_exclusion_and_sandwich(self):
        result=two_way_cluster_covariance(self.scores,self.bread,self.a,self.b)
        self.assertEqual(result['covariance_exact'],[['153','189'],['189','234']])
        self.assertEqual(result['cluster_meat_exact'],[['10','19'],['19','37']])
        self.assertEqual(result['cluster_counts'],{'a':3,'b':2,'intersection':6})
        # Independent ordered-pairs oracle; no use of clustered group sums.
        p=2; meat=[[F(0)]*p for _ in range(p)]
        for i,si in enumerate(self.scores):
            for j,sj in enumerate(self.scores):
                kernel=int(self.a[i]==self.a[j])+int(self.b[i]==self.b[j])-int(
                    (self.a[i],self.b[i])==(self.a[j],self.b[j]))
                for u in range(p):
                    for v in range(p): meat[u][v]+=kernel*si[u]*sj[v]
        expected=[[sum(self.bread[i][k]*meat[k][l]*self.bread[j][l]
                       for k in range(p) for l in range(p)) for j in range(p)] for i in range(p)]
        self.assertEqual(result['covariance_exact'],[list(map(str,row)) for row in expected])

    def test_identical_partition_collapses_to_one_way_cluster(self):
        out=two_way_cluster_covariance(self.scores,self.bread,self.a,self.a)
        self.assertEqual(out['cluster_counts'],{'a':3,'b':3,'intersection':3})
        sums=[[sum(s[j] for s,g in zip(self.scores,self.a) if g==tag)
               for j in range(2)] for tag in ('A','B','C')]
        meat=[[sum(s[i]*s[j] for s in sums) for j in range(2)] for i in range(2)]
        self.assertEqual(out['cluster_meat_exact'],[list(map(str,row)) for row in meat])

    def test_scalar_analytic_weights_exact_simple(self):
        score=[[1],[2],[-1],[3]]
        a=['a','a','b','b']; b=['x','y','x','y']
        result=two_way_cluster_covariance(score,[[F(1,2)]],a,b)
        # One dimensional independent pairwise kernel oracle.
        meat=sum((int(a[i]==a[j])+int(b[i]==b[j])-int(
            (a[i],b[i])==(a[j],b[j])))*score[i][0]*score[j][0]
            for i in range(4) for j in range(4))
        self.assertEqual(result['covariance_exact'],[[str(F(meat,4))]])

    def test_block_invalid_dimensions_bread_and_groups(self):
        cases=[
            (([[1,2]],[[1,0],[0,1]],['a'],['b']),'DIMENSIONS'),
            ((self.scores,[[1,2],[0,1]],self.a,self.b),'BREAD_NOT_SYMMETRIC'),
            ((self.scores,[[0,0],[0,1]],self.a,self.b),'BREAD_NOT_POSITIVE_DEFINITE'),
            ((self.scores,self.bread,['a']*6,self.b),'AT_LEAST_TWO_CLUSTERS'),
            ((self.scores,self.bread,['invalid space']*6,self.b),'GROUPS_LABELS_INVALID'),
            ((self.scores,self.bread,self.a,['x']*5),'GROUPS_DIMENSIONS_INVALID'),
            (([[1],[float('nan')]],[[1]],['a','b'],['x','y']),'NUMBER_INVALID'),
        ]
        for args,message in cases:
            with self.subTest(message=message),self.assertRaisesRegex(ValueError,message):
                two_way_cluster_covariance(*args)

    def test_reject_indefinite_two_way_meat(self):
        # Inclusion-exclusion is not automatically positive semidefinite.
        s=[[1,0],[1,1],[0,1],[1,0],[0,1],[1,1]]
        with self.assertRaisesRegex(ValueError,'CLUSTER_MEAT_NOT_POSITIVE_SEMIDEFINITE'):
            two_way_cluster_covariance(s,[[1,0],[0,1]],self.a,self.b)


if __name__=='__main__': unittest.main()
