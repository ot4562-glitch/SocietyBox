"""Mundlak augmentation versus explicit dense GLS and independent within OLS."""
import copy
from fractions import Fraction as F
import importlib
import math
import random
import unittest
from unittest.mock import patch

from tests.test_random_effects import case,dense_oracle,inverse,mm,tr,raw
from econscope.workspace import canonical,digest


def api():return importlib.import_module('econscope.correlated_random_effects')


def augmented(q):
    """Independent explicit augmentation for full-rank test fixtures only."""
    q=copy.deepcopy(q);labels=list(q['model']['regressors']);groups=sorted({r['entity'] for r in q['rows']});selected=[]
    for j,label in enumerate(labels):
        means={g:sum(F(r['regressors'][label]) for r in q['rows'] if r['entity']==g)/sum(r['entity']==g for r in q['rows']) for g in groups}
        if all(F(r['regressors'][label])==means[r['entity']] for r in q['rows']):continue
        name='oracle_mean_'+str(j);selected.append(label);q['model']['regressors'].append(name);q['units'][name]=q['units'][label]
        for r in q['rows']:r['regressors'][name]=str(means[r['entity']])
    return q,selected


def within_oracle(q,labels):
    x=[];y=[]
    for entity in sorted({r['entity'] for r in q['rows']}):
        rows=[r for r in q['rows'] if r['entity']==entity]
        mx=[sum(F(r['regressors'][v]) for r in rows)/len(rows) for v in labels];my=sum(F(r['outcome']) for r in rows)/len(rows)
        for r in rows:x.append([F(r['regressors'][v])-m for v,m in zip(labels,mx)]);y.append([F(r['outcome'])-my])
    return [row[0] for row in mm(mm(inverse(mm(tr(x),x)),tr(x)),y)]


def maximum_case(feasible=False):
    q=case(feasible);rng=random.Random(312944);q['model']['regressors']=['a','b','c','z']
    q['units']={'y':'units',**{v:'units' for v in q['model']['regressors']}}
    q['rows']=[dict(entity=str(i),period=5*t,outcome=str(5*(i%9)+rng.randint(-10,10)),
        regressors={**{v:str(rng.randint(-6,6)) for v in ['a','b','c']},'z':str(i%2)}) for i in range(128) for t in range(4)]
    return q


class CorrelatedRandomEffectsTests(unittest.TestCase):
    def test_balanced_and_unbalanced_coefficients_and_covariance_match_dense_augmented_gls(self):
        for ub in (False,True):
            for cov in ('MODEL_BASED','ENTITY_CR0','ENTITY_CR1'):
                q=case(unbalanced=ub);q['covariance']=cov;r=api().calculate(q);a,labels=augmented(q);o=dense_oracle(a,F(2),F(3))
                self.assertEqual(list(map(F,r['coefficients_exact'])),o['beta'])
                self.assertEqual([[F(v) for v in row] for row in r['covariance_exact']],o['cov'])
                self.assertEqual(r['included_mean_regressors'],labels)
                self.assertEqual(F(r['gls_objective_exact']),o['objective'])

    def test_time_varying_slopes_equal_independent_within_ols_for_any_declared_variance(self):
        for ub in (False,True):
            q=case(unbalanced=ub)
            for su in ('0','3','1000000'):
                q['variance_components']['entity']=su;r=api().calculate(q)
                self.assertEqual(F(r['within_coefficients_exact']['x']),within_oracle(q,['x'])[0])
                self.assertEqual(r['within_slope_equivalence'],'VERIFIED_EXACT_ON_OBSERVED_SAMPLE')
                self.assertEqual(r['within_normal_residuals_exact'],['0'])

    def test_joint_mean_wald_statistic_and_probability_match_dense_cofactor_oracle(self):
        q=case();q['covariance']='ENTITY_CR1';r=api().calculate(q);a,_=augmented(q);o=dense_oracle(a,F(2),F(3))
        t=r['mundlak_test'];expected=o['beta'][-1]**2/o['cov'][-1][-1]
        self.assertEqual(F(t['chi_squared_statistic_exact']),expected);self.assertEqual(t['degrees_of_freedom'],1)
        self.assertAlmostEqual(t['p_value'],math.erfc(math.sqrt(float(expected)/2)),places=13)

    def test_feasible_variance_components_use_same_original_within_between_spaces(self):
        from econscope import random_effects as re
        for ub in (False,True):
            q=case(feasible=True,unbalanced=ub);r=api().calculate(q);basic=re.calculate(q);m=r['variance_components']
            self.assertEqual(m,basic['variance_components'])
            a,_=augmented(q);o=dense_oracle(a,F(m['idiosyncratic_exact']),F(m['entity_exact']))
            self.assertEqual(list(map(F,r['coefficients_exact'])),o['beta'])
            self.assertEqual(F(r['within_coefficients_exact']['x']),within_oracle(q,['x'])[0])

    def test_time_invariant_and_absorbed_means_are_explicitly_excluded_without_false_test(self):
        q=case()
        for row in q['rows']:row['regressors']['x']=str(row['period']-1)
        r=api().calculate(q)
        self.assertEqual(r['included_mean_regressors'],[])
        self.assertEqual(r['mean_omissions']['z'],'TIME_INVARIANT_DUPLICATES_ORIGINAL')
        self.assertEqual(r['mean_omissions']['x'],'MEAN_ABSORBED_OR_REDUNDANT')
        self.assertEqual(r['mundlak_test']['status'],'BLOCKED_NO_IDENTIFIABLE_ADDITIONAL_MEANS')
        self.assertIsNone(r['mundlak_test']['p_value'])
        self.assertEqual(F(r['within_coefficients_exact']['x']),within_oracle(q,['x'])[0])

    def test_multiple_independent_and_redundant_mean_directions_preserve_span(self):
        q=maximum_case();q['rows']=q['rows'][:32]
        r=api().calculate(q);self.assertEqual(len(r['included_mean_regressors']),3)
        a,_=augmented(q);o=dense_oracle(a,F(2),F(3));d=o['beta'][-3:];v=[row[-3:] for row in o['cov'][-3:]]
        expected=mm(mm([d],inverse(v)),[[x] for x in d])[0][0]
        self.assertEqual(F(r['mundlak_test']['chi_squared_statistic_exact']),expected)
        self.assertEqual(list(map(F,r['within_coefficients_exact'].values())),within_oracle(q,['a','b','c']))
        # Independent within signals with exactly the same entity means.
        q=case();q['model']['regressors'].append('w');q['units']['w']='hours'
        for g in {r['entity'] for r in q['rows']}:
            rows=[r for r in q['rows'] if r['entity']==g];m=sum(F(r['regressors']['x']) for r in rows)/3
            for t,row in enumerate(rows):row['regressors']['w']=str(m+[F(1),F(-2),F(1)][t])
        r=api().calculate(q)
        self.assertEqual(r['included_mean_regressors'],['x']);self.assertEqual(r['mean_omissions']['w'],'MEAN_ABSORBED_OR_REDUNDANT')
        self.assertFalse(r['all_time_varying_means_testable'])
        self.assertEqual([F(r['within_coefficients_exact'][v]) for v in ['x','w']],within_oracle(q,['x','w']))

    def test_collinear_within_effects_are_rejected_even_when_original_pooled_design_is_full_rank(self):
        q=case();q['model']['regressors'].append('w');q['units']['w']='hours'
        for row in q['rows']:row['regressors']['w']=str(F(row['regressors']['x'])+int(row['entity']))
        with self.assertRaisesRegex(ValueError,'WITHIN_REGRESSORS_COLLINEAR'):api().calculate(q)

    def test_robust_zero_variance_does_not_claim_reference_probability(self):
        q=case()
        for row in q['rows']:row['outcome']=str(2*F(row['regressors']['x'])+3*F(row['regressors']['z']))
        r=api().calculate(q);t=r['mundlak_test']
        self.assertEqual(t['status'],'BLOCKED_SINGULAR_RESTRICTION_COVARIANCE');self.assertIsNone(t['p_value'])
        self.assertIsNone(t['chi_squared_statistic_exact'])

    def test_generated_mean_terms_cannot_collide_with_arbitrary_original_regressor_names(self):
        q=case();old=q['model']['regressors'][1];new='ENTITY_MEAN:x';q['model']['regressors'][1]=new;q['units'][new]=q['units'].pop(old)
        for row in q['rows']:row['regressors'][new]=row['regressors'].pop(old)
        r=api().calculate(q)
        terms=r['parameter_terms'];self.assertEqual(len({canonical(t) for t in terms}),len(terms))
        self.assertIn({'kind':'ENTITY_MEAN','name':'x'},terms)
        self.assertIn({'kind':'REGRESSOR','name':new},terms)

    def test_means_use_observed_sample_only_and_row_period_order_is_irrelevant(self):
        q=case(unbalanced=True);r=api().calculate(q);q['rows'].reverse()
        for row in q['rows']:row['period']=1234-8*row['period']
        s=api().calculate(q);self.assertEqual(r['coefficients_exact'],s['coefficients_exact'])
        self.assertEqual(r['mundlak_test'],s['mundlak_test'])
        self.assertEqual(r['entity_means_exact'],s['entity_means_exact'])

    def test_response_regressor_units_and_column_order_preserve_inference(self):
        q=case();r=api().calculate(q)
        for row in q['rows']:row['outcome']=str(5*F(row['outcome']));row['regressors']['x']=str(7*F(row['regressors']['x']))
        q['variance_components'].update(idiosyncratic='50',entity='75');q['model']['regressors'].reverse();s=api().calculate(q)
        self.assertEqual(F(s['within_coefficients_exact']['x']),F(5,7)*F(r['within_coefficients_exact']['x']))
        self.assertEqual(s['mundlak_test']['chi_squared_statistic_exact'],r['mundlak_test']['chi_squared_statistic_exact'])

    def test_owned_snapshot_source_change_cancellation_and_evidence_bind_calculation(self):
        m=api();q=case();baseline=m.calculate(copy.deepcopy(q));original=m._prepare
        def mutate(*args):q['rows'][0]['outcome']='999';return original(*args)
        with patch.object(m,'_prepare',side_effect=mutate):r=m.calculate(q)
        self.assertEqual(r['input_sha256'],baseline['input_sha256']);self.assertEqual(r['coefficients_exact'],baseline['coefficients_exact'])
        self.assertEqual(r['calculation_sha256'],digest(canonical({k:r[k] for k in r['calculation_fields']})))
        with patch.object(m,'_code_receipt',side_effect=[{'v':'old'},{'v':'new'}]):
            with self.assertRaisesRegex(ValueError,'CODE_CHANGED'):m.calculate(case())
        with patch.object(m,'checkpoint',side_effect=TimeoutError('cancelled')):
            with self.assertRaises(TimeoutError):m.calculate(case())

    def test_strict_original_schema_and_augmented_parameter_limit(self):
        for change in [lambda q:q.update(extra=1),lambda q:q['rights'].update(report_export='DENIED'),lambda q:q.update(rows=tuple(q['rows']))]:
            q=case();change(q)
            with self.assertRaises(ValueError):api().calculate(q)
        q=maximum_case();q['model']['regressors'].append('d');q['units']['d']='units'
        for i,row in enumerate(q['rows']):row['regressors']['d']=str((i*i+3*i)%11)
        with self.assertRaisesRegex(ValueError,'AUGMENTED_PARAMETER_LIMIT'):api().calculate(q)

    def test_capabilities_and_scientific_unknowns_do_not_certify_exogeneity(self):
        from econscope.study_capabilities import negotiate_capabilities
        r=api().calculate(case());c=negotiate_capabilities(required=['exact_correlated_random_effects'])['random_effects_capabilities']
        self.assertIn('econscope.correlated_random_effects.calculate',c['python'])
        self.assertEqual(r['native_execution'],'NOT_RUN');self.assertEqual(r['causal_identification'],'NOT_ESTABLISHED')
        self.assertEqual(r['time_invariant_interpretation'],'BETWEEN_ASSOCIATION_ENTITY_CORRELATION_NOT_SEPARATELY_IDENTIFIED')
        self.assertEqual(r['mundlak_test']['scientific_exogeneity'],'NOT_ESTABLISHED')
        self.assertIn('conditional_mean_specification',[o['claim_id'] for o in r['obligations']])

    def test_intercept_only_and_no_intercept_zero_means_have_honest_empty_test_status(self):
        q=case();q['model']['regressors']=[];q['units']={'y':'dollars'}
        for row in q['rows']:row['regressors']={}
        r=api().calculate(q)
        self.assertEqual(r['within_slope_equivalence'],'NO_TIME_VARYING_COEFFICIENTS')
        self.assertEqual(r['mundlak_test']['status'],'BLOCKED_NO_IDENTIFIABLE_ADDITIONAL_MEANS')
        q=case();q['model'].update(include_intercept=False,regressors=['x']);q['units'].pop('z')
        for row in q['rows']:row['regressors']={'x':str(row['period']-1)}
        r=api().calculate(q)
        self.assertEqual(r['included_mean_regressors'],[])
        self.assertEqual(F(r['within_coefficients_exact']['x']),within_oracle(q,['x'])[0])

    def test_maximum_observations_entities_and_augmented_parameters(self):
        for feasible in (False,True):
            q=maximum_case(feasible);r=api().calculate(q)
            self.assertEqual(r['sample']['observations'],512);self.assertEqual(r['sample']['entities'],128)
            self.assertEqual(r['sample']['parameters'],8);self.assertEqual(r['sample']['additional_means'],3)
            self.assertEqual(r['mundlak_test']['degrees_of_freedom'],3)
            self.assertEqual([F(r['within_coefficients_exact'][v]) for v in ['a','b','c']],within_oracle(q,['a','b','c']))


if __name__=='__main__':unittest.main()
