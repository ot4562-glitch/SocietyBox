"""Mundlak mean-augmented static GLS and identifiable mean-direction Wald.

Reuses the exact random-intercept engine. Means use only observed sample rows.
The entire span of time-varying regressor means is added modulo the original
design; rank-redundant means are recorded, not treated as evidence of no
correlation. Within-collinear original slopes cannot be separately identified.
This linear conditional-mean device does not establish exogeneity, selection
ignorability, a time-invariant causal coefficient or native-package parity.
"""
from fractions import Fraction as F
from pathlib import Path

from . import random_effects as re, instrumental_variables as iv
from .fixed_effects import _orthogonal
from .mechanism_dynamic import _dot
from .execution_control import checkpoint
from .workspace import canonical,digest
from .research_evidence import (make_evidence,EvidenceLedger,UnknownRegistry,
                                make_obligation,evaluate_coverage)


def _fail(reason):raise ValueError('CORRELATED_RANDOM_EFFECTS_'+reason)


def _prepare(q,x,y,groups):
    labels=q['model']['regressors'];offset=int(q['model']['include_intercept']);n=len(y);k=len(x[0])
    means={};columns=[[] for _ in labels];within=[[] for _ in labels]
    indexed={}
    for g,positions in groups.items():
        checkpoint();values=[sum(x[i][j+offset] for i in positions)/len(positions) for j in range(len(labels))]
        means[g]={v:re._exact(a) for v,a in zip(labels,values)}
        for i in positions:indexed[i]=values
    for i,row in enumerate(x):
        for j in range(len(labels)):
            columns[j].append(indexed[i][j]);within[j].append(row[j+offset]-indexed[i][j])
    varying=[j for j,col in enumerate(within) if any(col)];weights=[F(1)]*n
    if len(_orthogonal([within[j] for j in varying],weights)[0])!=len(varying):_fail('WITHIN_REGRESSORS_COLLINEAR')
    basis,norms=_orthogonal(list(map(list,zip(*x))),weights)
    if len(basis)!=k:_fail('ORIGINAL_DESIGN_RANK_DEFICIENT')
    selected=[];omitted={};added=[]
    for j,label in enumerate(labels):
        checkpoint()
        if j not in varying:omitted[label]='TIME_INVARIANT_DUPLICATES_ORIGINAL';continue
        v=list(columns[j])
        for b,norm in zip(basis,norms):
            c=_dot(v,b)/norm;v=[a-c*d for a,d in zip(v,b)]
        norm=_dot(v,v)
        if not norm:omitted[label]='MEAN_ABSORBED_OR_REDUNDANT';continue
        if k+len(selected)>=8 or k+len(selected)+1>=n:_fail('AUGMENTED_PARAMETER_LIMIT')
        selected.append(j);added.append(columns[j]);basis.append(v);norms.append(norm)
    augmented=[row+[column[i] for column in added] for i,row in enumerate(x)]
    return dict(x=augmented,means=means,selected=selected,varying=varying,omitted=omitted,
                within=[within[j] for j in varying],base_parameters=k)


def validate(q):
    """Own and validate the existing RE input plus exact augmentation ranks."""
    owned=iv._snapshot(q,limit=262144);x,y,g=re._validated(owned);_prepare(owned,x,y,g);return owned


def _wald(result,k,r):
    test=dict(method='MUNDLAK_IDENTIFIABLE_MEAN_WALD',degrees_of_freedom=r,
        scope='IDENTIFIABLE_ADDITIONAL_ENTITY_MEAN_DIRECTIONS',scientific_exogeneity='NOT_ESTABLISHED',
        reference_distribution='CONDITIONAL_ASYMPTOTIC_CHI_SQUARED',chi_squared_statistic_exact=None,
        p_value=None,log_p_value=None)
    if not r:return dict(test,status='BLOCKED_NO_IDENTIFIABLE_ADDITIONAL_MEANS')
    beta=list(map(F,result['coefficients_exact'][k:]));cov=[[F(v) for v in row[k:]] for row in result['covariance_exact'][k:]]
    test.update(restriction_residuals_exact=list(map(re._exact,beta)),restriction_covariance_exact=[list(map(re._exact,row)) for row in cov])
    try:precision=iv._inverse_small(cov,failure='MUNDLAK_RESTRICTION_COVARIANCE_SINGULAR')
    except ValueError as error:
        if str(error)!='IV_MUNDLAK_RESTRICTION_COVARIANCE_SINGULAR':raise
        return dict(test,status='BLOCKED_SINGULAR_RESTRICTION_COVARIANCE')
    stat=_dot(beta,[_dot(row,beta) for row in precision])
    if stat<0:_fail('WALD_STATISTIC_NEGATIVE')
    test.update(chi_squared_statistic_exact=re._exact(stat),**iv._chi_probability(stat,r))
    test['status']='COMPUTED_ASSUMPTIONS_UNVERIFIED' if test['p_value'] is not None else 'STATISTIC_COMPUTED_REFERENCE_PROBABILITY_BLOCKED'
    if test['p_value'] is not None:test['distribution_arithmetic']='BINARY64_NOT_INTERVAL_CERTIFIED'
    return test


def _code_receipt():
    code=re._code_receipt();code[Path(__file__).name]=digest(Path(__file__).read_bytes());return code


def _assurance(q,result,code):
    subject=digest(canonical(q));scope=dict(model=q['model'],units=q['units'],period_unit=q['period_unit'],
        variance_components=q['variance_components'],covariance=q['covariance'],sample=result['sample'],
        mean_construction='OBSERVED_SAMPLE_MEANS_ALL_TIME_VARYING_SPAN_MODULO_ORIGINAL_DESIGN')
    fields=sorted(result);calculation_sha=digest(canonical(result))
    claims={
        'conditional_mean_specification':'Is the conditional entity-effect projection adequately represented by these observed regressor means?',
        'idiosyncratic_strict_exogeneity':'Are idiosyncratic errors orthogonal to the full regressor history conditional on heterogeneity?',
        'time_invariant_separation':'Can time-invariant regressor effects be separated from correlated entity heterogeneity using additional assumptions?',
        'observation_selection':'Is the observed unbalanced estimation sample selection compatible with the declared conditional-mean model?',
        'independent_entities':'Are entity clusters independent and sufficiently numerous for Wald asymptotics?',
        'finite_sample_reference':'Are covariance/variance-component plug-in effects and finite-sample coverage independently assessed?',
        'source_vintage_replication':'Are source, rights, historical availability and original API replication independently verified?'}
    evidence=make_evidence(subject_sha256=subject,producer=dict(tool_id='econ.correlated_random_effects',version='1',code_sha256=digest(canonical(code))),
        operation=result['method'],authority='LOCAL_EXACT_AND_BINARY64_REFERENCE_DISTRIBUTION'
            if result['mundlak_test'].get('distribution_arithmetic') else 'LOCAL_EXACT_ARITHMETIC',
        epistemic_state='DERIVED',proof_kind='DECLARED_MUNDLAK_GLS_AND_IDENTIFIABLE_MEAN_WALD',scope=scope,
        coverage=dict(status='PARTIAL',unknown_dimensions=list(claims)),
        result=dict(calculation_sha256=calculation_sha,calculation_fields=fields,status=result['status'],
                    within_slope_equivalence=result['within_slope_equivalence'],mundlak_status=result['mundlak_test']['status']),
        parent_evidence_ids=[],limitations=[
            'Mean augmentation imposes a linear conditional-mean representation; it does not eliminate arbitrary endogeneity.',
            'Nonrejection is not exogeneity proof; absorbed/redundant means leave untestable directions.',
            'Time-invariant coefficients retain unseparated between-entity heterogeneity interpretation.',
            'Observed-sample FE numerical equivalence is not missingness ignorability, scientific validity or native parity.'])
    ledger=EvidenceLedger([evidence]);registry=UnknownRegistry(ledger);obligations=[]
    for claim,question in claims.items():
        u=registry.register(question,scope,'EXTERNAL_SCIENTIFIC_VALIDATION','CORRELATED_EFFECTS_ASSUMPTION_CHECK')
        obligations.append(make_obligation(claim_id=claim,subject_sha256=subject,scope=scope,
            required_authority='EXTERNAL_SCIENTIFIC_VALIDATION',required_proof_kind='CORRELATED_EFFECTS_ASSUMPTION_CHECK',
            verifier_id='econ.correlated_random_effects',dependencies=[],unknown_ids=[u['unknown_id']],blocking_reasons=[],next_probe=question))
    return dict(input_sha256=subject,code_sha256=code,calculation_sha256=calculation_sha,calculation_fields=fields,
                evidence=ledger.records(),unknowns=registry.records(),obligations=obligations,
                coverage=evaluate_coverage(obligations,ledger,registry))


def calculate(q):
    """Mundlak GLS plus joint Wald of identifiable additional entity means.

    Uses the existing societybox-random-effects-v1 input/variance/covariance
    contract. Means use the observed estimation sample; include all linearly
    independent mean directions modulo the original design, recording others.
    All time-varying within slopes must be identified. Augmented parameters<=8,
    rows<=512, entities<=128, exact serialized fractions<=12000bits per part.
    No finite-sample F, causal certification or general Hausman claim.
    """
    checkpoint();owned=iv._snapshot(q,limit=262144);x,y,groups=re._validated(owned);code=_code_receipt()
    p=_prepare(owned,x,y,groups);labels=owned['model']['regressors'];var=owned['variance_components']
    if var['method']=='DECLARED':
        se,su=re._number(var['idiosyncratic']),re._number(var['entity'])
        components=dict(method='DECLARED',idiosyncratic_exact=re._exact(se),entity_exact=re._exact(su),
            boundary='ENTITY_VARIANCE_ZERO' if su==0 else 'INTERIOR',variance_authentication='NOT_ESTABLISHED')
    else:
        # Augmentation changes neither within nor between column space. Use the
        # original bounded design for the same moment estimate, without doubling
        # its projection columns. This is not a refit with different df.
        se,su,components=re._moments(x,y,groups)
    result=re._fit(p['x'],y,groups,se,su,owned['covariance']);k=p['base_parameters'];r=len(p['selected']);n=len(y)
    residual=list(map(F,result['marginal_residuals_exact']))
    normal=[_dot(column,residual) for column in p['within']]
    if any(normal):_fail('WITHIN_SLOPE_EQUIVALENCE_FAILED')
    terms=([dict(kind='INTERCEPT')] if owned['model']['include_intercept'] else [])+[
        dict(kind='REGRESSOR',name=v) for v in labels]+[dict(kind='ENTITY_MEAN',name=labels[j]) for j in p['selected']]
    result.update(schema='societybox-correlated-random-effects-result-v1',method='MUNDLAK_CORRELATED_RANDOM_INTERCEPT_GLS',
        implementation_status='INDEPENDENT_IMPLEMENTED',status='COMPUTED_ASSUMPTIONS_UNVERIFIED',
        arithmetic='EXACT_RATIONAL_GLS_AND_WALD',variance_components=components,parameter_terms=terms,
        included_mean_regressors=[labels[j] for j in p['selected']],mean_omissions=p['omitted'],
        all_time_varying_means_testable=r==len(p['varying']),entity_means_exact=p['means'],
        within_coefficients_exact={labels[j]:result['coefficients_exact'][j+int(owned['model']['include_intercept'])] for j in p['varying']},
        within_slope_equivalence='VERIFIED_EXACT_ON_OBSERVED_SAMPLE' if p['varying'] else 'NO_TIME_VARYING_COEFFICIENTS',
        within_normal_residuals_exact=list(map(re._exact,normal)),
        time_invariant_interpretation='BETWEEN_ASSOCIATION_ENTITY_CORRELATION_NOT_SEPARATELY_IDENTIFIED',
        sample=dict(observations=n,entities=len(groups),parameters=k+r,original_parameters=k,additional_means=r,residual_df=n-k-r,
                    entity_observations={g:len(v) for g,v in groups.items()}),entity_order=list(groups),
        covariance_estimator=owned['covariance'],native_execution='NOT_RUN',native_parity='NOT_RUN',
        causal_identification='NOT_ESTABLISHED',source_authentication='NOT_ESTABLISHED',historical_as_of='NOT_ESTABLISHED',
        conditional_mean_specification='DECLARED_NOT_VERIFIED')
    result['mundlak_test']=_wald(result,k,r)
    result.update(_assurance(owned,result,code))
    if code!=_code_receipt():_fail('CODE_CHANGED')
    return result
