"""Exact partial-equilibrium per-unit-tax incidence under declared linear curves.

Demand: Qd = a - b*consumer_price, supply: Qs = c + d*producer_price.
A nonnegative specific unit tax imposes consumer_price - producer_price = tax.
All quantities are strictly positive interior equilibria, prices nonnegative,
and b,d strictly positive. Results describe this hypothetical model, NOT an
identified empirical policy effect, a general equilibrium, or a tax-law rule.
"""
from fractions import Fraction as F
import re

from .execution_control import checkpoint

_NUMBER = re.compile(r'-?(?:0|[1-9][0-9]{0,11})(?:\.[0-9]{1,9})?\Z', re.ASCII)
_UNIT = re.compile(r'[A-Za-z0-9][A-Za-z0-9_ ./%-]{0,39}\Z', re.ASCII)
_DIGEST = re.compile(r'[0-9a-f]{64}\Z', re.ASCII)


def _fail(reason):
    raise ValueError('LINEAR_TAX_INCIDENCE_'+reason)


def _number(value):
    if type(value) is not str or not _NUMBER.fullmatch(value):
        _fail('DECIMAL_STRING_REQUIRED')
    return F(value)


def _validate(spec):
    fields={'schema','data_kind','source_declared_sha256','rights','quantity_unit',
            'price_unit','demand','supply','unit_tax'}
    if type(spec) is not dict or set(spec)!=fields or spec['schema']!='societybox-linear-tax-incidence-v1':
        _fail('SCHEMA_INVALID')
    if type(spec['data_kind']) is not str or spec['data_kind'] not in ('SYNTHETIC','RESEARCHER_SUPPLIED'):
        _fail('DATA_KIND_INVALID')
    if type(spec['source_declared_sha256']) is not str or not _DIGEST.fullmatch(spec['source_declared_sha256']):
        _fail('SOURCE_DIGEST_REQUIRED')
    if spec['rights']!={'local_processing':'PERMITTED','report_export':'PERMITTED'}:
        _fail('RIGHTS_REQUIRED')
    for field in ('quantity_unit','price_unit'):
        if type(spec[field]) is not str or not _UNIT.fullmatch(spec[field]):
            _fail('UNIT_INVALID')
    for field in ('demand','supply'):
        part=spec[field]
        if type(part) is not dict or set(part)!={'quantity_intercept','price_slope'}:
            _fail('CURVE_INVALID')
    a=_number(spec['demand']['quantity_intercept'])
    b=_number(spec['demand']['price_slope'])
    c=_number(spec['supply']['quantity_intercept'])
    d=_number(spec['supply']['price_slope'])
    tau=_number(spec['unit_tax'])
    if a<=0 or b<=0 or d<=0:
        _fail('POSITIVE_DEMAND_INTERCEPT_AND_SLOPES_REQUIRED')
    if tau<0:
        _fail('NONNEGATIVE_UNIT_TAX_REQUIRED')
    return a,b,c,d,tau


def linear_tax_incidence(spec):
    """Compute exact interior before/after equilibrium and welfare identities.

    The partial-equilibrium surplus integrals assume the curves are marginal
    willingness-to-pay and marginal cost (with no other market distortions).
    Rejects shutdown/corner solutions, where interior price formulas are
    inappropriate. Tax revenue is quantity times tax, not a fiscal forecast.
    """
    a,b,c,d,tau=_validate(spec)
    checkpoint()
    denominator=b+d
    p0=(a-c)/denominator
    q0=a-b*p0
    pp=(a-c-b*tau)/denominator
    pc=pp+tau
    q=a-b*pc
    if min(p0,q0)<=0:
        _fail('BASELINE_INTERIOR_EQUILIBRIUM_REQUIRED')
    if pp<0 or pc<0 or q<=0:
        _fail('AFTER_TAX_INTERIOR_EQUILIBRIUM_REQUIRED')
    if q0 != c+d*p0 or q != c+d*pp or pc-pp!=tau:
        _fail('MARKET_CLEARING_IDENTITY_FAILED')

    cs0=q0*q0/(2*b)
    ps0=q0*q0/(2*d)
    cs=q*q/(2*b)
    ps=q*q/(2*d)
    revenue=tau*q
    dwl=tau*(q0-q)/2
    consumer_burden=pc-p0
    producer_burden=p0-pp
    if (consumer_burden+producer_burden!=tau or
            cs0+ps0 != cs+ps+revenue+dwl or
            dwl<0 or revenue<0 or consumer_burden<0 or producer_burden<0):
        _fail('WELFARE_ACCOUNTING_IDENTITY_FAILED')
    return dict(schema='societybox-linear-tax-incidence-result-v1',
                status='CONDITIONAL_MODEL_COUNTERFACTUAL_ONLY',
                data_kind=spec['data_kind'],source_declared_sha256=spec['source_declared_sha256'],
                source_authentication='NOT_ESTABLISHED',
                quantity_unit=spec['quantity_unit'],price_unit=spec['price_unit'],
                slope_contract='QD_A_MINUS_B_PC_QS_C_PLUS_D_PP_BOTH_SLOPES_STRICTLY_POSITIVE',
                market_scope='SINGLE_COMPETITIVE_PARTIAL_EQUILIBRIUM_INTERIOR_ONLY',
                elasticities_and_behavior='DECLARED_LINEAR_CURVES_NOT_EMPIRICALLY_VALIDATED',
                causal_policy_effect='NOT_IDENTIFIED',fiscal_forecast='NOT_ESTABLISHED',
                assumptions='NO_SPILLOVERS_EXTERNALITIES_OR_OTHER_DISTORTIONS_NOT_VERIFIED',
                original_upstream_parity='NOT_RUN',
                unit_tax_exact=str(tau),
                baseline=dict(price_exact=str(p0),quantity_exact=str(q0),
                              consumer_surplus_exact=str(cs0),producer_surplus_exact=str(ps0)),
                after_tax=dict(consumer_price_exact=str(pc),producer_price_exact=str(pp),
                               quantity_exact=str(q),consumer_surplus_exact=str(cs),
                               producer_surplus_exact=str(ps)),
                incidence=dict(consumer_burden_per_unit_exact=str(consumer_burden),
                               producer_burden_per_unit_exact=str(producer_burden),
                               consumer_price_pass_through_exact=str(d/denominator),
                               quantity_change_exact=str(q-q0)),
                welfare=dict(tax_revenue_exact=str(revenue),deadweight_loss_exact=str(dwl),
                             total_before_exact=str(cs0+ps0),
                             total_after_including_revenue_exact=str(cs+ps+revenue),
                             accounting_identity='EXACT_CONSUMER_SURPLUS_PLUS_PRODUCER_SURPLUS_PLUS_REVENUE_PLUS_DWL'))
