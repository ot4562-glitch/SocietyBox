"""Bounded, independently testable Stone-Geary consumer-demand duality.

Utility U(x) = product_i (x_i - gamma_i) ** alpha_i with sum alpha=1,
x_i >= gamma_i and p_i>0. These are declared preferences, not estimated
preferences, empirical demand, identified welfare or policy consequences.
"""
import math


def _fail(code):
    raise ValueError('CONSUMER_DEMAND_'+code)


def _num(v):
    if type(v) not in (int, float):
        _fail('FINITE_NUMBER_REQUIRED')
    try:
        x=float(v)
    except OverflowError as e:
        raise ValueError('CONSUMER_DEMAND_FLOAT_RANGE') from e
    if not math.isfinite(x):
        _fail('FINITE_NUMBER_REQUIRED')
    return x


def _parameters(prices, shares, subsistence):
    if (type(prices) is not list or type(shares) is not list or
            type(subsistence) is not list or not 2 <= len(prices) <= 32 or
            len(shares) != len(prices) or len(subsistence) != len(prices)):
        _fail('DIMENSIONS_INVALID')
    p=list(map(_num,prices)); a=list(map(_num,shares)); gamma=list(map(_num,subsistence))
    if any(v<=0 for v in p): _fail('POSITIVE_PRICES_REQUIRED')
    if any(v<=0 for v in a): _fail('POSITIVE_SHARES_REQUIRED')
    if any(v<0 for v in gamma): _fail('NONNEGATIVE_SUBSISTENCE_REQUIRED')
    total=math.fsum(a)
    if not math.isfinite(total) or abs(total-1)>1e-12:
        _fail('SHARES_SUM_ONE_REQUIRED')
    a=[v/total for v in a]
    floor=math.fsum(x*g for x,g in zip(p,gamma))
    if not math.isfinite(floor): _fail('SUBSISTENCE_BUDGET_RANGE')
    log_index=math.fsum(w*(math.log(price)-math.log(w)) for price,w in zip(p,a))
    if not math.isfinite(log_index): _fail('PRICE_INDEX_RANGE')
    try: index=math.exp(log_index)
    except OverflowError as e: raise ValueError('CONSUMER_DEMAND_PRICE_INDEX_RANGE') from e
    if index<=0 or not math.isfinite(index): _fail('PRICE_INDEX_RANGE')
    return p,a,gamma,floor,index


def _allocation(prices, shares, subsistence, excess):
    demand=[g+w*excess/p for p,w,g in zip(prices,shares,subsistence)]
    if any(not math.isfinite(x) for x in demand): _fail('DEMAND_RANGE')
    return demand


def marshallian_demand(prices, shares, subsistence, income):
    """Stone-Geary demands and the analytically derived Slutsky matrix.

    No estimates, standard errors or causal validity are computed. The
    subsistence quantities are a declared model parameter, not an externally
    established minimum-consumption level.
    """
    p,a,g,floor,index=_parameters(prices,shares,subsistence)
    m=_num(income)
    if m<floor: _fail('INCOME_BELOW_SUBSISTENCE')
    excess=m-floor
    demand=_allocation(p,a,g,excess)
    spent=math.fsum(price*quantity for price,quantity in zip(p,demand))
    if not math.isfinite(spent) or not math.isclose(spent,m,rel_tol=2e-12,abs_tol=2e-12):
        _fail('BUDGET_IDENTITY_FAILED')
    utility=excess/index
    if not math.isfinite(utility): _fail('UTILITY_RANGE')
    # Slutsky compensation matrix for Stone-Geary (symmetric, negative
    # semidefinite): D(a_i*a_j/(p_i*p_j)-I_ij*a_i/(p_i*p_i)).
    slutsky=[[excess*(a[i]*a[j]/(p[i]*p[j])-
                      (a[i]/(p[i]*p[i]) if i==j else 0.0))
              for j in range(len(p))] for i in range(len(p))]
    if any(not math.isfinite(v) for row in slutsky for v in row):
        _fail('SLUTSKY_RANGE')
    return dict(status='COMPUTED_DECLARED_PREFERENCES_ONLY',
                method='STONE_GEARY_MARSHALLIAN',
                quantities=demand,utility=utility,price_index=index,
                subsistence_expenditure=floor,supernumerary_income=excess,
                spending=spent,budget_residual=m-spent,
                slutsky_matrix=slutsky,
                preference_identification='NOT_ESTABLISHED',
                empirical_validation='NOT_RUN',
                welfare_interpretation='MODEL_CONDITIONAL_NOT_OBSERVED')


def hicksian_demand(prices, shares, subsistence, utility):
    """Solve minimum expenditure for target Stone-Geary utility U>=0.

    Dual: e(p,U) = sum(p*gamma) + U * product((p_i/a_i)**a_i).
    """
    p,a,g,floor,index=_parameters(prices,shares,subsistence)
    u=_num(utility)
    if u<0: _fail('NONNEGATIVE_UTILITY_REQUIRED')
    excess=u*index
    expense=floor+excess
    if not math.isfinite(excess) or not math.isfinite(expense):
        _fail('EXPENDITURE_RANGE')
    demand=_allocation(p,a,g,excess)
    spent=math.fsum(price*quantity for price,quantity in zip(p,demand))
    if not math.isfinite(spent) or not math.isclose(spent,expense,rel_tol=2e-12,abs_tol=2e-12):
        _fail('EXPENDITURE_IDENTITY_FAILED')
    return dict(status='COMPUTED_DECLARED_PREFERENCES_ONLY',
                method='STONE_GEARY_HICKSIAN',
                quantities=demand,expenditure=expense,
                price_index=index,subsistence_expenditure=floor,
                target_utility=u,spending=spent,
                preference_identification='NOT_ESTABLISHED',
                empirical_validation='NOT_RUN')
