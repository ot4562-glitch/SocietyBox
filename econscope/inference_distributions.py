"""Independent central F / chi-square upper tails and F critical values.

Public incomplete-beta/gamma identities, modified-Lentz continued fractions,
and a gamma series. Binary64 arithmetic, integer df in [1,10000]. Log tails
remain available when the returned probability underflows. These functions
evaluate a declared reference distribution; they never validate a study design.
F log tails accept exact rational statistics outside binary64 value range.
Chi-square log tails explicitly reject statistics whose logarithmic probability
cannot be evaluated in this binary64 implementation; callers retain exact fits.
"""
from fractions import Fraction
import math
import sys

from .execution_control import checkpoint

_EPS = 8e-15
_TINY = 1e-300
_ITERATIONS = 2048


def _degrees(value):
    if type(value) is not int or not 1 <= value <= 10000:
        raise ValueError('INFERENCE_DEGREES_OF_FREEDOM_INVALID')
    return value


def _statistic(value):
    if type(value) not in (int,float,Fraction):
        raise ValueError('INFERENCE_STATISTIC_INVALID')
    try: result = float(value)
    except (OverflowError,ValueError) as error:
        raise ValueError('INFERENCE_STATISTIC_FLOAT64_RANGE') from error
    if math.isnan(result) or result < 0:
        raise ValueError('INFERENCE_STATISTIC_INVALID')
    return result


def _log_statistic(value):
    if type(value) not in (int,float,Fraction):
        raise ValueError('INFERENCE_STATISTIC_INVALID')
    if type(value) is float:
        number = _statistic(value)
        return -math.inf if number == 0 else math.log(number)
    if value < 0: raise ValueError('INFERENCE_STATISTIC_INVALID')
    if value == 0: return -math.inf
    try: number = float(value)
    except OverflowError: number = math.inf
    if sys.float_info.min <= number < math.inf: return math.log(number)
    # Exact sufficient statistics can exceed or fall below binary64 range even
    # when all model inputs obey their individual numeric limits.
    return math.log(value.numerator)-math.log(value.denominator)


def _nonzero(value):
    return math.copysign(_TINY,value) if abs(value) < _TINY else value


def _log_complement(log_probability):
    # Keep log(1-p) accurate for both tiny p and p extremely close to one.
    return (math.log1p(-math.exp(log_probability)) if log_probability < -math.log(2)
            else math.log(-math.expm1(log_probability)))


def _beta_fraction(a,b,x):
    """Alternating even/odd beta continued-fraction convergents."""
    c = 1.0; d = 1/_nonzero(1-(a+b)*x/(a+1)); h = d
    for m in range(1,_ITERATIONS+1):
        if m%64 == 0: checkpoint()
        even = m*(b-m)*x/((a+2*m-1)*(a+2*m))
        d = 1/_nonzero(1+even*d); c = _nonzero(1+even/c); h *= d*c
        odd = -(a+m)*(a+b+m)*x/((a+2*m)*(a+2*m+1))
        d = 1/_nonzero(1+odd*d); c = _nonzero(1+odd/c); delta = d*c; h *= delta
        if abs(delta-1) <= _EPS:
            if not math.isfinite(h) or h <= 0: break
            return h
    raise ValueError('INFERENCE_BETA_CONVERGENCE_FAILED')


def _log_beta_lower(a,b,x,complement,log_x,log_complement):
    # Keep the logs formed before exponentiation: beta arguments can themselves
    # underflow while their square-root probabilities remain representable.
    log_front = math.lgamma(a+b)-math.lgamma(a)-math.lgamma(b)+a*log_x+b*log_complement
    direct = x < (a+1)/(a+b+2)
    if direct:
        value = log_front+math.log(_beta_fraction(a,b,x))-math.log(a)
    else:
        other = log_front+math.log(_beta_fraction(b,a,complement))-math.log(b)
        if other >= 0: raise ValueError('INFERENCE_BETA_PROBABILITY_INVALID')
        value = _log_complement(other)
    if value > 0 or math.isnan(value): raise ValueError('INFERENCE_BETA_PROBABILITY_INVALID')
    return value


def log_f_survival(statistic, numerator_df, denominator_df):
    """Log P(F >= statistic); central F and integer finite positive df."""
    log_x = _log_statistic(statistic); d1,d2 = _degrees(numerator_df),_degrees(denominator_df)
    if log_x == -math.inf: return 0.0
    if log_x == math.inf: return -math.inf
    log_ratio = log_x+math.log(d1)-math.log(d2)
    if log_ratio >= 0:
        denominator = math.log1p(math.exp(-log_ratio))
        log_z,log_complement = -log_ratio-denominator,-denominator
    else:
        denominator = math.log1p(math.exp(log_ratio))
        log_z,log_complement = -denominator,log_ratio-denominator
    return _log_beta_lower(d2/2,d1/2,math.exp(log_z),math.exp(log_complement),log_z,log_complement)


def f_survival(statistic, numerator_df, denominator_df):
    return math.exp(log_f_survival(statistic,numerator_df,denominator_df))


def log_chi_square_survival(statistic, degrees_of_freedom):
    """Log Q(df/2, statistic/2), with a gamma series or upper-tail fraction."""
    log_statistic = _log_statistic(statistic); a = _degrees(degrees_of_freedom)/2
    if log_statistic == -math.inf: return 0.0
    if log_statistic == math.inf: return -math.inf
    statistic = _statistic(statistic)
    x = statistic/2
    front = -x+a*(log_statistic-math.log(2))-math.lgamma(a)
    if x < a+1:
        term = total = 1/a
        for n in range(1,_ITERATIONS+1):
            if n%64 == 0: checkpoint()
            term *= x/(a+n); total += term
            if abs(term) <= abs(total)*_EPS:
                log_lower = front+math.log(total)
                if log_lower >= 0: break
                return _log_complement(log_lower)
    else:
        b = x+1-a; c = 1/_TINY; d = 1/_nonzero(b); h = d
        for n in range(1,_ITERATIONS+1):
            if n%64 == 0: checkpoint()
            coefficient = -n*(n-a); b += 2
            d = 1/_nonzero(b+coefficient*d); c = _nonzero(b+coefficient/c)
            delta = c*d; h *= delta
            if abs(delta-1) <= _EPS:
                if h <= 0 or not math.isfinite(h): break
                result = front+math.log(h)
                if result > 0 or math.isnan(result): break
                return result
    raise ValueError('INFERENCE_GAMMA_CONVERGENCE_FAILED')


def chi_square_survival(statistic, degrees_of_freedom):
    return math.exp(log_chi_square_survival(statistic,degrees_of_freedom))


def f_critical_value(upper_tail_probability, numerator_df, denominator_df):
    """Central F inverse upper tail, with numerical bracket and tail residual.

    Alpha in [1e-12,1-1e-12]; finite per-call numeric limits, not a product
    coverage ceiling. The bracket is numerical evidence, not interval arithmetic.
    """
    if type(upper_tail_probability) not in (int,float,Fraction):
        raise ValueError('INFERENCE_TAIL_PROBABILITY_INVALID')
    try: alpha = float(upper_tail_probability)
    except (OverflowError,ValueError) as error:
        raise ValueError('INFERENCE_TAIL_PROBABILITY_INVALID') from error
    if not math.isfinite(alpha) or not 1e-12 <= alpha <= 1-1e-12:
        raise ValueError('INFERENCE_TAIL_PROBABILITY_INVALID')
    d1,d2 = _degrees(numerator_df),_degrees(denominator_df)
    target = math.log(alpha); low,high = -700.0,700.0
    if not log_f_survival(math.exp(low),d1,d2) >= target >= log_f_survival(math.exp(high),d1,d2):
        raise ValueError('INFERENCE_QUANTILE_BRACKET_FAILED')
    for iteration in range(1,201):
        if iteration%32 == 0: checkpoint()
        middle = (low+high)/2
        if log_f_survival(math.exp(middle),d1,d2) > target: low = middle
        else: high = middle
        if high-low <= 2e-13: break
    else: raise ValueError('INFERENCE_QUANTILE_CONVERGENCE_FAILED')
    value = math.exp((low+high)/2); tail = f_survival(value,d1,d2)
    relative_error = abs(tail-alpha)/alpha
    if relative_error > 5e-10: raise ValueError('INFERENCE_QUANTILE_TAIL_RESIDUAL_FAILED')
    return dict(critical_value=value, lower=math.exp(low), upper=math.exp(high),
                requested_upper_tail=alpha, attained_upper_tail=tail,
                tail_absolute_error=abs(tail-alpha), tail_relative_error=relative_error,
                numerator_df=d1,denominator_df=d2,iterations=iteration,
                arithmetic='BINARY64_NOT_INTERVAL_CERTIFIED')
