"""Bounded sharp regression discontinuity with fixed-bandwidth local linear WLS.

Fits independent linear trends to observations on either side of a declared
cutoff; the jump in intercepts is a conditional descriptive estimand. HC0 is
computed from weighted score influences, not a confidence or causal-validity
certificate. Does not choose a bandwidth, bias-correct, or fit fuzzy RDD.
"""
import math

from .execution_control import checkpoint
from .numeric_linear import least_squares


def _fail(reason):
    raise ValueError('RDD_' + reason)


def _number(value):
    if type(value) not in (int, float):
        _fail('FINITE_NUMBER_REQUIRED')
    try:
        numeric = float(value)
    except OverflowError as error:
        raise ValueError('RDD_NUMBER_RANGE') from error
    if not math.isfinite(numeric):
        _fail('FINITE_NUMBER_REQUIRED')
    return numeric


def _fit_side(rows):
    """Return local intercept and HC0 variance, in the original outcome units."""
    design = [[1.0, row[0]] for row in rows]
    outcomes = [row[1] for row in rows]
    weights = [row[2] for row in rows]
    fitted = least_squares(design, outcomes, weights=weights)
    bread = fitted['inverse_information']
    # For V_HC0 = B (sum score_i score_i') B, only the intercept variance
    # is needed. This computes e_0' B score_i independently for each row.
    influence = []
    for (scaled_running, _y, weight), error in zip(rows, fitted['residuals']):
        checkpoint()
        direction = math.fsum((bread[0][0], bread[0][1] * scaled_running))
        influence.append(direction * weight * error)
    try:
        variance = math.fsum(v * v for v in influence)
    except (OverflowError, ValueError) as error:
        raise ValueError('RDD_COVARIANCE_RANGE') from error
    if not math.isfinite(variance) or variance < 0:
        _fail('COVARIANCE_RANGE')
    return dict(intercept=fitted['coefficients'][0],
                slope_scaled=fitted['coefficients'][1],
                intercept_variance_hc0=variance,
                observations=len(rows), weighted_sse=fitted['weighted_sse'])


def sharp_local_linear(running, outcomes, *, cutoff=0.0, bandwidth, kernel='triangular'):
    """Return a local-linear jump estimate and conditional HC0 variance.

    Observations with abs((running-cutoff)/bandwidth) >= 1 are excluded.
    At the cutoff, assignment follows the *right/treated* side. At least
    three locally supported observations with full-rank designs are required
    on each side. Positive analytic kernel weights are not survey weights.

    The returned value is NOT a verified causal treatment effect: continuity,
    manipulation, interference, local design, effective sample size and
    uncorrected local-linear bias all require separate evidence.
    """
    if (type(running) is not list or type(outcomes) is not list or
            not 6 <= len(running) <= 1024 or len(outcomes) != len(running)):
        _fail('DIMENSIONS_INVALID')
    center, width = _number(cutoff), _number(bandwidth)
    if width <= 0:
        _fail('POSITIVE_BANDWIDTH_REQUIRED')
    if kernel not in ('uniform', 'triangular') or type(kernel) is not str:
        _fail('KERNEL_UNSUPPORTED')
    left, right = [], []
    for idx, (x, y) in enumerate(zip(running, outcomes)):
        if idx % 32 == 0:
            checkpoint()
        x, y = _number(x), _number(y)
        scaled = (x - center) / width
        if not math.isfinite(scaled):
            _fail('RUNNING_DISTANCE_RANGE')
        if abs(scaled) >= 1:
            continue
        weight = 1.0 if kernel == 'uniform' else 1.0 - abs(scaled)
        if weight <= 0 or not math.isfinite(weight):
            _fail('KERNEL_WEIGHT_INVALID')
        (right if x >= center else left).append((scaled, y, weight))
    if len(left) < 3 or len(right) < 3:
        _fail('THREE_SUPPORTED_OBSERVATIONS_PER_SIDE_REQUIRED')
    lfit, rfit = _fit_side(left), _fit_side(right)
    estimate = rfit['intercept'] - lfit['intercept']
    variance = math.fsum((lfit['intercept_variance_hc0'],
                          rfit['intercept_variance_hc0']))
    if not math.isfinite(estimate) or not math.isfinite(variance):
        _fail('RESULT_RANGE')
    return dict(status='COMPUTED_ASSUMPTIONS_UNVERIFIED',
                method='SHARP_RDD_FIXED_BANDWIDTH_LOCAL_LINEAR',
                cutoff=center, bandwidth=width, kernel=kernel,
                jump=estimate, variance_hc0=variance,
                standard_error_hc0=math.sqrt(variance),
                left=lfit, right=rfit,
                confidence_interval='NOT_CALCULATED_BIAS_AND_DESIGN_UNVERIFIED',
                p_value='NOT_CALCULATED',
                causal_identification='NOT_ESTABLISHED',
                bandwidth_selection='RESEARCHER_DECLARED_NOT_OPTIMIZED',
                assignment='RIGHT_SIDE_INCLUDES_CUTOFF',
                covariance_assumptions='INDEPENDENT_SCORES_AND_LOCAL_LINEAR_APPROXIMATION_NOT_VERIFIED',
                numeric_method='BINARY64_PIVOTED_QR_ANALYTIC_KERNEL_WLS_HC0')
