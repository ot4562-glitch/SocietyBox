"""Two/three-state exact acceptance checker, NOT a native execution receipt.

For P=[[1-a,a],[b,1-b]], balance gives pi0*a=pi1*b and
normalization gives pi=(b/(a+b),a/(a+b)). Decimal inputs are converted
to exact rationals independently of QuantEcon's GTH implementation.
"""
from fractions import Fraction
import math


TOLERANCE = 1e-12


def _number(value):
    if type(value) not in (int, float):
        raise ValueError('FINITE_NUMBER_REQUIRED')
    # math.isfinite(int) converts to float: huge integers raise OverflowError.
    # Avoid that implicit conversion on untrusted JSON-shaped input.
    if type(value) is int:
        if abs(value) > 10**18:
            raise ValueError('FINITE_NUMBER_REQUIRED')
    elif not math.isfinite(value):
        raise ValueError('FINITE_NUMBER_REQUIRED')
    return Fraction(str(value))


def _transition(transition):
    if (type(transition) is not list or len(transition) not in (2, 3) or
            any(type(row) is not list or len(row) != len(transition) for row in transition)):
        raise ValueError('TRANSITION_SHAPE_INVALID')
    p = [[_number(x) for x in row] for row in transition]
    if any(x < 0 or x > 1 for row in p for x in row) or any(sum(row) != 1 for row in p):
        raise ValueError('TRANSITION_NOT_STOCHASTIC')
    return p


def _solve(p):
    """Independent rational elimination of pi P = pi and sum(pi) = 1."""
    n = len(p)
    rows = [[p[i][j] - int(i == j) for i in range(n)] + [Fraction(0)]
            for j in range(n - 1)]
    rows.append([Fraction(1)] * (n + 1))
    for col in range(n):
        pivot = next((r for r in range(col, n) if rows[r][col]), None)
        if pivot is None:
            raise ValueError('STATIONARY_NOT_UNIQUE')
        rows[col], rows[pivot] = rows[pivot], rows[col]
        divisor = rows[col][col]
        rows[col] = [x / divisor for x in rows[col]]
        for r in range(n):
            if r != col:
                factor = rows[r][col]
                rows[r] = [x - factor * y for x, y in zip(rows[r], rows[col])]
    return [row[-1] for row in rows]


def check_stationary(transition, stationary):
    """Compare a supplied result to exact balance; never attest its origin."""
    p = _transition(transition)
    n = len(p)
    expected = _solve(p)
    if (type(stationary) is not list or len(stationary) != 1 or
            type(stationary[0]) is not list or len(stationary[0]) != n):
        raise ValueError('STATIONARY_SHAPE_INVALID')
    actual = [_number(x) for x in stationary[0]]
    if any(x < 0 or x > 1 for x in actual) or abs(sum(actual) - 1) > TOLERANCE:
        raise ValueError('STATIONARY_NOT_NORMALIZED')
    error = max(abs(actual[i] - expected[i]) for i in range(n))
    residual = max(abs(sum(actual[i] * p[i][j] for i in range(n)) - actual[j])
                   for j in range(n))
    if error > TOLERANCE or residual > TOLERANCE:
        raise ValueError('ORACLE_MISMATCH')
    return {'expected_exact': [str(x) for x in expected],
            'expected_error_inf': float(error), 'residual_inf': float(residual),
            'tolerance': TOLERANCE, 'native_execution': 'NOT_ATTESTED'}


def optional_stationary(transition, *, allow_fallback=False):
    """Fail-closed optional QuantEcon boundary with an explicitly selected oracle.

    No source path, generated code or caller-supplied execution credential is
    accepted. Original API execution is unavailable until dependency provenance
    and a bounded trusted worker are reviewed. Discovery imports no solver code.
    """
    from .engine_prerequisites import quantecon_stationary_prerequisites

    if type(allow_fallback) is not bool:
        raise ValueError('FALLBACK_CHOICE_INVALID')
    p = _transition(transition)
    # This optional slice intentionally accepts only irreducible chains.
    for start in range(len(p)):
        reached = {start}
        for _ in p:
            reached |= {j for i in tuple(reached) for j, x in enumerate(p[i]) if x > 0}
        if len(reached) != len(p):
            raise ValueError('TRANSITION_REDUCIBLE')
    result = quantecon_stationary_prerequisites()
    result['fallback'] = None
    if allow_fallback:
        exact = _solve(p)
        result['fallback'] = {
            'engine': 'SOCIETY_BOX_FRACTION_ORACLE',
            'stationary_exact': [str(x) for x in exact],
            'residual_exact': [str(sum(exact[i] * p[i][j] for i in range(len(p))) - exact[j])
                               for j in range(len(p))],
            'mode': 'NO_LLM', 'native_execution': 'NOT_RUN',
        }
    return result
