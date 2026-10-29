"""Strict optional macro boundary. No upstream execution is currently enabled.

Independent oracle: maximize log(c0)+beta*log(c1), subject to
c0+c1/R=m+1/R. Two dates, terminal income 1, survival/growth 1,
unrestricted borrowing, terminal assets zero; no uncertainty or equilibrium.
This explicit model is never substituted for arbitrary researcher models.
"""
from fractions import Fraction
import hashlib
import json
import math
import sys

from .engine_prerequisites import macro_prerequisites
from .strict_json import loads_strict


def _number(value, limit=10**12):
    if type(value) not in (int, float):
        raise ValueError('FINITE_BOUNDED_NUMBER_REQUIRED')
    if (type(value) is int and abs(value) > limit or
            type(value) is float and (not math.isfinite(value) or abs(value) > limit)):
        raise ValueError('FINITE_BOUNDED_NUMBER_REQUIRED')
    return Fraction(str(value))


def _parameters(parameters):
    if type(parameters) is not dict or set(parameters) != {'beta', 'gross_return', 'resources'}:
        raise ValueError('MACRO_PARAMETERS_INVALID')
    beta, rate, wealth = (_number(parameters[k], 10**9) for k in ('beta', 'gross_return', 'resources'))
    if not Fraction(1, 10000) <= beta <= 1 or not Fraction(1, 10000) <= rate <= 100 or wealth < 0:
        raise ValueError('MACRO_PARAMETER_DOMAIN_INVALID')
    return beta, rate, wealth


def _oracle(parameters):
    beta, rate, wealth = _parameters(parameters)
    c0 = (wealth + 1/rate)/(1+beta)
    c1 = beta*rate*c0
    return [c0, c1]


def check_consumption(parameters, consumption):
    """Validate supplied numbers; this never certifies their upstream origin."""
    expected = _oracle(parameters)
    if type(consumption) is not list or len(consumption) != 2:
        raise ValueError('CONSUMPTION_SHAPE_INVALID')
    actual = [_number(x) for x in consumption]
    if min(actual) <= 0:
        raise ValueError('CONSUMPTION_POSITIVE_REQUIRED')
    error = max(abs(x-y)/max(Fraction(1), abs(y)) for x, y in zip(actual, expected))
    if error > Fraction(1, 10**12):
        raise ValueError('CONSUMPTION_ORACLE_MISMATCH')
    return {'expected_exact': [str(x) for x in expected],
            'scaled_error_inf': float(error), 'tolerance': 1e-12,
            'native_execution': 'NOT_ATTESTED'}


def optional_macro(request, *, allow_oracle=False):
    """Only a named data-only model is accepted; metadata grants no execution."""
    if type(allow_oracle) is not bool:
        raise ValueError('ORACLE_CHOICE_INVALID')
    if type(request) is not dict or set(request) != {'engine', 'model', 'parameters'}:
        raise ValueError('MACRO_REQUEST_SCHEMA_INVALID')
    if type(request['engine']) is not str or request['engine'] not in ('HARK', 'Dolo', 'Dynare'):
        raise ValueError('MACRO_ENGINE_INVALID')
    if request['model'] != 'two_period_log_consumption':
        raise ValueError('MACRO_MODEL_UNSUPPORTED')
    _parameters(request['parameters'])
    result = next(x for x in macro_prerequisites()['candidates'] if x['engine'] == request['engine'])
    result.update(model=request['model'], fallback=None, generated_code='BLOCKED',
                  os_isolation='NOT_ATTESTED',
                  request_sha256=hashlib.sha256(json.dumps(request, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode()).hexdigest())
    if allow_oracle:
        c0, c1 = _oracle(request['parameters'])
        beta, rate, wealth = _parameters(request['parameters'])
        result['fallback'] = {
            'engine': 'SOCIETY_BOX_FRACTION_ORACLE', 'mode': 'NO_LLM',
            'consumption': [float(c0), float(c1)],
            'consumption_exact': [str(c0), str(c1)],
            'budget_residual_exact': str(c0+c1/rate-wealth-1/rate),
            'euler_residual_exact': str(1/c0-beta*rate/c1),
            'assumptions': ['two dates', 'log utility', 'terminal income 1',
                            'survival and growth 1', 'unrestricted borrowing',
                            'terminal assets zero', 'no uncertainty'],
            'native_execution': 'NOT_RUN', 'equilibrium': 'NOT_ESTABLISHED',
        }
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-oracle', action='store_true')
    args = parser.parse_args()
    try:
        data = sys.stdin.buffer.read(8193)
        if len(data) > 8192:
            raise ValueError('MACRO_INPUT_TOO_LARGE')
        result = optional_macro(loads_strict(data), allow_oracle=args.allow_oracle)
    except (ValueError, UnicodeError, RecursionError) as exc:
        print(json.dumps({'status': 'INVALID', 'error': str(exc)[:200]}))
        return 2
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
