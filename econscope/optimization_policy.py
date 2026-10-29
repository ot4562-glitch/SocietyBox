"""Bounded data-only LP adapter with an explicitly selected independent fallback.

Maximize c*x subject to A*x <= b and x >= 0. No integer variables, equality
syntax, free variables, causal estimator, scripts or optional imports. Exact
vertex enumeration is for small models, not a native Pyomo/HiGHS substitute.
"""
from fractions import Fraction
from itertools import combinations

from .strict_json import loads_strict
import hashlib
import json
import math
import sys


def _number(value):
    if type(value) not in (int, float) or abs(value) > 10**9:
        raise ValueError('LP_FINITE_BOUNDED_NUMBER_REQUIRED')
    if type(value) is float and not math.isfinite(value):
        raise ValueError('LP_FINITE_BOUNDED_NUMBER_REQUIRED')
    q = Fraction(str(value))
    if max(q.numerator.bit_length(), q.denominator.bit_length()) > 128:
        raise ValueError('LP_NUMBER_PRECISION_LIMIT')
    return q


def _validate(request):
    fields = {'engine', 'model', 'objective', 'constraints', 'bounds'}
    if type(request) is not dict or set(request) != fields:
        raise ValueError('LP_REQUEST_SCHEMA_INVALID')
    if type(request['engine']) is not str or request['engine'] not in ('Pyomo', 'HiGHS'):
        raise ValueError('LP_ENGINE_UNSUPPORTED')
    if request['model'] != 'linear_program_max':
        raise ValueError('LP_MODEL_UNSUPPORTED')
    c, a, b = (request[k] for k in ('objective', 'constraints', 'bounds'))
    if (type(c) is not list or not 1 <= len(c) <= 4 or
            type(a) is not list or not 1 <= len(a) <= 8 or
            type(b) is not list or len(b) != len(a) or
            any(type(row) is not list or len(row) != len(c) for row in a)):
        raise ValueError('LP_DIMENSION_INVALID')
    return list(map(_number, c)), [list(map(_number, row)) for row in a], list(map(_number, b))


def _dot(a, b):
    return sum((x*y for x, y in zip(a, b)), Fraction(0))


def _solve_square(a, b):
    """Singular active sets are rejected; they are not identified solutions."""
    n = len(b)
    rows = [list(row)+[v] for row, v in zip(a, b)]
    for j in range(n):
        pivot = next((i for i in range(j, n) if rows[i][j]), None)
        if pivot is None:
            return None
        rows[j], rows[pivot] = rows[pivot], rows[j]
        scale = rows[j][j]
        rows[j] = [v/scale for v in rows[j]]
        for i in range(n):
            if i != j:
                factor = rows[i][j]
                rows[i] = [v-factor*w for v, w in zip(rows[i], rows[j])]
    return [row[-1] for row in rows]


def _vertices(a, b, n):
    # Nonnegativity makes this polyhedron pointed: if nonempty it has a vertex.
    a = a + [[Fraction(-int(i == j)) for j in range(n)] for i in range(n)]
    b = b + [Fraction(0)]*n
    for active in combinations(range(len(b)), n):
        x = _solve_square([a[i] for i in active], [b[i] for i in active])
        if x is not None and all(_dot(row, x) <= v for row, v in zip(a, b)):
            yield x


def _fallback(c, a, b):
    n, m = len(c), len(b)
    r = {'engine': 'SOCIETY_BOX_EXACT_LP', 'mode': 'NO_LLM',
         'status': 'INFEASIBLE', 'solution_exact': None, 'objective_exact': None,
         'dual_exact': None, 'duality_gap_exact': None,
         'native_execution': 'NOT_RUN', 'native_parity': 'NOT_RUN',
         'causal_effect': 'UNKNOWN', 'real_policy_counterfactual': 'UNKNOWN',
         'assumptions': ['researcher-supplied coefficients are unverified',
                         'continuous nonnegative decision variables',
                         'maximize objective; all supplied constraints are <=',
                         'linear deterministic model only; no causal identification'],
         'rank_policy': 'SINGULAR_ACTIVE_SETS_SKIPPED_NO_PSEUDOINVERSE'}
    points = list(_vertices(a, b, n))
    if not points:
        return r
    # A feasible improving recession ray proves unboundedness. Normalization
    # makes this auxiliary polytope compact; no arbitrary bounding box is used.
    rays = _vertices(a+[[Fraction(1)]*n], [Fraction(0)]*m+[Fraction(1)], n)
    ray = next((d for d in rays if _dot(c, d) > 0), None)
    if ray is not None:
        r.update(status='UNBOUNDED', improving_ray_exact=list(map(str, ray)))
        return r
    x = max(points, key=lambda p: _dot(c, p))
    # Dual: min b*y, A^T*y >= c, y >= 0. This independently supplies an upper
    # bound; only exact equality with the feasible primal authorizes OPTIMAL.
    da = [[-a[i][j] for i in range(m)] for j in range(n)]
    candidates = list(_vertices(da, [-v for v in c], m))
    if not candidates:
        r.update(status='CERTIFICATE_FAILED')
        return r
    y = min(candidates, key=lambda p: _dot(b, p))
    objective, upper = _dot(c, x), _dot(b, y)
    if objective != upper:
        r.update(status='CERTIFICATE_FAILED')
        return r
    r.update(status='OPTIMAL', solution_exact=list(map(str, x)),
             objective_exact=str(objective), dual_exact=list(map(str, y)),
             duality_gap_exact=str(upper-objective),
             primal_slack_exact=[str(v-_dot(row, x)) for row, v in zip(a, b)],
             dual_slack_exact=[str(-_dot(row, y)-v) for row, v in zip(da, c)])
    return r


def optimization_prerequisites():
    """Read metadata only. Presence never grants native execution or clearance."""
    from importlib import metadata
    dependencies = []
    for name in ('pyomo', 'highspy', 'dowhy', 'doubleml', 'numpy', 'scipy',
                 'pandas', 'scikit-learn'):
        try:
            version, state = metadata.version(name), 'METADATA_FOUND'
        except metadata.PackageNotFoundError:
            version, state = None, 'NOT_INSTALLED'
        except (OSError, ValueError, UnicodeError):
            version, state = None, 'UNKNOWN'
        dependencies.append(dict(distribution=name, version=version, state=state))
    return {'status': 'BLOCKED', 'native_execution': 'NOT_RUN', 'native_parity': 'NOT_RUN',
            'dependencies': dependencies, 'causal_effect': 'UNKNOWN',
            'redistribution': 'NOT_GRANTED', 'os_isolation': 'NOT_ATTESTED',
            'blockers': ['TRUSTED_NATIVE_WORKER_NOT_IMPLEMENTED',
                         'IMPORT_PROVENANCE_COMPILED_ABI_NOT_ATTESTED',
                         'FILE_LEVEL_TRANSITIVE_LICENSE_REVIEW_REQUIRED',
                         'CAUSAL_IDENTIFICATION_AND_ESTIMATOR_NOT_IMPLEMENTED']}


def optional_optimization(request, *, allow_fallback=False):
    """Explicit LP request; never substitute a model or execute supplied code."""
    if type(allow_fallback) is not bool:
        raise ValueError('LP_FALLBACK_CHOICE_INVALID')
    c, a, b = _validate(request)
    result = optimization_prerequisites()
    result.update(engine=request['engine'], model=request['model'], fallback=None,
                  request_sha256=hashlib.sha256(json.dumps(request, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode()).hexdigest())
    if allow_fallback:
        result['fallback'] = _fallback(c, a, b)
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-fallback', action='store_true')
    args = parser.parse_args()
    try:
        data = sys.stdin.buffer.read(8193)
        if len(data) > 8192:
            raise ValueError('LP_INPUT_TOO_LARGE')
        result = optional_optimization(loads_strict(data), allow_fallback=args.allow_fallback)
    except (ValueError, UnicodeError, RecursionError) as exc:
        print(json.dumps({'status': 'INVALID', 'error': str(exc)[:200]}))
        return 2
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
