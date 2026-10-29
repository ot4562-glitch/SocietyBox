"""Finite-horizon observation-law comparison for explicit synthetic assumptions.

Uses the scalar Gaussian model and decimal bounds of mechanism_gaussian.
All candidates share dimensionless observation coordinates, step indices and
independent Gaussian noises. Equality of exact means and full covariances is
equality of these Gaussian observation laws only over the requested horizon.
It cannot identify a unique economic mechanism, interventions or real history.
"""
from fractions import Fraction
import hashlib
import json
import re

from .mechanism_gaussian import _PARAMETERS, _number, _rational


_ID = re.compile(r'[a-z][a-z0-9_-]{0,63}\Z', re.ASCII)
_KEYS = {'schema', 'data_kind', 'horizon', 'candidates'}
_CANDIDATE_KEYS = {'candidate_id', *_PARAMETERS}


def compare_synthetic_gaussian_mechanisms(spec):
    """Compare 2–8 fixed scalar structures over 1–3 mathematical steps.

    Returns all equivalence classes, including singletons, in declaration order.
    Distinct distributions do not establish global or causal identification.
    No observed data, likelihood, fitting, ranking or execution grants are used.
    Invalid inputs produce a fixed diagnostic without partial results or hash.
    """
    result = dict(schema='societybox-synthetic-equivalence-result-v1',
                  status='BLOCKED', assurance='SYNTHETIC_GAUSSIAN_MATH_ONLY',
                  comparison_scope='DECLARED_FINITE_HORIZON_OBSERVATION_LAW',
                  conclusion='NOT_ASSESSED', global_identification='NOT_ESTABLISHED',
                  causal_identification='NOT_ESTABLISHED', historical_as_of='NOT_ESTABLISHED',
                  parameter_estimation='NOT_IMPLEMENTED', forecast='NOT_IMPLEMENTED',
                  causal_claim='NOT_IMPLEMENTED', authorization='NOT_GRANTED',
                  native_parity='NOT_RUN', observation_distributions=[],
                  equivalence_classes=[], diagnostics=[])
    try:
        if (type(spec) is not dict or spec.keys() != _KEYS
                or spec['schema'] != 'societybox-synthetic-equivalence-v1'
                or spec['data_kind'] != 'SYNTHETIC'
                or type(spec['horizon']) is not int or not 1 <= spec['horizon'] <= 3
                or type(spec['candidates']) is not list
                or not 2 <= len(spec['candidates']) <= 8):
            raise ValueError('synthetic schema bounds')
        candidates, ids = [], set()
        for candidate in spec['candidates']:
            if type(candidate) is not dict or candidate.keys() != _CANDIDATE_KEYS:
                raise ValueError('candidate fields')
            name = candidate['candidate_id']
            if (type(name) is not str or not _ID.fullmatch(name) or name in ids):
                raise ValueError('unique bounded id')
            ids.add(name)
            parameters = tuple(_number(candidate[k]) for k in _PARAMETERS)
            _, p, _, q, _, r = parameters
            if p < 0 or q < 0 or r <= 0:
                raise ValueError('variance')
            candidates.append((name, parameters))
    except (ValueError, TypeError):
        result['diagnostics'] = ['EQUIVALENCE_INPUT_INVALID']
        return result

    horizon = spec['horizon']
    distributions, groups = [], {}
    for name, (m, p, a, q, h, r) in candidates:
        # Direct noise-coefficient construction (no filter or likelihood).
        # x[t] = a^(t+1)*x[-1] + sum_j<=t a^(t-j)*w[j].
        weights = [[a**(t + 1)] +
                   [a**(t - j) if j <= t else Fraction(0) for j in range(horizon)]
                   for t in range(horizon)]
        variances = [p] + [q] * horizon
        means = tuple(h * a**(t + 1) * m for t in range(horizon))
        covariance = tuple(tuple(
            h * h * sum(x * y * v for x, y, v in
                        zip(weights[i], weights[j], variances))
            + (r if i == j else 0) for j in range(horizon))
                           for i in range(horizon))
        groups.setdefault((means, covariance), []).append(name)
        distributions.append(dict(candidate_id=name, mean=[_rational(x) for x in means],
                                  covariance=[[_rational(x) for x in row]
                                              for row in covariance]))
    classes = list(groups.values())
    result.update(status='READY', horizon=horizon,
                  observation_step_indices=list(range(horizon)),
                  conclusion=('MECHANISM_UNDERDETERMINED' if any(len(g) > 1 for g in classes)
                              else 'DISTINCT_FINITE_HORIZON_DISTRIBUTIONS'),
                  observation_distributions=distributions, equivalence_classes=classes,
                  arithmetic='EXACT_RATIONAL',
                  input_sha256=hashlib.sha256(json.dumps(
                      spec, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True).encode('ascii')).hexdigest())
    return result
