"""Bounded synthetic scalar Gaussian mathematics, independent of study execution.

Known parameters only; no fitting, sampling, native adapter or user code.
For t=0,...,n-1: x[t] = a*x[t-1] + w[t], y[t] = h*x[t] + v[t].
The supplied initial distribution is for x[-1]. Independent Gaussian noises
have variances q and r; all parameters are explicit dimensionless assumptions.
Filtering conditions on y[0:t+1]; RTS smoothing conditions on all supplied y.
Priors at t condition only on y[0:t]; next_prior is the mathematical one-step
propagation after the final filter, not a validated empirical forecast.
These indices are mathematical steps, not publication dates or access proofs.
"""
from fractions import Fraction
import hashlib
import json
import re


_NUMBER = re.compile(r'-?(?:0|[1-9][0-9]*)(?:\.[0-9]{1,12})?\Z', re.ASCII)
_PARAMETERS = ('initial_mean', 'initial_variance', 'transition',
               'process_variance', 'observation_loading', 'measurement_variance')
_KEYS = {'schema', 'data_kind', 'observations', *_PARAMETERS}


def _number(value):
    if (type(value) is not str or len(value) > 64
            or not _NUMBER.fullmatch(value)):
        raise ValueError('decimal required')
    number = Fraction(value)
    if abs(number) > 10**12:
        raise ValueError('magnitude bound')
    return number


def _rational(value):
    return dict(numerator=str(value.numerator), denominator=str(value.denominator))


def synthetic_gaussian_states(spec):
    """Return exact rational conditional means/variances for 1–3 synthetic steps.

    Strict JSON-shaped input, fixed fields, finite decimal strings (at most 12
    fractional digits, magnitude <= 1e12). Initial/process variances may be zero;
    measurement variance must be positive. No missing observations or matrices.
    The small fixed bounds limit exact-rational arithmetic and output growth.
    BLOCKED output contains no input values, hash or partial numerical results.
    READY proves only this declared mathematical calculation, never historical
    availability, causal identification, parameter fit or forecast skill.
    """
    result = dict(schema='societybox-synthetic-gaussian-result-v1', status='BLOCKED',
                  assurance='SYNTHETIC_GAUSSIAN_MATH_ONLY',
                  claim_level='SIMULATED_RETROSPECTIVE_ANALYSIS',
                  historical_as_of='NOT_ESTABLISHED', authorization='NOT_GRANTED',
                  parameter_estimation='NOT_IMPLEMENTED', forecast='NOT_IMPLEMENTED',
                  identify='NOT_IMPLEMENTED', causal_claim='NOT_IMPLEMENTED',
                  initial_prior=None, priors=[], next_prior=None,
                  filtered=[], smoothed=[], diagnostics=[])
    try:
        if (type(spec) is not dict or spec.keys() != _KEYS
                or spec['schema'] != 'societybox-synthetic-gaussian-v1'
                or spec['data_kind'] != 'SYNTHETIC'):
            raise ValueError('synthetic schema required')
        m, p, a, q, h, r = (_number(spec[k]) for k in _PARAMETERS)
        observations = spec['observations']
        if type(observations) is not list or not 1 <= len(observations) <= 3:
            raise ValueError('observation bounds')
        observations = [_number(y) for y in observations]
        if p < 0 or q < 0 or r <= 0:
            raise ValueError('variance')
    except (ValueError, TypeError):
        result['diagnostics'] = ['GAUSSIAN_INPUT_INVALID']
        return result

    initial = (m, p)
    # Forward scalar Kalman recursion, including the transition from x[-1].
    predicted, filtered = [], []
    for y in observations:
        m, p = a * m, a * a * p + q
        predicted.append((m, p))
        innovation_variance = h * h * p + r
        gain = p * h / innovation_variance
        m = m + gain * (y - h * m)
        # Joseph form is also valid for deterministic states (p=0).
        p = (1 - gain * h)**2 * p + gain * gain * r
        filtered.append((m, p))

    # Rauch–Tung–Striebel backward recursion. A zero predicted variance
    # means the next state is deterministic; its cross covariance is zero.
    smoothed = list(filtered)
    for i in range(len(filtered) - 2, -1, -1):
        mean, variance = filtered[i]
        next_mean, next_variance = predicted[i + 1]
        gain = variance * a / next_variance if next_variance else Fraction(0)
        smoothed[i] = (mean + gain * (smoothed[i + 1][0] - next_mean),
                       variance + gain * gain * (smoothed[i + 1][1] - next_variance))

    def cell(i, mean, variance, mode, count):
        return dict(step_index=i, information_mode=mode,
                    conditioned_observation_indices=list(range(count)),
                    uses_future_observations=(mode == 'SMOOTHED' and i < len(observations) - 1),
                    mean=_rational(mean), variance=_rational(variance))

    def cells(states, mode):
        return [cell(i, mean, variance, mode,
                     i if mode == 'PRIOR' else
                     i + 1 if mode == 'FILTERED' else len(observations))
                for i, (mean, variance) in enumerate(states)]

    result.update(status='READY', filtered=cells(filtered, 'FILTERED'),
                  smoothed=cells(smoothed, 'SMOOTHED'),
                  initial_prior=cell(-1, *initial, 'INITIAL_PRIOR', 0),
                  priors=cells(predicted, 'PRIOR'),
                  next_prior=cell(len(observations), a * m, a * a * p + q,
                                  'PRIOR', len(observations)),
                  input_sha256=hashlib.sha256(json.dumps(
                      spec, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True).encode('ascii')).hexdigest(),
                  arithmetic='EXACT_RATIONAL',
                  uncertainty_scope='LATENT_STATE_CONDITIONAL_ON_FIXED_DECLARED_PARAMETERS',
                  parameter_uncertainty='NOT_ASSESSED', model_uncertainty='NOT_ASSESSED',
                  source_availability='NOT_ASSESSED', native_parity='NOT_RUN')
    return result
