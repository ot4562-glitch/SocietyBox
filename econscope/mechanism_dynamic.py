"""Exact bounded multivariate lag comparison; data-only, offline, no LLM.

Fits y[t] = b0 + sum(bj * series_j[t-lag_j]). Candidate formulas and units
are researcher declarations. Selection uses training SSE/residual degrees of
freedom on a common sample, not holdout performance. This is conditional
prediction arithmetic, not structural identification or original-engine parity.
"""
from datetime import date
from fractions import Fraction as F
import hashlib
import json
import re
import sys

from .workspace import ScopeError, canonical
from .strict_json import loads_strict

_NAME = re.compile(r'[a-z][a-z0-9_]{0,39}\Z', re.ASCII)
_NUMBER = re.compile(r'-?(?:0|[1-9][0-9]{0,5})(?:\.[0-9]{1,4})?\Z', re.ASCII)
_DAY = re.compile(r'[0-9]{4}-[0-9]{2}-[0-9]{2}\Z', re.ASCII)
MAX_BYTES = 131072


def _bad(code):
    raise ScopeError('IMFE_DYNAMIC_' + code)


def _name(value):
    return type(value) is str and _NAME.fullmatch(value) is not None


def _day(value):
    if type(value) is not str or not _DAY.fullmatch(value):
        _bad('DATE_INVALID')
    try:
        return date.fromisoformat(value)
    except ValueError:
        _bad('DATE_INVALID')


def _validate_structure(q, *, candidate_limit=4, input_size_cap=MAX_BYTES, minimum_train=12, minimum_holdout=8):
    """Private structure parser; executing callers must apply their time policy."""
    fields = {'schema', 'data_kind', 'source_declared_sha256', 'target',
              'series_units', 'training_end_day', 'observations', 'candidates'}
    if type(q) is not dict or q.keys() != fields:
        _bad('SPEC_INVALID')
    if (q['schema'] != 'societybox-dynamic-inverse-v1' or
            q['data_kind'] not in ('SYNTHETIC', 'DECLARED_PUBLIC_OBSERVATIONS') or
            type(q['source_declared_sha256']) is not str or
            not re.fullmatch('[a-f0-9]{64}', q['source_declared_sha256'])):
        _bad('SPEC_INVALID')
    units = q['series_units']
    if (type(units) is not dict or not 2 <= len(units) <= 8 or
            any(not _name(k) or type(v) is not str or not 1 <= len(v) <= 64 or
                any(ord(ch) < 32 for ch in v) for k, v in units.items()) or
            not _name(q['target']) or q['target'] not in units):
        _bad('SERIES_INVALID')
    candidates = q['candidates']
    if type(candidates) is not list or not 1 <= len(candidates) <= candidate_limit:
        _bad('CANDIDATE_LIMIT')
    seen, maxlag = set(), 0
    for c in candidates:
        if (type(c) is not dict or c.keys() != {'candidate_id', 'features'} or
                not _name(c['candidate_id']) or c['candidate_id'] in seen or
                type(c['features']) is not list or not 1 <= len(c['features']) <= 4):
            _bad('CANDIDATE_INVALID')
        seen.add(c['candidate_id'])
        features = set()
        for f in c['features']:
            if (type(f) is not dict or f.keys() != {'series', 'lag'} or
                    not _name(f['series']) or f['series'] not in units or
                    type(f['lag']) is not int or not 1 <= f['lag'] <= 7):
                _bad('FEATURE_INVALID')
            key = (f['series'], f['lag'])
            if key in features:
                _bad('FEATURE_DUPLICATE')
            features.add(key)
            maxlag = max(maxlag, f['lag'])
    raw = q['observations']
    if type(raw) is not list or not 24 <= len(raw) <= 128:
        _bad('OBSERVATION_LIMIT')
    rows = []
    for row in raw:
        if (type(row) is not dict or row.keys() != {'day', 'values', 'available_on'} or
                type(row['values']) is not dict or row['values'].keys() != units.keys() or
                type(row['available_on']) is not dict or row['available_on'].keys() != units.keys()):
            _bad('ROW_INVALID')
        day = _day(row['day'])
        if rows and (day-rows[-1][0]).days != 1:
            _bad('DAILY_CONTINUITY_REQUIRED')
        values, available = {}, {}
        for name in units:
            v = row['values'][name]
            if type(v) is not str or not _NUMBER.fullmatch(v):
                _bad('NUMBER_INVALID')
            values[name] = F(v)
            available[name] = _day(row['available_on'][name])
            if available[name] < day:
                _bad('RELEASE_BEFORE_OBSERVATION')
        rows.append((day, values, available))
    cutoff_day = _day(q['training_end_day'])
    indices = [i for i, row in enumerate(rows) if row[0] == cutoff_day]
    if not indices:
        _bad('CUTOFF_NOT_OBSERVED')
    cutoff = indices[0]
    if cutoff-maxlag+1 < minimum_train or len(rows)-cutoff-1 < minimum_holdout:
        _bad('TRAIN_HOLDOUT_LIMIT')
    if len(canonical(q)) > input_size_cap:
        _bad('INPUT_TOO_LARGE')
    return rows, cutoff, maxlag


def _validate(q, *, candidate_limit=4, input_size_cap=MAX_BYTES):
    # Preserve the public historical-one-step contract. Forecast validation uses
    # the shared parser and separately checks the current origin's information set.
    rows, cutoff, maxlag = _validate_structure(q, candidate_limit=candidate_limit, input_size_cap=input_size_cap)
    cutoff_day = rows[cutoff][0]
    candidates = q['candidates']
    target = q['target']
    if any(rows[t][2][target] > cutoff_day for t in range(maxlag, cutoff+1)):
        _bad('TRAINING_TARGET_UNAVAILABLE')
    # Validate every candidate before fitting/ranking, including singular ones.
    # Information-set policy is conservative: each historical regressor must
    # have been released by its own one-step origin, even for training rows.
    for t in range(maxlag, len(rows)):
        origin = rows[t-1][0]
        for c in candidates:
            if any(rows[t-f['lag']][2][f['series']] > origin for f in c['features']):
                _bad('PREDICTOR_UNAVAILABLE')
        if t > cutoff and rows[t-1][2][target] > origin:
            _bad('BASELINE_UNAVAILABLE')
    return rows, cutoff, maxlag


def _solve(a, b):
    """Exact pivoted elimination, at most five parameters; no pseudoinverse."""
    n = len(b)
    m = [list(row)+[v] for row, v in zip(a, b)]
    for j in range(n):
        pivot = next((i for i in range(j, n) if m[i][j]), None)
        if pivot is None:
            return None
        m[j], m[pivot] = m[pivot], m[j]
        scale = m[j][j]
        m[j] = [v/scale for v in m[j]]
        for i in range(n):
            if i != j:
                scale = m[i][j]
                m[i] = [v-scale*w for v, w in zip(m[i], m[j])]
    return [row[-1] for row in m]


def _dot(a, b):
    return sum((v*w for v, w in zip(a, b)), F(0))


def _fit_design_exact(x, y):
    """Shared exact OLS for already validated, time-policy-filtered design rows."""
    n, p = len(y), len(x[0])
    gram = [[sum(v[i]*v[j] for v in x) for j in range(p)] for i in range(p)]
    rhs = [sum(v[i]*value for v, value in zip(x, y)) for i in range(p)]
    beta = _solve(gram, rhs)
    if beta is None:
        return None
    residuals = [value-_dot(v, beta) for v, value in zip(x, y)]
    if any(sum(v[j]*e for v, e in zip(x, residuals)) != 0 for j in range(p)):
        _bad('NORMAL_EQUATIONS_FAILED')
    sse = _dot(residuals, residuals)
    return beta, residuals, sse, sse/(n-p)


def reconstruct_dynamic(q):
    """Fit declared lag candidates and replay rolling one-step holdout forecasts.

    Release dates are declarations, not authenticated historical vintages.
    Holdout targets may be released after forecast origin: scoring is ex post.
    No latent identities, confidence intervals or future actuals are inferred.
    """
    return _reconstruct_with_validated_rows(q, _validate(q))


def reconstruct_dynamic_screen_batch(q):
    """Screen-only bounded batch: validate all generated candidates once.

    This is a trusted internal variant, not a general engine accepting arbitrary
    researcher code. Extra bytes only accommodate generated candidate records;
    original caller input has already passed inverse_screen's 128-KiB cap.
    Legacy reconstruct_dynamic keeps its four-candidate and 128-KiB contract.
    """
    return _reconstruct_with_validated_rows(
        q, _validate(q, candidate_limit=25, input_size_cap=MAX_BYTES + 8192))


def _reconstruct_with_validated_rows(q, validated):
    rows, cutoff, start = validated
    target = q['target']
    train_y = [rows[t][1][target] for t in range(start, cutoff+1)]
    n = len(train_y)
    holdout_count = len(rows)-cutoff-1
    results, ranking = [], []
    for c in q['candidates']:
        design = [[F(1)]+[rows[t-f['lag']][1][f['series']] for f in c['features']]
                  for t in range(start, len(rows))]
        x = design[:n]
        p = len(x[0])
        fit = _fit_design_exact(x, train_y)
        r = dict(candidate_id=c['candidate_id'], features=[dict(f) for f in c['features']],
                 parameter_order=['intercept']+[f"{f['series']}[t-{f['lag']}]" for f in c['features']])
        if fit is None:
            r['status'] = 'RANK_DEFICIENT'
            results.append(r)
            continue
        beta, residuals, sse, score = fit
        cells = []
        for t, v in zip(range(cutoff+1, len(rows)), design[n:]):
            pred = _dot(v, beta)
            actual = rows[t][1][target]
            cells.append(dict(origin_day=rows[t-1][0].isoformat(),
                              target_day=rows[t][0].isoformat(), prediction_exact=str(pred),
                              observed_exact=str(actual), residual_exact=str(actual-pred)))
        r.update(status='FIT', coefficients_exact=list(map(str, beta)),
                 training_sse_exact=str(sse), training_residuals_exact=list(map(str, residuals)),
                 residual_degrees_of_freedom=n-p, selection_score_exact=str(score),
                 holdout_mse_exact=str(sum(F(v['residual_exact'])**2 for v in cells)/holdout_count),
                 holdout=cells)
        ranking.append((score, p, c['candidate_id']))
        results.append(r)
    baseline_errors = [rows[t][1][target]-rows[t-1][1][target]
                       for t in range(cutoff+1, len(rows))]
    return dict(schema='societybox-dynamic-inverse-result-v1',
                status='READY' if ranking else 'INDETERMINATE',
                input_sha256=hashlib.sha256(canonical(q)).hexdigest(),
                source_declared_sha256=q['source_declared_sha256'], data_kind=q['data_kind'],
                target=target, series_units=q['series_units'],
                training=dict(start_day=rows[start][0].isoformat(), end_day=rows[cutoff][0].isoformat(),
                              observations=n, common_max_lag=start),
                selection='MIN_TRAINING_SSE_OVER_RESIDUAL_DF_TIES_FEWER_PARAMETERS_THEN_ID',
                selected_candidate_id=min(ranking)[2] if ranking else None,
                evaluation='RETROSPECTIVE_ROLLING_ONE_STEP_DECLARED_RELEASES_ONLY',
                baseline=dict(model='PREVIOUS_OBSERVED_TARGET', observations=holdout_count,
                              holdout_mse_exact=str(_dot(baseline_errors, baseline_errors)/holdout_count)),
                candidates=results,
                assurance=dict(causal_identification='NOT_ESTABLISHED', latent_identity='UNKNOWN',
                    original_paper_specification='UNKNOWN', upstream_native_parity='NOT_RUN',
                    historical_release_authentication='NOT_ESTABLISHED', source_authentication='NOT_ESTABLISHED',
                    future_actual_values='UNKNOWN', llm_execution='NOT_USED',
                    statistical_significance='NOT_CALCULATED',
                    assumptions=['DECLARED_LINEAR_LAG_FORMULAS', 'FIXED_PARAMETERS',
                                 'COMPLETE_DAILY_DATA', 'DECLARED_RELEASE_DATES_UNVERIFIED',
                                 'EXOGENEITY_AND_ERROR_LAW_NOT_VERIFIED']))


def main():
    try:
        data = sys.stdin.buffer.read(MAX_BYTES+1)
        if len(data) > MAX_BYTES:
            _bad('INPUT_TOO_LARGE')
        result = reconstruct_dynamic(loads_strict(data))
    except (ScopeError, ValueError, UnicodeError, RecursionError) as exc:
        print(json.dumps(dict(status='INVALID', error=str(exc)[:200])))
        return 2
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
