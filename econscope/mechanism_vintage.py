"""Read-only synthetic vintage contract, independent of societybox-study-v1.

No IO, execution, estimation, approval or source authentication. Release proofs
are caller-supplied trust anchors for synthetic tests, never external verification.
All instants require explicit offsets and seconds; intervals are [from, to).
Dates represent observation periods, never release instants. Output is bounded
to 1000 rows and diagnostics; values are plain decimal strings (128 chars max).
"""
from copy import deepcopy
from datetime import date, datetime, timezone
import re


_INSTANT = re.compile(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-](?:[01]\d|2[0-3]):[0-5]\d)\Z')
_DECIMAL = re.compile(r'-?(?:0|[1-9]\d*)(?:\.\d+)?\Z')
_DIGEST = re.compile(r'[0-9a-f]{64}\Z')
_LEVELS = ('VERIFIED_PROVIDER_RELEASE', 'VERIFIED_ARCHIVE_OBSERVED')
_ROW_KEYS = set(('country source_id series_id observation_period_start '
    'observation_period_end value_decimal unit frequency statistical_definition '
    'release_at valid_from valid_to local_acquired_at vintage_id source_digest '
    'availability_evidence rights').split())


def _text(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 512:
        raise ValueError('text')
    return value


def _instant(value):
    if (not isinstance(value, str) or not _INSTANT.fullmatch(value)
            or value.endswith('-00:00')):
        raise ValueError('offset instant required')
    return datetime.fromisoformat(value.replace('Z', '+00:00')).astimezone(timezone.utc)


def _date(value):
    if not isinstance(value, str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
        raise ValueError('date')
    return date.fromisoformat(value)


def _digest(value):
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise ValueError('digest')
    return value


def _keys(value, keys):
    if not isinstance(value, dict) or value.keys() != keys:
        raise ValueError('fields')


def _inspect(ledger, policy, as_of):
    diagnostics = []

    def issue(code, row=None, blocking=True):
        if len(diagnostics) < 1000:
            diagnostics.append(dict(code=code, row_index=row, blocking=blocking))
        elif blocking:
            # Truncation must never conceal a rejection behind exclusions.
            diagnostics[-1] = dict(code=code, row_index=row, blocking=True)

    try:
        cutoff = _instant(as_of)
    except (ValueError, OverflowError):
        issue('LEDGER_CUTOFF_INVALID')
        return diagnostics, []
    try:
        _keys(policy, {'schema', 'allowed_source_ids', 'release_proofs', 'claim_goal'})
        if policy['schema'] != 'societybox-ledger-policy-v1':
            raise ValueError('schema')
        sources = policy['allowed_source_ids']
        proofs = policy['release_proofs']
        if (not isinstance(sources, list) or not 1 <= len(sources) <= 1000
                or not isinstance(proofs, list) or len(proofs) > 1000):
            raise ValueError('policy bounds')
        sources = {_text(s) for s in sources}
        trusted = set()
        for proof in proofs:
            _keys(proof, {'proof_digest', 'source_digest', 'source_id', 'release_at', 'level'})
            level = _text(proof['level'])
            if level not in _LEVELS:
                raise ValueError('proof level')
            trusted.add((_digest(proof['proof_digest']), _digest(proof['source_digest']),
                         _text(proof['source_id']), _instant(proof['release_at']), level))
        if policy['claim_goal'] != 'SIMULATED_RETROSPECTIVE_ANALYSIS':
            issue('LEDGER_CLAIM_UNSUPPORTED')
    except (ValueError, TypeError, OverflowError):
        issue('LEDGER_POLICY_INVALID')
        return diagnostics, []
    try:
        _keys(ledger, {'schema', 'data_kind', 'rows', 'transformations'})
        if (ledger['schema'] != 'societybox-observation-ledger-v1'
                or ledger['data_kind'] != 'SYNTHETIC'):
            raise ValueError('synthetic schema required')
        rows, transforms = ledger['rows'], ledger['transformations']
        if (not isinstance(rows, list) or not 1 <= len(rows) <= 1000
                or not isinstance(transforms, list) or len(transforms) > 1000):
            raise ValueError('bounds')
    except (ValueError, TypeError):
        issue('LEDGER_SCHEMA_INVALID')
        return diagnostics, []

    # Transform declarations are inspected, never executed or taken as proof
    # that real preprocessing was restricted to the declared information set.
    for transform in transforms:
        try:
            _keys(transform, {'transform_id', 'fitted_at', 'fit_data_cutoff'})
            _text(transform['transform_id'])
            fitted = _instant(transform['fitted_at'])
            fit_cutoff = _instant(transform['fit_data_cutoff'])
            if fitted > cutoff or fit_cutoff > cutoff:
                issue('LEDGER_TRANSFORM_FUTURE')
            if fit_cutoff > fitted:
                issue('LEDGER_TRANSFORM_TIME_INVALID')
        except (ValueError, TypeError, OverflowError):
            issue('LEDGER_TRANSFORM_INVALID')

    groups, definitions, vintage_ids = {}, {}, set()
    for index, row in enumerate(rows):
        try:
            _keys(row, _ROW_KEYS)
            for field in ('country', 'source_id', 'series_id', 'unit', 'frequency',
                          'statistical_definition', 'vintage_id'):
                _text(row[field])
            if row['country'] not in ('US', 'CN', 'KR', 'JP'):
                raise ValueError('country')
            if row['frequency'] not in ('day', 'month', 'quarter', 'year'):
                raise ValueError('frequency')
            start, end = _date(row['observation_period_start']), _date(row['observation_period_end'])
            value = row['value_decimal']
            if (not isinstance(value, str) or len(value) > 128
                    or not _DECIMAL.fullmatch(value) or end < start):
                raise ValueError('value or period')
            release, begin = _instant(row['release_at']), _instant(row['valid_from'])
            finish = None if row['valid_to'] is None else _instant(row['valid_to'])
            acquired = _instant(row['local_acquired_at'])
            if begin < release or (finish is not None and finish <= begin) or acquired < release:
                raise ValueError('interval')
            if end > datetime.fromisoformat(row['release_at'].replace('Z', '+00:00')).date():
                raise ValueError('future observation')
            _digest(row['source_digest'])
            evidence, rights = row['availability_evidence'], row['rights']
            _keys(evidence, {'level', 'proof_digest'})
            level, proof = _text(evidence['level']), _digest(evidence['proof_digest'])
            _keys(rights, {'allow_local_analysis', 'license_id'})
            _text(rights['license_id'])
            if type(rights['allow_local_analysis']) is not bool:
                raise ValueError('rights type')
        except (ValueError, TypeError, OverflowError):
            issue('LEDGER_ROW_INVALID', index)
            continue
        if not rights['allow_local_analysis']:
            issue('LEDGER_RIGHTS_DENIED', index)
        if row['source_id'] not in sources:
            issue('LEDGER_SOURCE_DENIED', index)
        if (proof, row['source_digest'], row['source_id'], release, level) not in trusted:
            issue('LEDGER_RELEASE_UNVERIFIED', index)
        if release > cutoff or begin > cutoff:
            issue('LEDGER_FUTURE_EXCLUDED', index, blocking=False)
        series = (row['source_id'], row['series_id'])
        definition = tuple(row[f] for f in ('country', 'unit', 'frequency', 'statistical_definition'))
        if series in definitions and definitions[series] != definition:
            issue('LEDGER_DEFINITION_MIXED', index)
        definitions[series] = definition
        key = (*series, row['country'], start, end)
        vintage_key = (*key, row['vintage_id'])
        if vintage_key in vintage_ids:
            issue('LEDGER_VINTAGE_CONFLICT', index)
        vintage_ids.add(vintage_key)
        groups.setdefault(key, []).append((begin, release, finish, index, row))

    selected = []
    for versions in groups.values():
        versions.sort(key=lambda v: v[0])
        for previous, current in zip(versions, versions[1:]):
            if (previous[2] is None or previous[2] > current[0]
                    or previous[0] == current[0] or previous[1] >= current[1]):
                issue('LEDGER_VINTAGE_CONFLICT', current[3])
        for begin, release, finish, index, row in versions:
            if release <= cutoff and begin <= cutoff and (finish is None or cutoff < finish):
                selected.append(row)
    return diagnostics, selected


def review_ledger(ledger, policy, *, as_of):
    """Return bounded diagnostics without cell values, IO or input mutation.

    A future vintage is excluded, not an error. Any malformed, untrusted or
    unauthorized row (including future rows) blocks the entire snapshot.
    Caller proof declarations are not authenticated by this pure function.
    """
    diagnostics, selected = _inspect(ledger, policy, as_of)
    blocked = any(d['blocking'] for d in diagnostics)
    return dict(schema='societybox-ledger-review-v1', status='BLOCKED' if blocked else 'READY',
                assurance='SYNTHETIC_LEDGER_CONTRACT_ONLY', diagnostics=diagnostics,
                eligible_cell_count=0 if blocked else len(selected),
                claim_level='SIMULATED_RETROSPECTIVE_ANALYSIS',
                researcher_information_set='NOT_ESTABLISHED',
                source_authentication='NOT_ESTABLISHED',
                forecast='NOT_IMPLEMENTED', identify='NOT_IMPLEMENTED',
                causal_claim='NOT_IMPLEMENTED')


def as_of_snapshot(ledger, policy, *, as_of, vintage_target='AS_OF'):
    """Select only cutoff-eligible revisions; LATEST/FIRST_RELEASE are unsupported.

    Returned provenance is copied. No historical AS_OF knowledge/access claim
    is granted, even when a caller-supplied synthetic release proof matches.
    """
    result = review_ledger(ledger, policy, as_of=as_of)
    if vintage_target != 'AS_OF':
        result['status'] = 'BLOCKED'
        result['eligible_cell_count'] = 0
        result['diagnostics'].append(dict(code='LEDGER_VINTAGE_TARGET_UNSUPPORTED',
                                          row_index=None, blocking=True))
    result['schema'] = 'societybox-ledger-snapshot-v1'
    result['cells'] = []
    if result['status'] == 'READY':
        _, selected = _inspect(ledger, policy, as_of)
        result['cells'] = deepcopy(selected)
    return result
