"""Declared revision histories with explicit release/local availability cutoffs.

No provider authentication, downloads or historical information-set attestation.
The original synthetic mechanism_vintage contract remains unchanged. Its strict
timestamp/date/digest parsers are shared here; raw revision inputs travel inside
the approved engine study so a portable replay does not need a mutable dataset.
"""
from copy import deepcopy
from decimal import Decimal, localcontext
from fractions import Fraction
from hashlib import sha256
from html import escape
from pathlib import Path

from .mechanism_vintage import _instant, _date, _digest, _DECIMAL
from .workspace import canonical, digest
from .execution_control import checkpoint

_ROW = set(('series_id period_start period_end revision_id revision_sequence '
            'value_decimal source_sha256 source_release_at vendor_recorded_at local_acquired_at').split())
_META = {'unit', 'frequency', 'timezone', 'seasonal_adjustment', 'statistical_definition'}
_MODES = ('SOURCE_RELEASE_AS_OF', 'LOCAL_ACQUIRED_AS_OF')


def _code_digest():
    names = ('research_vintages.py', 'mechanism_vintage.py', 'workspace.py',
             'execution_control.py', 'research_evidence.py')
    return digest(canonical({name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                             for name in names}))


def _text(value, cap=512):
    if type(value) is not str or not value.strip() or len(value) > cap:
        raise ValueError('VINTAGE_TEXT_INVALID')


def _key(row):
    return row['series_id'], row['period_start'], row['period_end']


def _validate(q):
    if type(q) is not dict or set(q) != {'schema', 'ledger', 'earlier_as_of', 'later_as_of', 'availability_mode'}:
        raise ValueError('VINTAGE_COMPARISON_SCHEMA_INVALID')
    if q['schema'] != 'societybox-vintage-comparison-v1' or q['availability_mode'] not in _MODES:
        raise ValueError('VINTAGE_COMPARISON_MODE_INVALID')
    if _instant(q['earlier_as_of']) > _instant(q['later_as_of']):
        raise ValueError('VINTAGE_CUTOFF_ORDER_INVALID')
    ledger = q['ledger']
    if (type(ledger) is not dict or set(ledger) != {'schema', 'data_kind', 'source', 'rights', 'series', 'revisions'}
            or ledger['schema'] != 'societybox-revision-ledger-v1'
            or ledger['data_kind'] not in ('SYNTHETIC', 'RESEARCHER_SUPPLIED')):
        raise ValueError('VINTAGE_LEDGER_SCHEMA_INVALID')
    if ledger['rights'] != {'local_processing': 'PERMITTED', 'report_export': 'PERMITTED'}:
        raise ValueError('VINTAGE_RIGHTS_REQUIRED')
    source = ledger['source']
    if type(source) is not dict or set(source) != {'id', 'license', 'basis'}:
        raise ValueError('VINTAGE_SOURCE_DECLARATION_REQUIRED')
    _text(source['id'], 200); _text(source['basis'], 2000)
    if source['license'] not in ('CC0-1.0', 'CC-BY-4.0', 'OPERATOR_OWNED'):
        raise ValueError('VINTAGE_SOURCE_LICENSE_UNSUPPORTED')
    series = ledger['series']; rows = ledger['revisions']
    if type(series) is not dict or not 1 <= len(series) <= 1000:
        raise ValueError('VINTAGE_SERIES_REQUIRED')
    for name, meta in series.items():
        _text(name, 128)
        if type(meta) is not dict or set(meta) != _META:
            raise ValueError('VINTAGE_SERIES_METADATA_INVALID')
        for value in meta.values(): _text(value)
        if meta['frequency'] not in ('day', 'month', 'quarter', 'year', 'irregular'):
            raise ValueError('VINTAGE_FREQUENCY_INVALID')
    if type(rows) is not list or not 1 <= len(rows) <= 1000:
        raise ValueError('VINTAGE_REVISIONS_REQUIRED')
    groups = {}
    for row in rows:
        checkpoint()
        if type(row) is not dict or set(row) != _ROW:
            raise ValueError('VINTAGE_REVISION_FIELDS_INVALID')
        _text(row['series_id'], 128); _text(row['revision_id'], 200)
        if row['series_id'] not in series or _date(row['period_start']) > _date(row['period_end']):
            raise ValueError('VINTAGE_PERIOD_OR_SERIES_INVALID')
        if type(row['revision_sequence']) is not int or not 1 <= row['revision_sequence'] <= 1000000000:
            raise ValueError('VINTAGE_REVISION_SEQUENCE_INVALID')
        value = row['value_decimal']
        if type(value) is not str or len(value) > 128 or not _DECIMAL.fullmatch(value):
            raise ValueError('VINTAGE_DECIMAL_REQUIRED')
        if row['source_sha256'] is not None: _digest(row['source_sha256'])
        for field in ('source_release_at', 'vendor_recorded_at', 'local_acquired_at'):
            if row[field] is not None: _instant(row[field])
        groups.setdefault(_key(row), []).append(row)
    for versions in groups.values():
        versions.sort(key=lambda row: row['revision_sequence'])
        if (len({r['revision_id'] for r in versions}) != len(versions) or
                len({r['revision_sequence'] for r in versions}) != len(versions)):
            raise ValueError('VINTAGE_REVISION_CONFLICT')
        known = [_instant(r['source_release_at']) for r in versions if r['source_release_at'] is not None]
        if any(a >= b for a, b in zip(known, known[1:])):
            raise ValueError('VINTAGE_RELEASE_ORDER_CONFLICT')
    if len(canonical(q)) > 60000:
        raise ValueError('VINTAGE_INPUT_BUDGET')
    return groups


def validate(q):
    """Schema/time/rights review only: no snapshot selection or delta math."""
    _validate(q)
    return q


def _availability(row, cutoff, mode):
    gates = [row['source_release_at']]
    if mode == 'LOCAL_ACQUIRED_AS_OF': gates.append(row['local_acquired_at'])
    if any(value is not None and _instant(value) > cutoff for value in gates): return 'FUTURE'
    return 'UNKNOWN' if None in gates else 'ELIGIBLE'


def _snapshot(groups, as_of, mode):
    cutoff = _instant(as_of)
    cells, unresolved = [], []
    for key, versions in sorted(groups.items()):
        checkpoint()
        statuses = [_availability(row, cutoff, mode) for row in versions]
        eligible = [row for row, status in zip(versions, statuses) if status == 'ELIGIBLE']
        chosen = eligible[-1] if eligible else None
        # Only a potentially superseding unknown revision can change this
        # cell. Older unknown history does not invalidate a known later value.
        if any(status == 'UNKNOWN' and (chosen is None or row['revision_sequence'] > chosen['revision_sequence'])
               for row, status in zip(versions, statuses)):
            unresolved.append(dict(series_id=key[0], period_start=key[1], period_end=key[2],
                                   reason='REVISION_AVAILABILITY_UNKNOWN'))
        else:
            if chosen: cells.append(deepcopy(chosen))
    return dict(as_of=as_of, availability_mode=mode, cells=cells, unresolved=unresolved,
                assurance='DECLARED_TIMESTAMPS_ONLY', source_authentication='NOT_ESTABLISHED',
                historical_information_set='NOT_ESTABLISHED')


def _verify_selection(q, snapshot):
    """Separate unsorted scan chooses the greatest eligible release instant."""
    cutoff = _instant(snapshot['as_of']); mode = q['availability_mode']
    rows = q['ledger']['revisions']
    chosen = {_key(row): row for row in snapshot['cells']}
    unresolved = {_key(row) for row in snapshot['unresolved']}
    for key in {_key(row) for row in rows}:
        versions = [row for row in rows if _key(row) == key]
        eligible = [row for row in versions if row['source_release_at'] is not None
            and _instant(row['source_release_at']) <= cutoff and
            (mode != 'LOCAL_ACQUIRED_AS_OF' or (row['local_acquired_at'] is not None
             and _instant(row['local_acquired_at']) <= cutoff))]
        reference = max(eligible, key=lambda r: _instant(r['source_release_at'])) if eligible else None
        ambiguous = False
        for row in versions:
            instants = [row['source_release_at']]
            if mode == 'LOCAL_ACQUIRED_AS_OF': instants.append(row['local_acquired_at'])
            definitely_future = any(value is not None and _instant(value) > cutoff for value in instants)
            if (not definitely_future and None in instants and
                    (reference is None or row['revision_sequence'] > reference['revision_sequence'])):
                ambiguous = True
        if ambiguous: reference = None
        if (key in unresolved) != ambiguous or chosen.get(key) != reference:
            raise ValueError('VINTAGE_SELECTION_ORACLE_MISMATCH')


def calculate(q):
    groups = _validate(q)
    code = _code_digest()
    before = _snapshot(groups, q['earlier_as_of'], q['availability_mode'])
    after = _snapshot(groups, q['later_as_of'], q['availability_mode'])
    _verify_selection(q, before); _verify_selection(q, after)
    left = {_key(row): row for row in before['cells']}; right = {_key(row): row for row in after['cells']}
    unknown = {_key(row) for row in before['unresolved'] + after['unresolved']}
    changes = []
    for key in sorted(groups):
        checkpoint()
        a, b = left.get(key), right.get(key)
        delta = None
        if key in unknown: status = 'UNRESOLVED_AVAILABILITY'
        elif a is not None and b is not None:
            delta = str(Fraction(b['value_decimal']) - Fraction(a['value_decimal']))
            with localcontext() as context:
                context.prec = 512
                independently = Fraction(Decimal(b['value_decimal']) - Decimal(a['value_decimal']))
            if Fraction(delta) != independently: raise ValueError('VINTAGE_DELTA_ORACLE_MISMATCH')
            status = 'COMPARABLE'
        else: status = 'ADDED' if b is not None else 'REMOVED' if a is not None else 'NOT_AVAILABLE'
        changes.append(dict(series_id=key[0], period_start=key[1], period_end=key[2],
            unit=q['ledger']['series'][key[0]]['unit'], status=status,
            earlier_value_decimal=None if a is None else a['value_decimal'],
            later_value_decimal=None if b is None else b['value_decimal'], delta_exact=delta,
            earlier_revision_id=None if a is None else a['revision_id'],
            later_revision_id=None if b is None else b['revision_id']))
    from .research_evidence import make_evidence
    subject = digest(canonical(q))
    evidence = make_evidence(subject_sha256=subject,
        producer=dict(tool_id='societybox.vintage_comparison', version='1', code_sha256=code),
        operation='DECLARED_REVISION_SNAPSHOT_COMPARISON', authority='LOCAL_EXACT_ARITHMETIC',
        epistemic_state='DERIVED', proof_kind='REVISION_SELECTION_AND_DECIMAL_DIFFERENCE',
        scope=dict(availability_mode=q['availability_mode'], earlier_as_of=q['earlier_as_of'], later_as_of=q['later_as_of']),
        coverage=dict(status='PARTIAL', unknown_dimensions=['source_authenticity', 'timestamp_authenticity',
                      'archive_completeness', 'historical_researcher_access', 'data_truth_and_rights']),
        result=dict(comparable_cells=sum(row['status'] == 'COMPARABLE' for row in changes)),
        parent_evidence_ids=[], limitations=['All timestamps, values, source digests and rights are declarations',
            'No latest-value fallback for unresolved availability; missing cells are not zero',
            'Vendor-record time is retained as provenance and does not establish source release or local access'])
    if _code_digest() != code:
        raise ValueError('VINTAGE_CODE_CHANGED_DURING_RUN')
    return dict(schema='societybox-vintage-comparison-result-v1', status='CONDITIONAL_REVISION_COMPARISON',
        input_sha256=subject, ledger_sha256=digest(canonical(q['ledger'])),
        source_authentication='NOT_ESTABLISHED', historical_information_set='NOT_ESTABLISHED',
        native_execution='NOT_RUN', statistical_and_causal_validity='NOT_ESTABLISHED',
        exact_oracle='INDEPENDENT_RELEASE_SELECTION_AND_DECIMAL_DIFFERENCES',
        earlier_snapshot=before, later_snapshot=after, changes=changes, evidence=evidence)


def verify(q, result):
    if canonical(calculate(q)) != canonical(result):
        raise ValueError('VINTAGE_RECOMPUTATION_MISMATCH')
    return dict(status='RECOMPUTED', scope='SAME_CODE_AND_DECLARED_INPUT_ONLY')


def render_review_section(q, result, locale):
    words = {
        'en': ('Series / period', 'Earlier value', 'Later value', 'Exact difference', 'Unit', 'Availability',
               'Declared timestamps only. Source truth, archive completeness and historical researcher access are NOT_ESTABLISHED. Missing values are not zero.',
               'Unresolved availability; difference withheld'),
        'ko': ('계열 / 관측기간', '이전 값', '이후 값', '정확한 차이', '단위', '이용시점 상태',
               '시점은 연구자 선언입니다. 출처 진실성·개정 이력 완전성·당시 연구자 접근은 NOT_ESTABLISHED입니다. 누락 값을 0으로 바꾸지 않습니다.',
               '이용시점 불명확: 차이 계산 보류'),
        'zh-Hans': ('序列 / 观测期间', '较早数值', '较晚数值', '精确差异', '单位', '可用性状态',
               '时间戳仅为研究者声明。来源真实性、修订档案完整性和当时研究者访问权限均为 NOT_ESTABLISHED。缺失值不视为零。',
               '可用时点不明；暂不计算差异'),
        'ja': ('系列 / 観測期間', '前の値', '後の値', '正確な差分', '単位', '利用時点の状態',
               '時点は研究者の申告です。出典の真実性・改定履歴の完全性・当時の研究者のアクセスは NOT_ESTABLISHED です。欠測をゼロにしません。',
               '利用時点が不明なため差分計算を保留'),
    }[locale]
    statuses = dict(zip(('COMPARABLE', 'ADDED', 'REMOVED', 'NOT_AVAILABLE', 'UNRESOLVED_AVAILABILITY'), {
        'en': ('Comparable', 'Newly available', 'No longer selected', 'Unavailable at both cutoffs', 'Availability unresolved'),
        'ko': ('비교 가능', '새로 이용 가능', '이후 선택에서 제외', '두 시점 모두 이용 불가', '이용시점 불명확'),
        'zh-Hans': ('可比较', '新增可用值', '不再被选中', '两个时点均不可用', '可用时点不明'),
        'ja': ('比較可能', '新たに利用可能', '後の選択から除外', '両時点で利用不可', '利用時点不明'),
    }[locale]))
    rows = []
    for row in result['changes']:
        values = (row['series_id']+' / '+row['period_start']+' – '+row['period_end'],
                  row['earlier_value_decimal'], row['later_value_decimal'], row['delta_exact'], row['unit'],
                  statuses[row['status']]+' · '+row['status'])
        rows.append('<tr>'+''.join('<td>'+escape('—' if value is None else str(value))+'</td>' for value in values)+'</tr>')
    return ('<section><p>'+escape(words[6])+'</p><p>'+escape(q['availability_mode'])+' · '
        +escape(q['earlier_as_of'])+' → '+escape(q['later_as_of'])+'</p>'
        +'<div class="table-scroll"><table class="vintage-revision-values"><thead><tr>'
        +''.join('<th scope="col">'+escape(word)+'</th>' for word in words[:6])+'</tr></thead><tbody>'
        +''.join(rows)+'</tbody></table></div><p>'+escape(result['exact_oracle'])+'</p></section>')


def example_parameters():
    """Owned invented example; never mislabel this as an actual provider series."""
    revisions = []
    for sequence, value, release, acquired in (
            (1, '100.01', '2025-01-10T00:00:00Z', '2025-01-10T00:00:00Z'),
            (2, '110.03', '2025-02-10T00:00:00Z', '2025-02-15T00:00:00Z')):
        revisions.append(dict(series_id='output', period_start='2025-01-01', period_end='2025-01-01',
            revision_id='authored-revision-'+str(sequence), revision_sequence=sequence,
            value_decimal=value, source_sha256=None, source_release_at=release,
            vendor_recorded_at=release, local_acquired_at=acquired))
    return dict(schema='societybox-vintage-comparison-v1',
        earlier_as_of='2025-01-20T00:00:00Z', later_as_of='2025-02-12T00:00:00Z',
        availability_mode='SOURCE_RELEASE_AS_OF', ledger=dict(
            schema='societybox-revision-ledger-v1', data_kind='SYNTHETIC',
            source=dict(id='authored-vintage-example', license='CC0-1.0',
                        basis='Invented values and release dates; no actual provider or historical evidence.'),
            rights=dict(local_processing='PERMITTED', report_export='PERMITTED'),
            series={'output': dict(unit='declared index', frequency='day', timezone='UTC',
                seasonal_adjustment='UNADJUSTED', statistical_definition='Invented output index')},
            revisions=revisions))
