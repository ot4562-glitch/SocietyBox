"""Offline review of one pinned public WDI retrieval, never historical as-of data.

Pins identify the locally preserved acquisition, not authenticated publisher
signatures. No network, LLM calls, arbitrary models or code execution.
"""
import csv
from datetime import datetime, timezone
from decimal import Decimal, localcontext
import hashlib
import io
import json
from pathlib import Path

from .workspace import ScopeError

INDICATOR = 'NY.GDP.MKTP.CD'
YEARS = ('2019', '2020', '2021')
PINS = {
    'provenance.json': 'b3d5bba6721f8df76239228b6506488d0b4f65ad11438c107b4616c74001747a',
    'wdi-USA-NY.GDP.MKTP.CD-2019-2021.json': '1bea5a7151e819df0a2d927d7a9907f1106a3e34fb37c9169786c00f92474f67',
    'wdi-KOR-NY.GDP.MKTP.CD-2019-2021.json': '05d4fa1f32184bc0ad07b0ed5471b7e49f85e57d0e4dad9cb8ec06285713475b',
}


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('WDI_DUPLICATE_JSON_KEY')
        result[key] = value
    return result


def _decode(raw):
    if len(raw) > 64000:
        raise ValueError('WDI_SOURCE_SIZE_INVALID')
    def invalid(_):
        raise ValueError('WDI_NONFINITE')
    return json.loads(raw, parse_float=Decimal, parse_int=Decimal,
                      parse_constant=invalid, object_pairs_hook=_unique)


def parse_source(raw, country):
    """Validate six-cell study coordinates without a float round trip."""
    doc = _decode(raw)
    if not isinstance(doc, list) or len(doc) != 2:
        raise ValueError('WDI_SCHEMA_INVALID')
    meta, rows = doc
    if (not isinstance(meta, dict) or meta.get('sourceid') != '2' or
            any(type(meta.get(key)) is not Decimal or meta[key] != expected
                for key,expected in (('page',1),('pages',1),('total',3))) or
            not isinstance(rows, list) or len(rows) != 3):
        raise ValueError('WDI_INCOMPLETE_SOURCE')
    values = {}
    for row in rows:
        if (not isinstance(row, dict) or not isinstance(row.get('indicator'), dict) or
                row['indicator'].get('id') != INDICATOR or
                row['indicator'].get('value') != 'GDP (current US$)' or
                row.get('countryiso3code') != country or row.get('unit') != '' or
                row.get('obs_status') != '' or row.get('date') not in YEARS or
                row['date'] in values):
            raise ValueError('WDI_CELL_COORDINATES_INVALID')
        value = row.get('value')
        if not isinstance(value, Decimal) or not value.is_finite() or value <= 0:
            raise ValueError('WDI_CELL_VALUE_INVALID')
        values[row['date']] = value
    return values


def _instant(value):
    try:
        result = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if result.tzinfo is None or result.utcoffset() is None:
            raise ValueError
        return result.astimezone(timezone.utc)
    except (ValueError, TypeError, AttributeError):
        raise ScopeError('WDI_OFFSET_TIMESTAMP_REQUIRED') from None


def _read(directory, filename):
    target = directory / filename
    if target.is_symlink() or not target.is_file() or target.resolve().parent != directory:
        raise ScopeError('WDI_SOURCE_FILE_REQUIRED')
    with target.open('rb') as stream:
        raw = stream.read(64001)
    if len(raw) > 64000:
        raise ScopeError('WDI_SOURCE_SIZE_INVALID')
    if hashlib.sha256(raw).hexdigest() != PINS[filename]:
        raise ScopeError('WDI_HASH_MISMATCH')
    return raw


def review_snapshot(directory, cutoff):
    """Return exact current-retrieval observations and descriptive differences.

    A cutoff governs local snapshot availability, not observation years or the
    API lastupdated field. No historical publication date is inferred.
    """
    directory = Path(directory).resolve(strict=True)
    provenance_raw = _read(directory, 'provenance.json')
    provenance = json.loads(provenance_raw)
    retrieved = _instant(provenance['retrieved_utc'])
    limit = _instant(cutoff)
    if limit < retrieved:
        raise ScopeError('WDI_RETRIEVAL_AFTER_CUTOFF')
    if limit > datetime.now(timezone.utc):
        raise ScopeError('WDI_FUTURE_CUTOFF')
    cells, sources, changes = [], [], {}
    for item in provenance['observations']:
        raw = _read(directory, item['raw_file'])
        try:
            values = parse_source(raw, item['country'])
        except ValueError as error:
            raise ScopeError(str(error)) from None
        if ({year: str(values[year]) for year in YEARS} != item['parsed']['values'] or
                item['raw_sha256'] != PINS[item['raw_file']] or
                item['parsed']['unit'] != 'current_USD'):
            raise ScopeError('WDI_DECLARATION_SOURCE_MISMATCH')
        sources.append({'file': item['raw_file'], 'sha256': item['raw_sha256'],
                        'url': item['url'], 'api_lastupdated': item['parsed']['api_lastupdated']})
        for year in YEARS:
            cells.append({'country': item['country'], 'year': year,
                          'value_decimal': str(values[year]), 'unit': 'current_USD',
                          'source_sha256': item['raw_sha256']})
        with localcontext() as context:
            context.prec = 64
            changes[item['country']] = str(values['2021'] - values['2019'])
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(cells[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(cells)
    return {
        'schema': 'societybox-wdi-snapshot-review-v1', 'status': 'VERIFIED_PINNED_SNAPSHOT',
        'provenance_sha256': PINS['provenance.json'], 'sources': sources,
        'retrieved_utc': provenance['retrieved_utc'], 'cutoff': limit.isoformat(),
        'vintage': provenance['claims']['vintage'], 'missing_cells': [],
        'rights': {'declared_license': provenance['license']['name'],
                   'terms_url': provenance['license']['terms'],
                   'attribution': 'World Bank, World Development Indicators, NY.GDP.MKTP.CD; retrieved 2026-10-24',
                   'indicator_specific_clearance': 'NOT_ESTABLISHED',
                   'scope': provenance['license']['scope']},
        'evidence_context': {
            'numeric_values': 'PARSED_FROM_HASH_PINNED_API_BYTES',
            'unit': 'INDICATOR_ID_AND_API_LABEL_CURRENT_USD; raw unit field is empty',
            'rights_and_retrieval_time': 'LOCAL_PROVENANCE_DECLARATIONS',
            'historical_release_at': 'UNKNOWN',
            'formula': 'GDP(2021) - GDP(2019), current USD; nominal descriptive change only'},
        'cells': cells, 'csv': stream.getvalue(), 'change_2019_2021': changes,
        'publisher_authentication': 'NOT_ESTABLISHED',
        'independent_science': 'NOT_ESTABLISHED', 'historical_information_set': 'NOT_ESTABLISHED',
        'data_sent_to_LLM': False,
    }
