"""Local WDI descriptive report bundle; no network, models or implicit execution."""
import csv
from decimal import Context, Decimal, localcontext
from html import escape
import io
from pathlib import Path
import xml.etree.ElementTree as ET

from .study_recompute import compute_wdi_replication
from .study_runs import put_file
from .workspace import ScopeError, canonical, digest


def _geometry(rows, country):
    """Signed difference axes; all exact source cells accompany rounded points."""
    series = {}
    for role, column in (('baseline', 'annual_change'), ('alternative', 'change_since_2019')):
        series[role] = [{'year': r['year'], 'value': r[column], 'column': column,
                         'source_sha256': r['source_sha256']}
                        for r in rows if r['country'] == country and r[column] != 'NOT_APPLICABLE']
    scale = max(Decimal(p['value']).copy_abs() for points in series.values() for p in points) or Decimal(1)
    with localcontext(Context(prec=40)):
        for points in series.values():
            for point in points:
                point['x'] = format(Decimal(60) + (Decimal(point['year']) - 2019) * 210, '.3f')
                point['y'] = format(Decimal(200) - Decimal(point['value']) / scale * 120, '.3f')
    return {'variable': country + ' nominal GDP changes', 'unit': 'current_USD',
            'time_unit': 'calendar year', 'scale': str(scale), 'last_step': 2021,
            'series': series, 'calendar_ticks': [[60, '2019'], [270, '2020'], [480, '2021']],
            'geometry': 'ROUNDED_TO_0.001_CANVAS_UNITS_NOT_ERROR_BOUNDS',
            'title': 'Observed descriptive differences: ',
            'time_label': 'Observation time: ',
            'sample_notice': 'Annual observed cells only; no estimates between years.',
            'legend': 'Solid gray: annual change. Dashed blue: change since 2019.',
            'execution_notice': 'Separate Fraction arithmetic: PASS; external replication NOT_ESTABLISHED.'}


def _svg(geometry, report):
    root = ET.Element('svg', xmlns='http://www.w3.org/2000/svg', viewBox='0 0 640 560', role='img')
    ET.SubElement(root, 'title').text = geometry['title'] + geometry['variable']
    ET.SubElement(root, 'metadata').text = canonical(geometry).decode()
    ET.SubElement(root, 'desc').text = canonical(report).decode()
    labels = [(20, 25, geometry['variable'] + ' | current_USD'),
              (20, 48, geometry['legend']), (10, 80, geometry['scale']),
              (10, 200, '0'), (10, 320, '-' + geometry['scale'])]
    labels.extend((x, 345, label) for x, label in geometry['calendar_ticks'])
    labels.extend((20, 370 + 16 * i, notice) for i, notice in enumerate(report['disclosures']))
    for x, y, label in labels:
        ET.SubElement(root, 'text', x=str(x), y=str(y), **{'font-size': '10'}).text = label
    ET.SubElement(root, 'path', d='M60 80V320H480 M60 200H480', fill='none', stroke='#64748b')
    for role, color in (('baseline', '#475569'), ('alternative', '#0369a1')):
        points = geometry['series'][role]
        attrs = {'points': ' '.join(p['x'] + ',' + p['y'] for p in points),
                 'fill': 'none', 'stroke': color}
        if role == 'alternative':
            attrs['stroke-dasharray'] = '5 3'
        ET.SubElement(root, 'polyline', attrs)
        for point in points:
            circle = ET.SubElement(root, 'circle', cx=point['x'], cy=point['y'], r='3', fill=color)
            ET.SubElement(circle, 'title').text = point['year'] + ': ' + point['value'] + ' current_USD'
    return ET.tostring(root, encoding='utf-8')


def export_wdi_replication(directory, cutoff, output):
    """Render before writing; exclusive directory, manifest committed last.

    A failed filesystem write may leave an incomplete new directory, never a
    success manifest. Existing researcher files are never replaced or removed.
    """
    from .study_pdf import wdi_replication_pdf

    target = Path(output).absolute()
    if target.exists() or target.is_symlink():
        raise ScopeError('WDI_OUTPUT_EXISTS')
    report = compute_wdi_replication(directory, cutoff)
    report['figures'] = {country: _geometry(report['rows'], country) for country in ('USA', 'KOR')}
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(report['rows'][0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(report['rows'])
    blobs = {'report.json': canonical(report) + b'\n', 'table.csv': stream.getvalue().encode()}
    for country, geometry in report['figures'].items():
        blobs[country + '.svg'] = _svg(geometry, report)
    heads = ''.join('<th>' + escape(k) + '</th>' for k in report['rows'][0])
    table = ''.join('<tr>' + ''.join('<td>' + escape(v) + '</td>' for v in row.values()) + '</tr>'
                    for row in report['rows'])
    html = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>WDI descriptive replication</title>'
            '<h1>World Bank WDI GDP: descriptive replication</h1><pre>' +
            escape(canonical({k: v for k, v in report.items() if k not in ('rows', 'figures')}).decode()) +
            '</pre><table><thead><tr>' + heads + '</tr></thead><tbody>' + table + '</tbody></table>' +
            ''.join(blobs[c + '.svg'].decode() for c in ('USA', 'KOR')) + '</html>')
    blobs['report.html'] = html.encode()
    blobs['report.pdf'], pdf_details = wdi_replication_pdf(report)
    manifest = {'schema': 'societybox-wdi-artifacts-v1', 'status': report['status'],
                'artifacts': {name: digest(raw) for name, raw in sorted(blobs.items())},
                'pdf': pdf_details, 'receipt_authenticity': 'NOT_ESTABLISHED'}
    try:
        target.mkdir()  # Exclusive, does not create parents or follow an existing symlink.
    except FileExistsError:
        raise ScopeError('WDI_OUTPUT_EXISTS') from None
    for name, raw in blobs.items():
        put_file(target / name, raw)
    put_file(target / 'manifest.json', canonical(manifest) + b'\n')
    return manifest
