"""Offline ASCII PDF with vector trajectories; no external renderers or execution."""
from html.parser import HTMLParser
import textwrap

from .study_html import _document_with_record, _report_target
from .study_recompute import _inspect_receipt
from .study_charts import _trajectory_geometry
from .study_runs import _verified_comparison_snapshot, put_file
from .workspace import ScopeError, digest


class _ReportText(HTMLParser):
    """Retain evidence, tables and exact appendix; omit HTML styles and SVGs."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.skipped = 0
        self.parts = []
        self.blocks = []
        self.preformatted = False

    def flush(self):
        if self.parts:
            self.blocks.append((''.join(self.parts), self.preformatted))
            self.parts = []

    def close(self):
        super().close()
        self.flush()

    def handle_starttag(self, tag, attrs):
        if tag == 'pre' and not self.skipped:
            self.flush()
            self.preformatted = True
        if tag in ('head', 'svg', 'figure'):
            self.skipped += 1
        if not self.skipped and tag in ('h1', 'h2', 'h3', 'p', 'pre', 'tr', 'caption', 'figure'):
            self.parts.append('\n\n')

    def handle_endtag(self, tag):
        if tag in ('head', 'svg', 'figure'):
            self.skipped -= 1
        elif not self.skipped and tag in ('h1', 'h2', 'h3', 'p', 'pre', 'tr', 'caption', 'figure', 'td', 'th'):
            self.parts.append('\n')
        if tag == 'pre' and not self.skipped:
            self.flush()
            self.preformatted = False

    def handle_data(self, data):
        if not self.skipped:
            self.parts.append(data)


def _literal(line):
    return line.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')


def _figure_commands(geometry):
    """Use precisely the shared SVG coordinates, reflecting the PDF y axis."""
    labels = [geometry.get('title', 'Conditional model trajectory: ') + geometry['variable'],
              'Unit: ' + geometry['unit'] + ' | ' + geometry.get('time_label', 'Model time: ') + geometry['time_unit'],
              geometry.get('legend', 'Baseline: solid gray. Alternative: dashed blue.'),
              geometry.get('sample_notice', 'Recorded cells only; segments do not estimate between model steps.'),
              'Geometry rounded to 0.001 canvas units; not numerical error bounds.',
              'Exact cells, graph references and observed precision: report text.',
              geometry.get('execution_notice', 'Engine reexecution / independent clean rerun: NOT_RUN.'),
              'Numerical accuracy / external validity: NOT_ESTABLISHED.']
    commands = ['BT /F1 9 Tf 14 TL 36 800 Td']
    for label in labels:
        for line in textwrap.wrap(label, 96, break_on_hyphens=False):
            commands.append(f'({_literal(line)}) Tj T*')
    commands.extend(['ET', 'q 1 0 0 -1 0 650 cm',
                     '0.4 G 0.5 w 60 80 m 60 320 l 480 320 l S 60 200 m 480 200 l S'])
    for role, color, dash in (('baseline', '0.28 0.33 0.41', '[]'),
                               ('alternative', '0.01 0.41 0.63', '[5 3]')):
        commands.append(f'{color} RG {dash} 0 d 1.5 w')
        for index, row in enumerate(geometry['series'][role]):
            commands.append(f"{row['x']} {row['y']} {'m' if index == 0 else 'l'}")
        commands.append('S')
    commands.append('Q')
    ticks = geometry.get('calendar_ticks', ((60, '0'), (460, str(geometry['last_step']))))
    for x, y, label in (*((x, 300, label) for x, label in ticks),
                        (10, 570, geometry['scale']), (10, 450, '0'),
                        (10, 330, '-' + geometry['scale'])):
        # Axis labels with unusually long exact decimals wrap instead of clipping.
        for index, line in enumerate(textwrap.wrap(label, 90)):
            commands.append(f'BT /F1 8 Tf {x} {y - index * 10} Td ({_literal(line)}) Tj ET')
    return commands


def wdi_replication_pdf(report):
    """ASCII exact rows plus shared descriptive vector geometry; no CJK claim."""
    from .workspace import canonical
    lines = ['World Bank WDI GDP: descriptive replication',
             'Unit: current_USD. Annual change and change since 2019.',
             'PDF_GENERATED. CJK rendering: NOT_RUN. Rendered-page QA: NOT_RUN.',
             'Exact JSON appendix lines are compressed; readable operand tables precede them.']
    lines.extend(report['disclosures'])
    for key, value in report['analyses'].items():
        lines.append(key + ': ' + value)
    # Preserve exact numeric strings and source IDs without inserted line breaks.
    # These use the existing PDF backend's horizontal-fit contract.
    for source in report['sources']:
        for key, value in source.items():
            lines.extend(textwrap.wrap(key + ': ' + value, 96, break_on_hyphens=False))
    lines.append('Provenance SHA256: ' + report['provenance_sha256'])
    lines.append('Retrieved UTC: ' + report['retrieved_utc'])
    lines.append('Cutoff: ' + report['cutoff'])
    lines.extend(textwrap.wrap('Rights: ' + canonical(report['rights']).decode(), 96,
                               break_on_hyphens=False))
    for row in report['rows']:
        lines.append('')
        lines.append(row['country'] + ' / ' + row['year'] + ' / current_USD')
        for key, value in row.items():
            if key != 'disclosures':  # Already printed verbatim above.
                lines.append(key + ': ' + value)
        # Machine-readable appendix, alongside the readable operands above.
        lines.append(canonical(row).decode())
    if any(ord(c) > 126 or ord(c) < 32 for line in lines for c in line):
        raise ScopeError('STUDY_PDF_UNSUPPORTED_TEXT_USE_HTML')
    blob, pages = _pdf_bytes(lines, report['figures'].values())
    return blob, {'pages': pages, 'format': 'ASCII_VECTOR_PDF',
                  'cjk_rendering': 'NOT_RUN', 'rendered_page_qa': 'NOT_RUN'}


def _pdf_bytes(lines, figures=()):
    """Minimal PDF 1.4: A4 pages, built-in Courier, uncompressed literal text."""
    pages = []
    for i in range(0, len(lines), 64):
        commands = ['BT /F1 9 Tf 11 TL 36 800 Td']
        for line in lines[i:i + 64]:
            # Courier is 0.6 em per ASCII character. Fit long verbatim JSON
            # lines in the same 96-column width without inserting characters
            # inside decimal strings, references or declarations. Integer math
            # keeps layout independent of the caller's Decimal context.
            scale = min(100000, 9600000 // max(1, len(line)))
            commands.append(f'{scale // 1000}.{scale % 1000:03d} Tz '
                            f'({_literal(line)}) Tj T*')
        pages.append(commands + ['ET'])
    pages.extend(_figure_commands(geometry) for geometry in figures)
    objects = [b'<< /Type /Catalog /Pages 2 0 R >>', b'',
               b'<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>']
    kids = []
    for index, page in enumerate(pages):
        page_id = len(objects) + 1
        kids.append(f'{page_id} 0 R')
        objects.append((f'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] '
                        f'/Resources << /Font << /F1 3 0 R >> >> /Contents {page_id + 1} 0 R >>').encode('ascii'))
        commands = page + [f'BT /F1 8 Tf 36 25 Td (Page {index + 1} of {len(pages)}) Tj ET']
        stream = '\n'.join(commands).encode('ascii')
        objects.append(f'<< /Length {len(stream)} >>\nstream\n'.encode('ascii') + stream + b'\nendstream')
    objects[1] = f'<< /Type /Pages /Kids [{" ".join(kids)}] /Count {len(pages)} >>'.encode('ascii')
    blob = bytearray(b'%PDF-1.4\n%\xe2\xe3\xcf\xd3\n')
    offsets = [0]
    for ident, obj in enumerate(objects, 1):
        offsets.append(len(blob))
        blob.extend(f'{ident} 0 obj\n'.encode('ascii') + obj + b'\nendobj\n')
    xref = len(blob)
    blob.extend(f'xref\n0 {len(offsets)}\n0000000000 65535 f \n'.encode('ascii'))
    for offset in offsets[1:]:
        blob.extend(f'{offset:010d} 00000 n \n'.encode('ascii'))
    blob.extend((f'trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\n'
                 f'startxref\n{xref}\n%%EOF\n').encode('ascii'))
    return bytes(blob), len(pages)


def comparison_pdf(folder, output, *, recomputation_receipt=None):
    """Export the verified HTML's evidence/methods/cells as searchable text PDF.

    This deliberately limited backend rejects non-ASCII and control characters
    before writing. It never substitutes glyphs or embeds researcher programs.
    """
    target = _report_target(folder, output, '.pdf')
    snapshot = _verified_comparison_snapshot(folder)
    local_record = (_inspect_receipt(snapshot, recomputation_receipt)
                    if recomputation_receipt is not None else None)
    blob, details = _pdf_with_record(snapshot, local_record)
    put_file(target, blob)
    return {'file': str(target), **details}


def _pdf_with_record(snapshot, local_record):
    """Render an already verified snapshot/record without rereads or file writes."""
    html, assurance = _document_with_record(snapshot, local_record)
    _, comparison, records = snapshot
    figures = [_trajectory_geometry(comparison, records, variable)
               for variable in sorted(comparison['lineage']['final_cells'])]
    parser = _ReportText()
    parser.feed(html.decode('utf-8'))
    parser.close()
    notice = ('Society Box - ASCII PDF with vector trajectories\n'
              'PDF_GENERATED. Rendered-page QA: NOT_RUN. Multilingual fonts: UNSUPPORTED.\n'
              'Text/tables/lineage and vector trajectories; final SVG bars are omitted.\n'
              'Straight segments join recorded cells only. Prose may wrap.\n'
              'Exact JSON lines are horizontally compressed to fit; readability QA: NOT_RUN.\n'
              'This PDF uses the same verified evidence/methods/cells as the HTML report.\n\n')
    text = notice + ''.join(block for block, _ in parser.blocks)
    if any(ord(char) > 126 or (ord(char) < 32 and char not in '\n\r\t') for char in text):
        raise ScopeError('STUDY_PDF_UNSUPPORTED_TEXT_USE_HTML')
    lines = []
    for block, verbatim in [(notice, False), *parser.blocks]:
        if not verbatim:
            # Change only report prose, never a researcher's JSON declaration.
            block = block.replace('PDF generation and rendered pagination/typography are NOT_RUN.',
                                  'HTML browser-print pagination/typography are NOT_RUN.')
        for line in block.splitlines():
            lines.extend([line] if verbatim else
                         textwrap.wrap(line, width=96, expand_tabs=True, replace_whitespace=True,
                                       break_long_words=True, break_on_hyphens=False) or [''])
    blob, pages = _pdf_bytes(lines, figures)
    return blob, {'sha256': digest(blob), 'comparison_id': assurance['comparison_id'],
            'assurance': assurance, 'pages': pages, 'format': 'ASCII_VECTOR_PDF',
            'trajectory_figures': len(figures),
            'rendered_page_qa': 'NOT_RUN', 'multilingual_fonts': 'UNSUPPORTED'}
