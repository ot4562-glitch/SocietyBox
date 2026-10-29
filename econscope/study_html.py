"""Self-contained evidence-first reports from verified portable comparisons."""
from html import escape
import json
from pathlib import Path

from .study_charts import _chart_blob, _trajectory_geometry, _trajectory_svg
from .study_core import temporal_assurance, dataset_references
from .study_runs import _verified_comparison_snapshot, put_file
from .study_recompute import _inspect_receipt
from .workspace import ScopeError, digest


def comparison_report(folder, output, *, recomputation_receipt=None):
    """Check artifact consistency once and render without executing model code."""
    target = _report_target(folder, output, '.html')
    blob, assurance = _comparison_document(folder, recomputation_receipt)
    put_file(target, blob)
    return {'file': str(target), 'sha256': digest(blob), 'comparison_id': assurance['comparison_id'],
            'assurance': assurance}


def _report_target(folder, output, suffix):
    target = Path(output)
    if target.suffix.lower() != suffix:
        raise ScopeError('STUDY_REPORT_' + suffix[1:].upper() + '_PATH_REQUIRED')
    if Path(folder).resolve() in (target.resolve(), *target.resolve().parents):
        raise ScopeError('STUDY_REPORT_OUTPUT_INSIDE_BUNDLE')
    return target


def _comparison_document(folder, recomputation_receipt=None):
    """Build the shared evidence snapshot once, without writing any output."""
    snapshot = _verified_comparison_snapshot(folder)
    return _document_from_snapshot(snapshot, recomputation_receipt)


def _document_from_snapshot(snapshot, recomputation_receipt=None):
    """Render the shared immutable-in-memory snapshot without rereading a bundle."""
    assurance, comparison, records = snapshot
    local_record = (_inspect_receipt(snapshot, recomputation_receipt)
                    if recomputation_receipt is not None else None)
    return _document_with_record(snapshot, local_record)


def _document_with_record(snapshot, local_record):
    """Render a verified snapshot and its already inspected optional local record."""
    assurance, comparison, records = snapshot
    spec = records["baseline"]["spec"]

    def pre(value):
        return '<pre>' + escape(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)) + '</pre>'

    def section(ident, title, body):
        return f'<section id="{ident}"><h2>{title}</h2>{body}</section>'

    parts = ['<!doctype html><html lang="en"><head><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width, initial-scale=1">',
             '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; '
             'style-src \'unsafe-inline\'; base-uri \'none\'; form-action \'none\'">',
             '<title>Society Box conditional comparison</title><style>',
             'body{font-family:system-ui,sans-serif;max-width:1100px;margin:auto;padding:24px;'
             'color:#17212b;background:white;line-height:1.5}pre{white-space:pre-wrap;overflow-wrap:anywhere;'
             'background:#f1f5f9;padding:16px}table{border-collapse:collapse;width:100%}'
             'th,td{border:1px solid #94a3b8;padding:8px;text-align:left;overflow-wrap:anywhere}'
             'svg{width:100%;height:auto}section{margin:32px 0}.assurance{border:2px solid #64748b;padding:16px}'
             '@page{size:A4;margin:16mm}'
             '@media print{body{max-width:none;padding:0;font-size:10pt;line-height:1.35}'
             'section{break-inside:auto}h1,h2,h3,caption{break-after:avoid}'
             'pre{background:white;padding:0;white-space:pre-wrap;overflow-wrap:anywhere;'
             'word-break:break-word;break-inside:auto;font-size:8pt}'
             'table{table-layout:fixed;font-size:8pt}thead{display:table-header-group}'
             'tr,figure{break-inside:avoid}th,td{padding:4pt;word-break:break-word}'
             'figure{margin:8mm 0}svg{display:block;width:100%;height:auto}'
             'a{color:inherit;text-decoration:none}.assurance{padding:8pt}}',
             '</style></head><body><h1>' + escape(spec['title']) + '</h1>',
             '<p>' + escape(spec['question']) + '</p>',
             '<p class="assurance">Conditional model comparison. ARTIFACT_CONSISTENCY_PASS. '
             'Engine reexecution: NOT_RUN. Independent clean rerun: NOT_RUN. '
             'Original receipt authenticity and external validity: NOT_ESTABLISHED. '
             'A consistently rewritten bundle can pass these checks.</p>']
    if local_record is not None:
        outcome = local_record['recorded_outcome']
        notice = ('Recorded local attempt: ' + outcome + '. '
                  'UNAUTHENTICATED_LOCAL_RECORD. ')
        if outcome == 'RECOMPUTATION_NOT_RUN':
            notice += ('Recorded reason: ' + local_record['payload']['reason'] + '. '
                       'No successful engine rerun is recorded by this receipt. ')
        notice += ('This report checks artifact consistency and does not execute the engine. '
                   'The attached record is an unsigned declaration of a separate attempt.')
        parts.append('<p class="assurance">' + escape(notice) + '</p>')
        parts.append(section('local-recomputation', 'Separate local recomputation record',
            '<p>UNAUTHENTICATED_LOCAL_RECORD. Bundle, engine, recomputation code and runtime '
            'bindings match. Recorded outcomes below are unsigned declarations, not a new '
            'execution by this report. Anyone can rewrite and rehash this record. Source hashes '
            'refer to bytes checked at the recorded attempt; source files are not inspected '
            'again here. Independent replication and authenticated identity are not established.</p>'
            + pre(local_record)))
    parts.append(section('basis', '1. Source declarations and researcher assumptions',
                         '<p>All declarations below are NOT_INDEPENDENTLY_VERIFIED. '
                         'User assumptions are not observations. Export rights are researcher-declared, '
                         'not legally verified. Raw data and private execution-review context are excluded.</p>'
                         + pre({'inference': spec['inference'], 'evidence': spec['evidence']})))
    parts.append(section('methods', '2. Model definitions and timing',
                         '<p>Transitions use synchronous previous-step state; detectors inspect '
                         'post-transition state. Model steps do not imply historical calendar dates. '
                         'Only declared rules apply; no external or causal lag estimate is established.</p>'
                         + pre(spec['model'])))
    methods = []
    for role in ('baseline', 'alternative'):
        record = records[role]
        methods.append('<h3>' + role.title() + '</h3>' + pre({
            'references': comparison['lineage']['runs'][role],
            'engine': record['engine'], 'engine_code_sha256': record['engine_code_sha256'],
            'runtime': record['runtime'], 'scope': record['scope'],
            'capabilities': record['runtime'].get('capabilities', {'status': 'NOT_RECORDED'}),
            'numerical_assurance': record['runtime'].get('numerical_assurance', {'status': 'NOT_RECORDED'}),
            'numerical_diagnostics': record['results'].get('numerical_diagnostics', {'status': 'NOT_RECORDED'}),
            'overrides': record['overrides'], 'effective_parameters': record['results']['parameters'],
            'source_lineage': record['results']['source_lineage'],
            'publication_date_claims': {key: 'NOT_RECORDED_IN_EXPORTED_RUN'
                                        for key in sorted(dataset_references(record['spec']))},
            'temporal_assurance': temporal_assurance(record['scope'], dataset_references(record['spec'])),
            'research_assurance': record['research_assurance']}))
    parts.append(section('execution', '3. Projected execution records and effective parameters',
                         '<p>Recorded capabilities do not establish enforced network isolation or institutional authorization. '
                         'Missing historical capabilities are NOT_RECORDED. '
                         'These are exported projections; original receipts are not authenticated '
                         'by this report. Temporal assurance describes the projected execution scope, '
                         'not a new temporal check by this report. Original dataset publication-date '
                         'claims are NOT_RECORDED_IN_EXPORTED_RUN; recorded availability labels do '
                         'not recover those dates. Any publication dates in the separate local '
                         'recomputation record are declarations for that attempt and do not establish '
                         'the original claims. Publication dates are not authenticated; '
                         'row information sets, revision vintages and observation '
                         'dates are unsupported. Leakage-free prediction is not established. '
                         'Numerical assurance is a recorded fixed execution policy, not per-run error '
                         'measurement. Observed Decimal diagnostics record whole-calculation sticky flags '
                         'across initialization, transitions and detectors, not error magnitudes or operation counts. '
                         'Rounded arithmetic does not establish numerical accuracy; '
                         'convergence, stability and error bounds are unsupported. '
                         'Run and graph references identify the cells below.</p>' + ''.join(methods)))
    rows = []
    figures = []
    for index, (variable, cell) in enumerate(sorted(comparison['lineage']['final_cells'].items())):
        ident = f'cell-{index}'
        values = [variable, cell['unit'], cell['step'], cell['baseline'], cell['alternative'],
                  cell['delta'], cell['graph_node']]
        rows.append(f'<tr id="{ident}">' + ''.join('<td>' + escape(str(v)) + '</td>' for v in values) + '</tr>')
        # Generated SVG has only fixed elements/attributes and XML-escaped text.
        # Prefix IDs so multiple accessible figures have unique references.
        svg = _chart_blob(assurance, comparison, variable).decode('utf-8')
        svg = svg.replace('id="title"', f'id="{ident}-title"').replace(
            'id="description"', f'id="{ident}-description"').replace(
            'aria-labelledby="title description"', f'aria-labelledby="{ident}-title {ident}-description"')
        figures.append('<figure>' + svg + f'<figcaption><a href="#{ident}">Exact table cell: '
                       + escape(variable) + '</a>. Both graph hashes and run IDs appear in figure metadata '
                       'and the execution records above.</figcaption></figure>')
        geometry = _trajectory_geometry(comparison, records, variable)
        figures.append('<figure>' + _trajectory_svg(geometry, f'trajectory-{index}')
                       + '<figcaption>Recorded conditional model steps; straight segments only join '
                       'recorded cells, not continuous-time estimates. Coordinates rounded to 0.001 '
                       'canvas units; this is not a numerical error bound. Exact cells and graph '
                       'references follow.</figcaption></figure>' + pre(geometry))
    headings = ['Variable', 'Unit', 'Model step', 'Baseline', 'Alternative', 'Alternative minus baseline',
                'Node in both ClaimGraphs']
    table = '<table><caption>Exact final model cells</caption><thead><tr>' + ''.join(
        '<th scope="col">' + h + '</th>' for h in headings) + '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table>'
    parts.append(section('results', '4. Computed conditional contrast',
                         '<p>Under the declarations and rules above, the projected results contain '
                         'these final model values. Deltas are exact alternative minus baseline; '
                         'no empirical effect or causal identification is established.</p>' + table + ''.join(figures)))
    parts.append(section('events', '5. Recorded model events',
                         '<p>Only recorded detector observations are listed. Empty lists mean no recorded events. '
                         'These are conditional model observations, not historical events.</p>'
                         + pre({role: records[role]['results']['events'] for role in ('baseline', 'alternative')})))
    parts.append(section('cell-lineage', '6. Exact numeric and ClaimGraph reference appendix',
                         '<p>Searchable text references for every table and figure, including exact decimal '
                         'strings, units, model time, run IDs and both graph hashes. These references '
                         'identify exported claims; they do not authenticate execution. The HTML supports '
                         'browser printing with repeating table headers and flowing evidence blocks. '
                         'PDF generation and rendered pagination/typography are NOT_RUN.</p>'
                         + pre({'comparison_id': assurance['comparison_id'],
                                'time_unit': comparison['lineage']['time_unit'],
                                'runs': comparison['lineage']['runs'],
                                'final_cells': comparison['lineage']['final_cells']})))
    parts.append(section('limits', '7. Limitations and reproduction', pre(spec['limitations'])
                         + '<p>Independent clean-environment reproduction is NOT_RUN. Obtain the matching engine '
                         'and legally available data, initialize a fresh workspace, review and explicitly '
                         'approve each parameter configuration, execute both and compare numeric outputs. '
                         'No environment image or source data is embedded. Original IDs depend on private '
                         'context and need not match. PDF and rendered-page QA are separate acceptance gates.</p>'
                         + pre(assurance)))
    parts.append('</body></html>\n')
    blob = ''.join(parts).encode('utf-8')
    return blob, assurance
