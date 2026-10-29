"""Exact paired trajectory cells from a verified portable bundle; no execution."""
import csv
import io

from .study_html import _report_target
from .study_runs import _verified_comparison_snapshot, put_file
from .workspace import digest


def comparison_trajectory_csv(folder, output):
    """Write deterministic long-form CSV, retaining decimal strings and lineage.

    This checks artifact consistency and declared export rights, not execution
    authenticity or numerical accuracy. No source datasets or private receipt
    context are accessed. Coordinates and interpolated values are not exported.
    """
    target = _report_target(folder, output, '.csv')
    assurance, comparison, records = _verified_comparison_snapshot(folder)
    blob, rows = _comparison_trajectory_bytes(assurance, comparison, records)
    put_file(target, blob)
    return {'file': str(target), 'sha256': digest(blob), 'rows': rows,
            'format': 'EXACT_RECORDED_TRAJECTORY_CSV',
            'comparison_id': assurance['comparison_id'], 'assurance': assurance,
            'numerical_accuracy': 'NOT_ESTABLISHED'}


def _comparison_trajectory_bytes(assurance, comparison, records):
    """Encode one already verified snapshot without disk reads or writes."""
    stream = io.StringIO(newline='')
    writer = csv.writer(stream, lineterminator='\n')
    writer.writerow(('role', 'variable', 'step', 'value', 'unit', 'time_unit',
                     'graph_node', 'run_id', 'claim_graph_sha256', 'comparison_id',
                     'artifact_consistency', 'engine_reexecution',
                     'independent_clean_rerun', 'numerical_accuracy'))
    rows = 0
    for role in ('baseline', 'alternative'):
        ref = comparison['lineage']['runs'][role]
        for variable in sorted(comparison['lineage']['final_cells']):
            unit = comparison['lineage']['final_cells'][variable]['unit']
            for point in records[role]['results']['trajectory']:
                writer.writerow((role, variable, point['step'], point['state'][variable],
                                 unit, comparison['lineage']['time_unit'],
                                 f"cell/{point['step']}/{variable}", ref['run_id'],
                                 ref['claim_graph_sha256'], assurance['comparison_id'],
                                 assurance['status'], assurance['engine_reexecution'],
                                 assurance['independent_clean_rerun'], 'NOT_ESTABLISHED'))
                rows += 1
    blob = stream.getvalue().encode('utf-8')
    return blob, rows
