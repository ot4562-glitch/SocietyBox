"""Offline vector figures from a single verified paired-bundle snapshot."""
from decimal import Context, Decimal, localcontext
from pathlib import Path
import xml.etree.ElementTree as ET

from .study_runs import _verified_comparison_snapshot, put_file
from .study_core import numerical_context
from .workspace import ScopeError, canonical, digest

SVG = "http://www.w3.org/2000/svg"


def _trajectory_geometry(comparison, records, variable):
    """Shared HTML/PDF point geometry from an already verified snapshot.

    Coordinates use a 540 x 400 top-origin canvas. No smoothing, interpolation
    or downsampling: every recorded model step has one exact source cell.
    """
    cell = comparison['lineage']['final_cells'][variable]
    series = {}
    for role in ('baseline', 'alternative'):
        series[role] = [{'step': row['step'], 'value': row['state'][variable],
                         'graph_node': f"cell/{row['step']}/{variable}"}
                        for row in records[role]['results']['trajectory']]
    values = [Decimal(row['value']) for rows in series.values() for row in rows]
    if not all(value.is_finite() for value in values):
        raise ScopeError('STUDY_CHART_NONFINITE_VALUE')
    scale = max(value.copy_abs() for value in values) or Decimal(1)
    last_step = max(row['step'] for rows in series.values() for row in rows)
    with localcontext(Context(prec=40)):
        for rows in series.values():
            for row in rows:
                row['x'] = format(Decimal(60) + Decimal(row['step']) / (last_step or 1) * 420, '.3f')
                row['y'] = format(Decimal(200) - Decimal(row['value']) / scale * 120, '.3f')
    return {'schema': 'societybox-trajectory-figure-v1', 'variable': variable,
            'unit': cell['unit'], 'time_unit': comparison['lineage']['time_unit'],
            'scale': str(scale), 'last_step': last_step, 'series': series,
            'runs': comparison['lineage']['runs'],
            'numeric_labels': 'EXACT_RECORDED_DECIMAL_STRINGS',
            'geometry': 'ROUNDED_TO_0.001_CANVAS_UNITS_NOT_ERROR_BOUNDS'}


def _trajectory_svg(geometry, ident, *, labels=None):
    title, unit, time, legend = labels or (
        'Conditional model trajectory: ', 'Unit: ', 'Model time: ',
        'Baseline: solid gray. Alternative: dashed blue.')
    root = ET.Element('svg', {'xmlns': SVG, 'viewBox': '0 0 540 400',
                             'role': 'img', 'aria-labelledby': ident})
    ET.SubElement(root, 'title', id=ident).text = title + geometry['variable']
    ET.SubElement(root, 'metadata').text = canonical(geometry).decode('utf-8')
    for x, y, label in ((20, 25, unit + geometry['unit']),
                        (20, 50, time + geometry['time_unit']),
                        (60, 345, '0'), (460, 345, str(geometry['last_step'])),
                        (10, 80, geometry['scale']), (10, 200, '0'),
                        (10, 320, '-' + geometry['scale']),
                        (20, 375, legend)):
        ET.SubElement(root, 'text', x=str(x), y=str(y), **{'font-size': '11'}).text = label
    ET.SubElement(root, 'path', d='M60 80V320H480 M60 200H480', fill='none', stroke='#64748b')
    for role, color in (('baseline', '#475569'), ('alternative', '#0369a1')):
        attrs = {'points': ' '.join(row['x'] + ',' + row['y'] for row in geometry['series'][role]),
                 'fill': 'none', 'stroke': color, 'data-role': role}
        if role == 'alternative':
            attrs['stroke-dasharray'] = '5 3'
        ET.SubElement(root, 'polyline', attrs)
    return ET.tostring(root, encoding='unicode')


def comparison_chart(folder, variable, output):
    """Write one deterministic figure; no engine execution or external assets.

    Each figure has its own declared unit and scale. Decimal strings remain
    exact in labels/metadata; only SVG pixel coordinates are rounded.
    """
    assurance, comparison, _ = _verified_comparison_snapshot(folder)
    cells = comparison["lineage"]["final_cells"]
    if variable not in cells:
        raise ScopeError("STUDY_CHART_VARIABLE_UNKNOWN")
    target = Path(output)
    if target.suffix.lower() != ".svg":
        raise ScopeError("STUDY_CHART_SVG_PATH_REQUIRED")
    if Path(folder).resolve() in (target.resolve(), *target.resolve().parents):
        raise ScopeError("STUDY_CHART_OUTPUT_INSIDE_BUNDLE")
    blob = _chart_blob(assurance, comparison, variable)
    put_file(target, blob)
    return {"file": str(target), "sha256": digest(blob), "variable": variable,
            "comparison_id": assurance["comparison_id"], "assurance": assurance}


def _chart_blob(assurance, comparison, variable):
    """Render only trusted snapshot values; callers own validation and output."""
    cell = comparison["lineage"]["final_cells"][variable]
    values = [Decimal(cell[key]) for key in ("baseline", "alternative", "delta")]
    if not all(value.is_finite() for value in values):
        raise ScopeError("STUDY_CHART_NONFINITE_VALUE")
    # Notebook precision, exponent bounds, traps and rounding must not alter
    # exported bytes. The geometry is approximate, labels are exact.
    with localcontext(numerical_context()) as context:
        context.prec = 40
        scale = max(value.copy_abs() for value in values)
        if scale == 0:
            scale = Decimal(1)
        offsets = [format(value / scale * 220, ".3f") for value in values]
        scale_label = str(scale)
    root = ET.Element("svg", {"xmlns": SVG, "viewBox": "0 0 1100 610",
                              "role": "img", "aria-labelledby": "title description"})
    ET.SubElement(root, "title", id="title").text = f"Conditional model contrast: {variable}"
    ET.SubElement(root, "desc", id="description").text = (
        f"Final model {comparison['lineage']['time_unit']} {cell['step']}; unit {cell['unit']}. "
        f"Baseline {cell['baseline']}, alternative {cell['alternative']}, "
        f"alternative minus baseline {cell['delta']}. Artifact consistency only; "
        "engine reexecution and independent clean rerun NOT_RUN. "
        "Original receipt authenticity and external validity NOT_ESTABLISHED.")
    metadata = {"schema": "societybox-comparison-chart-v1", "variable": variable,
                "comparison_id": assurance["comparison_id"], "cell": cell,
                "runs": comparison["lineage"]["runs"],
                "time_unit": comparison["lineage"]["time_unit"], "assurance": assurance,
                "numeric_labels": "EXACT_DECIMAL_STRINGS", "geometry": "ROUNDED_PIXEL_COORDINATES"}
    ET.SubElement(root, "metadata").text = canonical(metadata).decode("utf-8")
    ET.SubElement(root, "rect", width="1100", height="610", fill="#ffffff")

    def text(x, y, content, size="15", **attrs):
        ET.SubElement(root, "text", x=str(x), y=str(y), fill="#17212b",
                      **{"font-family": "sans-serif", "font-size": size, **attrs}).text = content

    text(28, 30, "Basis: study.json evidence and researcher assumptions; inspect both ClaimGraphs first.")
    text(28, 59, f"Conditional model contrast: {variable}", "23")
    text(28, 86, f"Unit: {cell['unit']} | Model time: {cell['step']} {comparison['lineage']['time_unit']}")
    text(28, 112, "Computed model values; no empirical or causal interpretation is established.")
    ET.SubElement(root, "line", x1="510", x2="510", y1="145", y2="335", stroke="#64748b")
    text(510, 355, "0", "13", **{"text-anchor": "middle"})
    text(290, 355, "-" + scale_label, "13", **{"text-anchor": "middle"})
    text(730, 355, scale_label, "13", **{"text-anchor": "middle"})
    for index, (key, label, color) in enumerate((
            ("baseline", "Baseline", "#475569"),
            ("alternative", "Alternative", "#0369a1"),
            ("delta", "Alternative minus baseline", "#7c3aed"))):
        y = 153 + index * 65
        offset = Decimal(offsets[index])
        with localcontext(numerical_context()) as context:
            context.prec = 40
            x = format(Decimal(510) + min(Decimal(0), offset), ".3f")
        text(28, y + 23, label)
        ET.SubElement(root, "rect", {"x": x, "y": str(y), "width": format(offset.copy_abs(), ".3f"),
                                    "height": "32", "fill": color, "data-role": key,
                                    "data-value": cell[key]})
        fit = {"textLength": "300", "lengthAdjust": "spacingAndGlyphs"} if len(cell[key]) > 28 else {}
        text(765, y + 23, cell[key], **fit)
    text(28, 388, "Exact decimal labels; shared symmetric scale. Pixel coordinates are rounded.", "13")
    text(28, 413, f"Cell in both graphs: {cell['graph_node']}", "13")
    for index, role in enumerate(("baseline", "alternative")):
        ref = comparison["lineage"]["runs"][role]
        text(28, 439 + index * 40, f"{role} run: {ref['run_id']}", "12")
        text(28, 455 + index * 40, f"{role} graph SHA256: {ref['claim_graph_sha256']}", "12")
    text(28, 528, "ARTIFACT_CONSISTENCY_PASS | Engine reexecution: NOT_RUN | Independent clean rerun: NOT_RUN", "13")
    text(28, 552, "Receipt authenticity: NOT_ESTABLISHED | External validity: NOT_ESTABLISHED", "13")
    text(28, 578, "Comparison SHA256: " + assurance["comparison_id"], "12")
    return ET.tostring(root, encoding="utf-8") + b"\n"
