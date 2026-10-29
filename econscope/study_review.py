"""Read-only model/initializer inspection; never executes models or grants approval."""
from __future__ import annotations

from copy import deepcopy
import csv
from decimal import DecimalException, localcontext

from .study_runs import diagnose, execution_review
from .study_core import (aggregate_from_dataset, dataset_references, temporal_assurance,
                         numerical_assurance, numerical_context, value_in_bounds, number)
from .study_capabilities import negotiate_capabilities
from .workspace import ScopeError, canonical, digest


def _literal_initializer_readiness(spec):
    """Check known scalar bounds without reading data or evaluating a model."""
    checks = []
    with localcontext(numerical_context()):
        for key, state in sorted(spec['model']['states'].items()):
            if isinstance(state['initial'], dict):
                continue
            item = {'state': key, 'status': 'CHECKED'}
            try:
                value_in_bounds(number(state['initial']), state)
            except ScopeError as exc:
                item.update(status='FAILED', reason=exc.code)
            checks.append(item)
    return {
        'status': ('FAILED' if any(c['status'] == 'FAILED' for c in checks) else
                   'CHECKED_INITIALIZERS_ONLY' if checks else
                   'NOT_APPLICABLE_NO_LITERAL_INITIALIZERS'),
        'initializers': checks,
        'scope': 'LITERAL_INITIALIZATION_ONLY_NOT_TRANSITIONS_OR_EXECUTION_READINESS',
    }


def _numeric_csv_readiness(spec, root, bound):
    """Reuse initialization checks only; no rules, events or raw values returned."""
    checks = []
    for key, state in sorted(spec['model']['states'].items()):
        if not isinstance(state['initial'], dict):
            continue
        item = {'state': key, 'status': 'CHECKED'}
        with localcontext(numerical_context()) as context:
            try:
                value, lineage = aggregate_from_dataset(root, state['initial'], bound['context']['scope'])
                value_in_bounds(value, state)
                item['rows'] = lineage['rows']
                flags = sorted(s.__name__ for s, raised in context.flags.items() if raised)
                if flags:
                    item['raised_flags'] = flags
            except (ScopeError, DecimalException, csv.Error) as exc:
                item.update(status='FAILED', reason=(exc.code if isinstance(exc, ScopeError)
                            else 'STUDY_ARITHMETIC_ERROR' if isinstance(exc, DecimalException)
                            else 'STUDY_CSV_PARSE_ERROR'))
        checks.append(item)
    if checks and execution_review(root, spec, bound['context']['overrides']) != bound:
        raise ScopeError('STUDY_REVIEW_CHANGED_DURING_CSV_INSPECTION')
    return {
        'status': ('FAILED' if any(c['status'] == 'FAILED' for c in checks) else
                   'CHECKED_INITIALIZERS_ONLY' if checks else
                   'NOT_APPLICABLE_NO_NUMERIC_INITIALIZERS'),
        'initializers': checks,
        'scope': 'CSV_INITIALIZATION_ONLY_NOT_TRANSITIONS_OR_EXECUTION_READINESS',
        'numerical_accuracy': 'NOT_ESTABLISHED',
    }


def _data_requirements(spec):
    """Project validated declarations only; never resolve dataset IDs or paths."""
    datasets = []
    for data_id in dataset_references(spec):
        datasets.append({
            "data_id": data_id,
            "evidence_refs": sorted(key for key, item in spec["evidence"].items()
                                    if item.get("data_id") == data_id),
            "numeric_initializers": [
                {"state": key, "column": state["initial"]["column"],
                 "aggregate": state["initial"]["aggregate"]}
                for key, state in sorted(spec["model"]["states"].items())
                if isinstance(state["initial"], dict)
                and state["initial"]["data_id"] == data_id],
        })
    return {
        "status": "DECLARED_REQUIREMENTS_ONLY",
        "datasets": datasets,
        "explanation": (
            "Dataset IDs are declarations, not proof of available or authentic data. "
            "Evidence-only datasets still require workspace checks."
            if datasets else
            "No dataset IDs are declared. Literal initial values and parameters remain "
            "researcher declarations; workspace binding and explicit approval are still required."
        ),
        "next_steps": [
            {"stage": "WORKSPACE_REVIEW", "instruction":
             "Use study-review --file STUDY.json --workspace WORKSPACE with the intended "
             "overrides. It binds the specification, overrides, workspace scope, inventory "
             "and engine; resolves every referenced dataset and verifies stored bytes. "
             "Strict historical scope rejects unknown or after-cutoff declared publication dates."},
            {"stage": "CALCULATION_PREREQUISITES", "instruction":
             "When a workspace is supplied, read-only study review parses numeric CSV "
             "initializers and checks UTF-8 CSV, size/row limits, headers, selected columns, "
             "row shape, numeric values, nonempty input and aggregation bounds. Inspect "
             "numeric_csv_readiness for the actual outcome; without a workspace it is NOT_RUN. "
             "Evidence-only references are not numerically aggregated. Transitions and detectors "
             "are not evaluated; checked initializers do not establish runtime success. "
             "Separately approved execution rechecks inputs and may still fail."},
            {"stage": "RESEARCHER_REVIEW", "instruction":
             "Review source authenticity, data rights and scientific suitability separately. "
             "Publication checks use declared dataset dates only; row dates and revision "
             "vintages are unsupported. Inspect the bound review before separate explicit "
             "execution acknowledgement using its exact review digest; changed inputs need a new review."},
        ],
    }


def _parameter_review(spec, bound=None):
    """Compare declarations with a completed local binding, never infer evidence."""
    overrides = bound['context']['overrides'] if bound is not None else {}
    result = {
        'status': 'BOUND_VALUES_NOT_EXECUTED' if bound is not None else 'DECLARED_ONLY_NO_BOUND_REVIEW',
        'evidence_verification': 'NOT_PERFORMED',
        'items': [
            {'name': name, 'declared_value': deepcopy(parameter['value']),
             'effective_value': deepcopy(overrides.get(name, parameter['value'])) if bound is not None else None,
             'value_source': ('BOUND_OVERRIDE' if name in overrides else 'SPEC_DECLARATION')
                             if bound is not None else 'UNRESOLVED',
             'unit': parameter['unit'], 'evidence': deepcopy(parameter['evidence'])}
            for name, parameter in sorted(spec['model']['parameters'].items())],
    }
    if bound is not None:
        result['review_sha256'] = bound['review_sha256']
    return result


def review(spec, root=None, overrides=None):
    """Describe a proposal using the existing validator, with explicit scope gaps.

    Text and expressions remain untrusted data. A future UI must escape them;
    these cards are neither executable HTML nor an execution authorization.
    Without a root, workspace checks remain NOT_RUN. An optional root binds
    the proposal to workspace content and checks referenced bytes; execution,
    data rights and scientific validity are not established by that binding.
    """
    # Displayed cards, initializer checks and the execution digest must describe
    # the same proposal even if a host edits its caller-owned objects mid-review.
    spec = deepcopy(spec)
    overrides = deepcopy(overrides)
    diagnostic = diagnose(spec)
    valid = diagnostic["ready_for_execution"]
    validation = {
        "kind": "validation",
        **diagnostic,
        "workspace_data_checks": "NOT_RUN",
        "numeric_csv_readiness": {"status": "NOT_RUN", "initializers": []},
        "literal_initializer_readiness": {"status": "NOT_RUN", "initializers": []},
        "temporal_assurance": temporal_assurance(),
        "numerical_assurance": numerical_assurance(),
    }
    # Structural readiness is not a claim that a study can run in a workspace.
    validation.pop("ready_for_execution")
    if not valid and overrides is not None:
        # An invalid model cannot supply a trustworthy parameter declaration
        # against which to validate overrides. Preserve the structural reason,
        # but do not silently discard the researcher's supplied override input.
        validation["override_review"] = {
            "supplied": True, "status": "NOT_RUN",
            "reason": "STUDY_STRUCTURE_INVALID",
            "explanation": "Parameter overrides have not been validated or bound. Correct the study structure and review again.",
        }
    cards = []
    if valid:
        validation['literal_initializer_readiness'] = _literal_initializer_readiness(spec)
    # diagnose validates the complete structure before checking processing rights.
    # A rights denial may be inspected as declarations, but must not trigger any
    # workspace reads, initializer checks or binding below.
    if valid or diagnostic.get('reason') == 'STUDY_DATA_PROCESSING_RIGHTS_DENIED_OR_UNKNOWN':
        validation["data_requirements"] = _data_requirements(spec)
        cards.extend([
            {"kind": "evidence", "verification": "NOT_PERFORMED",
             "items": deepcopy(spec["evidence"])},
            {"kind": "model", "title": spec["title"],
             "question": spec["question"], "inference": spec["inference"],
             "definition": deepcopy(spec["model"]),
             "parameter_review": _parameter_review(spec),
             "export_rights": deepcopy(spec.get("export_rights")),
             "limitations": deepcopy(spec["limitations"])},
        ])
    cards.extend([
        validation,
        {"kind": "approval", "status": "REQUIRED" if valid else "BLOCKED",
         "explanation": "Review does not grant execution approval. A separate explicit operator approval is required; workspace data checks may still fail."},
    ])
    if not valid and overrides is not None:
        cards[-1]["explanation"] += " " + validation["override_review"]["explanation"]
    result = {
        "schema": "societybox-study-review-v1",
        "spec_sha256": digest(canonical(spec)),
        "execution": "NOT_RUN",
        "capabilities": negotiate_capabilities(),
        "cards": cards,
    }
    if root is not None and valid:
        result["execution_review"] = execution_review(root, spec, overrides)
        validation["workspace_data_checks"] = result["execution_review"]["data_checks"]
        validation["temporal_assurance"] = deepcopy(result["execution_review"]["temporal_assurance"])
        validation["numeric_csv_readiness"] = _numeric_csv_readiness(spec, root, result['execution_review'])
        cards[1]['parameter_review'] = _parameter_review(spec, result['execution_review'])
        cards[-1]["explanation"] = "A separate explicit operator acknowledgement and this exact execution review digest are required. Calculation prerequisites may still fail."
        if validation['numeric_csv_readiness']['status'] == 'FAILED':
            cards[-1].update(status='BLOCKED', explanation=
                            'Numeric CSV initialization failed. Correct the inputs and review again before separate execution approval.')
    if valid and validation['literal_initializer_readiness']['status'] == 'FAILED':
        cards[-1].update(status='BLOCKED', explanation=
                        'Literal state initialization violates declared bounds. Correct the initial values or bounds and review again before separate execution approval.')
    if valid:
        parameters = cards[1]['parameter_review']
        cards[-1]['parameter_evidence_review'] = {
            'parameters': deepcopy(parameters),
            'override_parameters_requiring_scientific_review': [
                item['name'] for item in parameters['items']
                if item['value_source'] == 'BOUND_OVERRIDE'],
            'evidence_support': 'NOT_ESTABLISHED',
            'explanation': (
                'Inspect declared and effective values, units and evidence IDs before '
                'separate execution acknowledgement. Evidence IDs are original declarations '
                'and do not establish support for overrides, including overrides equal to '
                'declared values. Scientific suitability and source authenticity require '
                'researcher review; this card records no evidence verification or approval.'),
        }
    return result
