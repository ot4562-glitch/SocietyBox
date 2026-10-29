"""Offline fixture or local response import with read-only study inspection."""
from copy import deepcopy
from html import escape
import json

from .agent_host import prepare_host_request
from .agent_preview_labels import PREVIEW_TEXT
from .agent_study_preview_labels import STUDY_PREVIEW_TEXT
from .agent_review_html import render_host_turn_html
from .agent_turn import receive_host_turn
from .workspace import ScopeError, canonical
from .study_review import review as study_review
from .agent_dataset_preview import dataset_section


def _study_section(spec, workspace=None, overrides=None, locale='en'):
    labels = STUDY_PREVIEW_TEXT[locale]
    try:
        review = study_review(spec, workspace, overrides)
    except ScopeError as exc:
        # Keep the structural inspection visible when workspace binding fails.
        review = study_review(spec)
        validation = next(card for card in review['cards'] if card['kind'] == 'validation')
        validation['workspace_data_checks'] = 'FAILED'
        validation['workspace_failure'] = exc.code
        if overrides is not None:
            validation['parameter_override_checks'] = 'FAILED_NO_BOUND_REVIEW'
        review['cards'][-1]['status'] = 'BLOCKED'
    validation = next(card for card in review['cards'] if card['kind'] == 'validation')
    checks = validation['workspace_data_checks']
    numeric = validation['numeric_csv_readiness']['status']
    parts = [
        '<section><h2>' + escape(labels['heading']) + '</h2>',
        '<p>' + escape(labels['limits']) + ' '
        + escape(labels['workspace']) + ': ' + escape(checks) + '. '
        + escape(labels['numeric']) + ': ' + escape(numeric) + '. '
        + escape(labels['execution']) + ': NOT_RUN. '
        + escape(labels['authorization']) + ': NOT_GRANTED. '
        + escape(labels['approval_limits']) + '</p>',
        '<p>' + escape(labels['sha256']) + ': ' + escape(review['spec_sha256']) + '</p>',
    ]
    parts.append(dataset_section(review, locale))
    for card in review['cards']:
        parts.extend([
            '<section><h3>' + escape(labels[card['kind']]) + '</h3><pre><code>',
            escape(json.dumps(card, ensure_ascii=False, allow_nan=False, indent=2), quote=True),
            '</code></pre></section>',
        ])
    if 'execution_review' in review:
        parts.extend([
            '<section><h3>' + escape(labels['bound']) + '</h3><pre><code>',
            escape(json.dumps(review['execution_review'], ensure_ascii=False,
                              allow_nan=False, indent=2), quote=True),
            '</code></pre><p>' + escape(labels['bound_limits']) + '</p></section>',
        ])
    parts.append('</section>')
    return ''.join(parts)


def conversation_preview(locale="en", *, study_spec=None, workspace=None, overrides=None,
                         researcher_request=None, response_bytes=None, run_id=None,
                         comparison_bundle=None, recomputation_receipt=None,
                         model_directory=None, model_file=None, model_expected_sha256=None,
                         model_artifact_id=None, model_show_source=False,
                         model_compare_artifact_id=None):
    """Return inert HTML through the normal turn boundary.

    An optional study is inspected separately, never put in the host request.
    Optional workspace binding checks bytes and declared dates read-only.
    Selected-run verification recomputes the supported deterministic model only.
    No new run or approval is created. Without paired request/response inputs,
    use authored fixture text. Imported bytes have no authenticated model origin
    or proof they were generated for the locally paired researcher request.
    Portable comparisons check artifact consistency only, without reexecution.
    A selected .mod is reviewed separately as inert bytes, never host content.
    Stored selection verifies workspace hash/size without reopening the source.
    Source display requires explicit opt-in and preserved artifact selection.
    """
    # Keep study cards and selected-result bindings on one caller-independent
    # proposal across host preparation, turn rendering and workspace inspection.
    study_spec = deepcopy(study_spec)
    overrides = deepcopy(overrides)
    if model_compare_artifact_id is not None and (
            workspace is None or model_artifact_id is None or model_show_source):
        raise ScopeError('MODEL_DIFF_SELECTION_INVALID')
    if (type(model_show_source) is not bool or
            (model_show_source and model_artifact_id is None)):
        raise ScopeError('CONVERSATION_PREVIEW_MODEL_SOURCE_SELECTION_INVALID')
    if model_artifact_id is not None and (
            workspace is None or model_directory is not None or model_file is not None):
        raise ScopeError('CONVERSATION_PREVIEW_MODEL_ARTIFACT_SELECTION_INVALID')
    if ((model_directory is None) != (model_file is None) or
            (model_expected_sha256 is not None and model_file is None and model_artifact_id is None)):
        raise ScopeError('CONVERSATION_PREVIEW_MODEL_SELECTION_REQUIRED')
    if recomputation_receipt is not None and comparison_bundle is None:
        raise ScopeError('STUDY_PREVIEW_RECEIPT_REQUIRES_COMPARISON')
    if run_id is not None and (workspace is None or study_spec is None):
        raise ScopeError('STUDY_PREVIEW_RUN_REQUIRES_STUDY_AND_WORKSPACE')
    if workspace is not None and study_spec is None and model_artifact_id is None:
        raise ScopeError('STUDY_PREVIEW_WORKSPACE_REQUIRES_STUDY')
    if overrides is not None and (workspace is None or study_spec is None):
        raise ScopeError('STUDY_PREVIEW_OVERRIDES_REQUIRE_STUDY_AND_WORKSPACE')
    if (researcher_request is None) != (response_bytes is None):
        raise ScopeError('CONVERSATION_PREVIEW_REQUEST_RESPONSE_PAIR_REQUIRED')
    imported = response_bytes is not None
    # Unsupported locales still fail at the normal host boundary below.
    labels = PREVIEW_TEXT.get(locale, PREVIEW_TEXT['en']) if isinstance(locale, str) else PREVIEW_TEXT['en']
    request = prepare_host_request(
        researcher_request if imported else labels['request'],
        locale,
    )
    model_html = None
    if model_compare_artifact_id is not None:
        from .agent_model_preview import compared_model_section
        model_html = compared_model_section(workspace, model_artifact_id,
            model_compare_artifact_id, model_expected_sha256, locale)
    elif model_file is not None or model_artifact_id is not None:
        from .agent_model_preview import selected_model_section
        model_html = selected_model_section(
            model_directory, model_file, model_expected_sha256, locale,
            workspace=workspace, artifact_id=model_artifact_id,
            show_source=model_show_source)
    wire = canonical({
        "schema": "societybox-host-response-v1",
        "assistant_text": labels['assistant'],
        "tool_calls": [{"id": "fixture-review", "name": "study-review",
                        "arguments": {"file": "UNSELECTED_FIXTURE_ONLY"}}],
    })
    result = receive_host_turn(request, lambda _: response_bytes if imported else wire)
    html = render_host_turn_html(result)
    provenance = 'LOCAL RESPONSE IMPORT' if imported else 'OFFLINE FIXTURE'
    introduction = (
        '<main><section><h2>' + escape(labels['heading']) + ': ' + provenance + '</h2><p>'
        + escape(labels['import' if imported else 'fixture']) + '</p></section>'
    )
    html = html.replace("<main>", introduction, 1)
    if model_html is not None:
        html = html.replace('</main>', model_html + '</main>', 1)
    if study_spec is not None:
        html = html.replace('</main>', _study_section(study_spec, workspace, overrides, locale) + '</main>', 1)
    if run_id is not None:
        from .agent_run_preview import selected_run_section
        html = html.replace('</main>', selected_run_section(
            workspace, run_id, study_spec, overrides, locale) + '</main>', 1)
    if comparison_bundle is not None:
        from .agent_comparison_preview import comparison_section
        html = html.replace('</main>', comparison_section(
            comparison_bundle, study_spec, locale, recomputation_receipt) + '</main>', 1)
    return html
