"""Read-only researcher cards for a locally consistent, untrusted host turn.

Strings and proposed arguments are data, including any embedded markup or
approval claims. Renderers must display them as escaped text, never execute
them. These cards are not study evidence, run receipts or approval tokens.
"""
import json

from .agent_turn import inspect_host_turn


def review_host_turn(result):
    """Reinspect current context before projecting detached review card data.

    No provider, tool, report writer or execution/approval path is called. Full
    response text may be sensitive; callers own display and storage policy.
    A consistent local record establishes neither model origin nor claim truth.
    """
    response = inspect_host_turn(result)
    return {
        "schema": "societybox-host-turn-review-v1",
        "binding": {
            "result_sha256": result.result_sha256,
            "request_sha256": result.request_sha256,
            "raw_response_sha256": result.raw_response_sha256,
            "consistency": "CURRENT_CONTEXT_CHECK_PASS",
            "context_scope": "PRODUCT_HOST_CONTRACT_ONLY",
            "selected_research_inputs": "NOT_BOUND",
            "response_request_binding": "UNAUTHENTICATED_LOCAL_PAIRING",
            "assurance": result.assurance,
            "model_consumption": result.model_consumption,
        },
        "execution": "NOT_RUN",
        "authorization": "NOT_GRANTED",
        "report_evidence": "INELIGIBLE_UNTRUSTED_MODEL_OUTPUT",
        "cards": [
            {"kind": "assistant_text", "text": response.assistant_text,
             "presentation": "PLAIN_TEXT", "trust": response.trust,
             "claim_verification": response.claim_verification},
            {"kind": "tool_proposals", "execution": "NOT_RUN", "items": [
                {"id": call.id, "name": call.name,
                 "arguments": json.loads(call.arguments_json),
                 "presentation": "PLAIN_TEXT", "trust": response.trust,
                 "authorization": call.authorization, "status": call.status}
                for call in response.tool_calls]},
            {"kind": "review_limits", "text":
             "Consistency checks do not verify scientific claims or model origin. "
             "Current context means the product host contract only. Selected studies, "
             "workspace data, models, runs and comparisons are not bound to this turn; "
             "their changes are not detected by this check. Pairing local request and "
             "response bytes does not prove the response was generated for that request. "
             "Model-authored approval and tool arguments grant no authority. "
             "Study execution requires separate product review and explicit operator approval; "
             "reports require checked computation and source or assumption lineage."},
        ],
    }
