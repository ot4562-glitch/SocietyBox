"""Transport-neutral request boundary; no model, network or tool executor included.

Host adapters must map host_instruction to their highest supported instruction
role without prefixing, merging or truncating the first message, and separately
verify actual model consumption. A local callback is not that verification.
"""
import json

from .agent_bootstrap import agent_context
from .agent_response_schema import host_response_format
from .workspace import ScopeError, canonical


MAX_REQUEST_BYTES = 64_000
MAX_REQUEST_PACKET_BYTES = 1024 * 1024


def parse_host_request(raw):
    """Inspect an explicit saved packet against CURRENT context, without a host.

    Packet consistency is not origin authentication or proof of consumption.
    The file bound includes the product instructions and supplemental context.
    """
    if not isinstance(raw, bytes):
        raise ScopeError("AGENT_HOST_REQUEST_BYTES_REQUIRED")
    if len(raw) > MAX_REQUEST_PACKET_BYTES:
        raise ScopeError("AGENT_HOST_REQUEST_TOO_LARGE")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError("nonfinite JSON constant")

    try:
        request = json.loads(raw.decode('utf-8'), object_pairs_hook=unique_object,
                             parse_constant=invalid_constant)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ScopeError("AGENT_HOST_REQUEST_INVALID") from exc
    # Reuse the exact adapter boundary; this callback only returns the snapshot.
    consume_host_request(request, lambda snapshot: snapshot)
    return request


def prepare_host_request(researcher_request, locale="ko"):
    """Build a prospective host input with no granted tools or data transfer rights."""
    if not isinstance(researcher_request, str) or not researcher_request.strip():
        raise ScopeError("AGENT_HOST_RESEARCHER_REQUEST_REQUIRED")
    try:
        size = len(researcher_request.encode("utf-8"))
    except UnicodeError as exc:
        raise ScopeError("AGENT_HOST_REQUEST_INVALID") from exc
    if size > MAX_REQUEST_BYTES:
        raise ScopeError("AGENT_HOST_REQUEST_TOO_LARGE")
    context = agent_context(locale)
    supplemental = {key: context[key] for key in (
        "capabilities", "reference_coverage", "background", "limitations",
    )}
    supplemental.update({
        "approval": {"status": "NOT_AUTHENTICATED", "granted_actions": []},
        "authorized_tool_schemas": [],
        "data_transfer_authorization": "NOT_GRANTED",
        "response_format": host_response_format(),
    })
    return {
        "schema": "societybox-host-request-v1",
        "locale": locale,
        "host_injection": "NOT_IMPLEMENTED",
        "first_message_sha256": context["first_message_sha256"],
        "messages": [
            {"role": "host_instruction", "content": context["first_message"]},
            {"role": "host_instruction", "content": canonical(supplemental).decode("utf-8")},
            {"role": "user", "content": researcher_request},
        ],
    }


def consume_host_request(request, consumer):
    """Validate a detached snapshot immediately before an adapter callback.

    The callback receives immutable canonical JSON bytes. The return value is
    passed through without being certified as a model result or tool approval.
    No configured provider/transport exists here. Callers remain responsible for
    authorized transfer and host-specific role/truncation/consumption checks.
    """
    try:
        snapshot = canonical(request)
        detached = json.loads(snapshot)
        expected = prepare_host_request(detached["messages"][2]["content"], detached["locale"])
    except (TypeError, ValueError, KeyError, IndexError, UnicodeError, RecursionError) as exc:
        raise ScopeError("AGENT_HOST_REQUEST_INVALID") from exc
    if snapshot != canonical(expected):
        raise ScopeError("AGENT_HOST_REQUEST_STALE_OR_ALTERED")
    if not callable(consumer):
        raise ScopeError("AGENT_HOST_CONSUMER_REQUIRED")
    return consumer(snapshot)
