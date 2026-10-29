"""Local turn consistency records, not authenticated model or execution receipts.

Records retain full request/response bytes, potentially sensitive researcher text.
Nothing is persisted or transmitted here; callers must enforce storage/transfer policy.
"""
from dataclasses import dataclass, field
import hashlib
import json

from .agent_host import consume_host_request
from .agent_response import HostResponse, parse_host_response
from .workspace import ScopeError, canonical


@dataclass(frozen=True)
class HostTurnResult:
    request_json: bytes
    raw_response: bytes
    response: HostResponse
    request_sha256: str
    raw_response_sha256: str
    result_sha256: str
    schema: str = field(default="societybox-host-turn-v1", init=False)
    assurance: str = field(default="UNAUTHENTICATED_LOCAL_RECORD", init=False)
    model_consumption: str = field(default="NOT_ESTABLISHED", init=False)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _build(request_json, raw):
    response = parse_host_response(raw)
    request_sha = _sha(request_json)
    raw_sha = _sha(raw)
    projection = {
        "schema": "societybox-host-turn-v1",
        "assurance": "UNAUTHENTICATED_LOCAL_RECORD",
        "model_consumption": "NOT_ESTABLISHED",
        "request_sha256": request_sha,
        "raw_response_sha256": raw_sha,
        "response": {
            "assistant_text": response.assistant_text,
            "trust": response.trust,
            "claim_verification": response.claim_verification,
            "tool_calls": [{
                "id": call.id, "name": call.name,
                "arguments": json.loads(call.arguments_json),
                "authorization": call.authorization, "status": call.status,
            } for call in response.tool_calls],
        },
    }
    return HostTurnResult(request_json, raw, response, request_sha, raw_sha,
                          _sha(canonical(projection)))


def receive_host_turn(request, consumer):
    """Bind the validated callback snapshot and exact wire bytes to parsed output.

    Callback errors and invalid responses produce no result. The callback remains
    trusted adapter code, with the same transfer/side-effect limits as consume.
    """
    if not callable(consumer):
        raise ScopeError("AGENT_HOST_CONSUMER_REQUIRED")

    def receive(snapshot):
        return _build(snapshot, consumer(snapshot))

    return consume_host_request(request, receive)


def inspect_host_turn(result):
    """Revalidate against CURRENT request context and reparse exact response bytes.

    Rejects stale contexts as well as inconsistent fields/digests. No callback,
    execution or provider contact occurs. A consistently rewritten record can
    pass: unkeyed hashes do not authenticate who produced any bytes or when.
    """
    if type(result) is not HostTurnResult or type(result.request_json) is not bytes:
        raise ScopeError("AGENT_HOST_TURN_INVALID")
    try:
        request = json.loads(result.request_json)
        snapshot = consume_host_request(request, lambda raw: raw)
        if snapshot != result.request_json:
            raise ScopeError("AGENT_HOST_TURN_INVALID")
        expected = _build(snapshot, result.raw_response)
        if result != expected:
            raise ScopeError("AGENT_HOST_TURN_ALTERED")
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        if isinstance(exc, ScopeError):
            raise
        raise ScopeError("AGENT_HOST_TURN_INVALID") from exc
    return expected.response
