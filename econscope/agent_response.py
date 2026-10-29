"""Typed, inert model output. Parsing never approves, dispatches or verifies claims.

No tools are currently authorized at the host boundary. Even recognizable tool
names and approval-shaped arguments are retained only as blocked proposals.
Future dispatch needs a separate authenticated product boundary; these records
must never be used as approval tokens, run receipts or verified report evidence.
"""
from dataclasses import dataclass, field
import json
import re

from .agent_host import consume_host_request
from .workspace import ScopeError, canonical


MAX_RESPONSE_BYTES = 64_000
MAX_TOOL_CALLS = 64
_IDENTIFIER = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.:-]{0,127}", re.ASCII)


@dataclass(frozen=True)
class ProposedToolCall:
    id: str
    name: str
    arguments_json: bytes
    authorization: str = field(default="NOT_GRANTED", init=False)
    status: str = field(default="BLOCKED_NO_AUTHORIZED_TOOLS", init=False)


@dataclass(frozen=True)
class HostResponse:
    assistant_text: str
    tool_calls: tuple[ProposedToolCall, ...]
    trust: str = field(default="UNTRUSTED_MODEL_OUTPUT", init=False)
    claim_verification: str = field(default="NOT_RUN", init=False)


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _constant(value):
    raise ValueError("nonfinite JSON constant")


def _identifier(value):
    return isinstance(value, str) and _IDENTIFIER.fullmatch(value) is not None


def parse_host_response(raw):
    """Parse bounded UTF-8 JSON bytes to immutable, explicitly untrusted records.

    The exact wire fields are schema, assistant_text and tool_calls. Each call
    contains id, name and object-valued arguments. Unknown envelope fields and
    duplicate keys fail closed; arguments have no authority regardless of text.
    This is wire validation only, with no scientific or tool-argument validation.
    """
    if not isinstance(raw, bytes):
        raise ScopeError("AGENT_HOST_RESPONSE_BYTES_REQUIRED")
    if len(raw) > MAX_RESPONSE_BYTES:
        raise ScopeError("AGENT_HOST_RESPONSE_TOO_LARGE")
    try:
        data = json.loads(raw.decode("utf-8"), object_pairs_hook=_object,
                          parse_constant=_constant)
        if (not isinstance(data, dict) or
                set(data) != {"schema", "assistant_text", "tool_calls"} or
                data["schema"] != "societybox-host-response-v1" or
                not isinstance(data["assistant_text"], str) or
                not isinstance(data["tool_calls"], list) or
                len(data["tool_calls"]) > MAX_TOOL_CALLS):
            raise ValueError("invalid response envelope")
        # Also rejects unpaired Unicode surrogates and numeric overflow (1e999),
        # including values nested within otherwise inert proposed arguments.
        canonical(data)
        calls = []
        seen = set()
        for call in data["tool_calls"]:
            if (not isinstance(call, dict) or set(call) != {"id", "name", "arguments"} or
                    not _identifier(call["id"]) or not _identifier(call["name"]) or
                    not isinstance(call["arguments"], dict) or call["id"] in seen):
                raise ValueError("invalid proposed tool call")
            seen.add(call["id"])
            calls.append(ProposedToolCall(call["id"], call["name"], canonical(call["arguments"])))
        return HostResponse(data["assistant_text"], tuple(calls))
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise ScopeError("AGENT_HOST_RESPONSE_INVALID") from exc


def receive_host_response(request, consumer):
    """Validate request ordering before callback, then parse its untrusted bytes.

    The callback remains trusted adapter code; this API cannot sandbox callback
    effects or establish model consumption, provider connection or transfer rights.
    There is deliberately no tool executor or approval callback parameter.
    """
    return parse_host_response(consume_host_request(request, consumer))
