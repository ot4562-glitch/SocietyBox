"""Packaged prospective response contract; no provider or tool authorization."""
import hashlib
import json
from importlib import resources

from .workspace import ScopeError


def host_response_format():
    """Return a fresh schema and its exact asset digest for prospective adapters.

    JSON Schema describes structure. The parser additionally enforces the wire
    rules, such as duplicate-key and byte limits, which schema validation alone
    cannot establish. No external schema references are resolved or fetched.
    """
    raw = resources.files("econscope").joinpath(
        "resources", "host-response.v1.schema.json").read_bytes()
    try:
        if len(raw) > 16_000:
            raise ValueError("oversized schema")
        schema = json.loads(raw.decode("utf-8"))
        if schema.get("$id") != "urn:societybox:host-response:v1":
            raise ValueError("wrong schema identity")
    except (ValueError, UnicodeError, AttributeError) as exc:
        raise ScopeError("AGENT_HOST_RESPONSE_SCHEMA_INVALID") from exc
    return {
        "status": "PROSPECTIVE_FORMAT_NOT_HOST_NEGOTIATED",
        "asset": "host-response.v1.schema.json",
        "asset_sha256": hashlib.sha256(raw).hexdigest(),
        "json_schema": schema,
        "parser_required": "econscope.agent_response.parse_host_response",
        "tool_authorization": "NOT_GRANTED",
        "claim_verification": "NOT_RUN",
    }
