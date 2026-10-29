"""Offline, read-only first-message instructions and factual capability/knowledge index.

This does not connect a model, execute upstream source or grant authorization.
An actual host adapter must inject the returned first_message before study turns,
then separately enforce the tool schema, input approval and institution policy.
"""
import hashlib
import json
from importlib import resources

from .study_capabilities import negotiate_capabilities
from .workspace import ScopeError


LOCALES = ("en", "zh-Hans", "ko", "ja")
_ASSETS = frozenset(("AGENT.md", "background.v1.json", "upstream.v1.json"))


def _read_asset(name):
    if name not in _ASSETS:
        raise ScopeError("AGENT_ASSET_UNKNOWN")
    # Resolved from the installed package, not the caller's working directory.
    raw = resources.files("econscope").joinpath("resources", name).read_bytes()
    if len(raw) > 250_000:
        raise ScopeError("AGENT_ASSET_TOO_LARGE")
    # Text-mode reads normalize CRLF, violating exact first-message byte identity.
    return raw.decode('utf-8')


def _read_registry(name, expected_schema):
    try:
        result = json.loads(_read_asset(name))
    except (UnicodeError, ValueError) as exc:
        raise ScopeError("AGENT_CATALOG_INVALID") from exc
    if not isinstance(result, dict) or result.get("schema") != expected_schema:
        raise ScopeError("AGENT_CATALOG_SCHEMA_INVALID")
    return result


def _locale(value):
    if value not in LOCALES:
        raise ScopeError("AGENT_LOCALE_UNSUPPORTED")
    return value


def _catalog():
    data = _read_registry("upstream.v1.json", "societybox-upstream-feature-family-registry-v1")
    rows = data.get("sources")
    if not isinstance(rows, list) or len(rows) != 9:
        raise ScopeError("AGENT_CATALOG_INVALID")
    all_ids = [f["id"] for source in rows for f in source.get("families", ())]
    if len(all_ids) != 54 or len(set(all_ids)) != 54:
        raise ScopeError("AGENT_CATALOG_INVALID")
    if data.get("complete_public_api_inventory") is not False or any(
        f.get("callable") is not False or f.get("native_execution") != "NOT_RUN"
        for source in rows for f in source["families"]
    ):
        raise ScopeError("AGENT_CATALOG_OVERCLAIM")
    return data


def _background():
    data = _read_registry("background.v1.json", "societybox-offline-background-v1")
    cards = data.get("cards")
    if not isinstance(cards, list) or not cards:
        raise ScopeError("AGENT_BACKGROUND_INVALID")
    if len(set(c["id"] for c in cards)) != len(cards):
        raise ScopeError("AGENT_BACKGROUND_INVALID")
    for c in cards:
        if (set(c.get("title", {})) != set(LOCALES) or
                set(c.get("summary", {})) != set(LOCALES) or
                c.get("source_verified_for_empirical_facts") is not False or
                c.get("numeric_results_present") is not False):
            raise ScopeError("AGENT_BACKGROUND_INVALID")
    return data


def reference_capabilities(source_id=None):
    """All nine local source families; *not* all native APIs or callable adapters."""
    data = _catalog()
    source_items = data["sources"]
    if source_id is not None:
        source_items = [x for x in source_items if x["source_id"] == source_id]
        if not source_items:
            raise ScopeError("REFERENCE_SOURCE_UNKNOWN")
    return {
        "schema": data["schema"],
        "status": "INVENTORY_ONLY_NO_NATIVE_ADAPTERS",
        "source_manifest": data["source_manifest"],
        "public_api_denominator": data["public_api_denominator"],
        "complete_public_api_inventory": False,
        "total_upstream_projects": len(data["sources"]),
        "registered_feature_families": data["registered_feature_families"],
        "research_feature_families": data["research_feature_families"],
        "sources": source_items,
    }


def knowledge(topic=None, locale="ko"):
    """Curated *original* offline primers. No scientific source verification implied."""
    locale = _locale(locale)
    data = _background()
    if topic is None:
        return {
            "schema": data["schema"], "locale": locale,
            "status": "BACKGROUND_INDEX_NOT_EMPIRICAL_EVIDENCE",
            "offline_dataset_payloads": data["offline_dataset_payloads"],
            "topics": [{"id": x["id"], "category": x["category"],
                        "title": x["title"][locale], "summary": x["summary"][locale]}
                       for x in data["cards"]],
            "source_portal_index": data["source_portal_index"],
            "limitations": data["limits"],
        }
    matched = [x for x in data["cards"] if x["id"] == topic]
    if not matched:
        raise ScopeError("AGENT_BACKGROUND_TOPIC_UNKNOWN")
    item = matched[0]
    return {
        "schema": data["schema"], "locale": locale,
        "status": "ORIGINAL_OFFLINE_METHOD_PRIMER_NOT_EMPIRICAL_EVIDENCE",
        "id": item["id"], "title": item["title"][locale],
        "summary": item["summary"][locale],
        "method_brief_en": item["method_brief_en"],
        "critical_prerequisites": item["critical_prerequisites"],
        "reference_ids": item["reference_ids"],
        "source_kind": item["source_kind"],
        "source_verified_for_empirical_facts": False,
        "numeric_results_present": False,
        "limits": data["limits"],
    }


def agent_context(locale="ko"):
    """Create first-instruction bytes and *current* read-only catalogs for host use.

    The returned packet is a testable prospective host input; it is NOT proof that
    any actual remote/local model host injected the message.
    """
    locale = _locale(locale)
    agent = _read_asset("AGENT.md")
    if not agent.startswith("# Society Box / AGENT.md"):
        raise ScopeError("AGENT_INSTRUCTION_INVALID")
    catalog = _catalog()
    index = knowledge(locale=locale)
    capabilities = negotiate_capabilities()
    return {
        "schema": "societybox-llm-first-context-v1",
        "status": "BOOTSTRAP_ASSET_READY_HOST_NOT_CONNECTED",
        "host_injection": "NOT_IMPLEMENTED",
        "first_message_role": "HIGHEST_ALLOWED_HOST_INSTRUCTION",
        "required_delivery_order": [
            "AGENT.md exact first_message",
            "actual current study-capabilities and authorized tool schemas",
            "pinned reference capability statuses and scoped background index",
            "researcher request",
        ],
        "first_message": agent,
        "first_message_sha256": hashlib.sha256(agent.encode("utf-8")).hexdigest(),
        "locale": locale,
        "capabilities": capabilities,
        "reference_coverage": {
            "projects": catalog["total_upstream_projects"],
            "research_projects": catalog["research_compute_projects"],
            "documented_feature_families": catalog["registered_feature_families"],
            "research_feature_families": catalog["research_feature_families"],
            "complete_public_api_inventory": False,
            "native_adapters_verified_by_this_registry": 0,
            "source_ids": [x["source_id"] for x in catalog["sources"]],
            "full_catalog_command": "reference-capabilities",
        },
        "background": {
            "locale": locale,
            "offline_topic_count": len(index["topics"]),
            "topics": index["topics"],
            "empirical_evidence": "NOT_INCLUDED",
            "full_topic_command": "study-knowledge --topic ID --locale LOCALE",
        },
        "limitations": [
            "This is read-only local context, not a connected LLM session or actual host instruction injection.",
            "The full host adapter must enforce authenticated approvals, tool schemas and institution security policy outside prompts.",
            "The registry covers documented feature families, not every upstream API or executed adapter.",
            "Background cards do not contain validated empirical source data or legally cleared third-party books.",
        ],
    }
