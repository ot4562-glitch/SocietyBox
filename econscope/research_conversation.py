"""Explicit, receipt-bound local model conversation over a separate research chat.

A review shows exactly one selected text before any network transfer. An
explicit begin authorizes *only that text* for one local loopback model; send
uses the existing validated host transport and retains all returned tool calls
as inert, untrusted proposals. Study calculation always uses the independent
ResearchChat approval path, never an LLM message. No cloud/remote endpoint,
implicit history/attachment export or automatic retry is available.
"""
from copy import deepcopy
import os
from pathlib import Path
import tempfile
import threading
import uuid

from . import engine_study as engine
from .agent_host import prepare_host_request
from .llm_host_transport import (_local_endpoint, _model_id,
                                 call_loopback_host, MAX_WIRE_BYTES)
from .execution_control import safe_local_path
from .workspace import ScopeError

_SCHEMA = "societybox-research-conversation-v1"
_LIMIT = 2 * 1024 * 1024
_TURNS = 128
_CLASSES = ("PUBLIC_OR_SYNTHETIC", "PRIVATE_APPROVED_LOCAL")
_ID_LENGTH = 32
# A validated provider envelope is <= MAX_WIRE_BYTES. This conservatively
# includes its decoded text/arguments plus fixed inert tool-call metadata.
# Reserve for every outstanding grant before any approved text can be sent.
_RESULT_RESERVE = 4 * MAX_WIRE_BYTES


def _boundary_code_sha256():
    """Bind local transfer review to the exact transport/host execution boundary."""
    names = ("research_conversation.py", "llm_host_transport.py",
             "agent_host.py", "agent_bootstrap.py", "agent_response.py")
    return engine._sha(engine._bytes({
        name: engine._sha(Path(__file__).with_name(name).read_bytes())
        for name in names}))


class ResearchConversation:
    def __init__(self, chat, *, provider=None, model=None, base_url=None):
        self.chat = chat
        self._lock = threading.RLock()
        if provider is None and model is None and base_url is None:
            self.provider = self.model = self.base_url = None
        else:
            if provider not in ("ollama", "openai-compatible-local"):
                raise ValueError("CONVERSATION_LOCAL_PROVIDER_REQUIRED")
            _model_id(model)
            if base_url is None:
                raise ValueError("CONVERSATION_EXPLICIT_ENDPOINT_REQUIRED")
            host, port = _local_endpoint(base_url)
            self.provider, self.model = provider, model
            # Preserve an exact literal loopback route, never accept DNS or
            # non-root paths. The URL is not user-controlled after startup.
            self.base_url = ("http://[" + host + "]:" + str(port) if ":" in host
                             else "http://" + host + ":" + str(port))
        self._file = safe_local_path(chat.root / "conversation.json")
        # Fail at startup if an earlier process left tampered state.
        self._load()

    def status(self):
        return {
            "schema": "societybox-local-conversation-capabilities-v1",
            "mode": "NO_LLM" if self.provider is None else
                    "LOCAL_LLM_CONFIGURED_NOT_CONNECTED",
            "external_transfer": "DISABLED" if self.provider is None else
                                 "EXPLICIT_SELECTED_TEXT_ONLY_TO_LOOPBACK",
            "tool_execution": "NOT_GRANTED",
            "model_instruction_consumption": "NOT_INDEPENDENTLY_ATTESTED",
            "local_provider": self.provider or "NONE",
        }

    def _load(self):
        if not self._file.exists():
            return {"schema": _SCHEMA, "turns": []}, None
        raw = engine._read(safe_local_path(self._file), _LIMIT)
        sealed = engine._parse(raw)
        if (type(sealed) is not dict or set(sealed) != {"state", "sha256"} or
                sealed["sha256"] != engine._sha(engine._bytes(sealed["state"]))):
            raise ValueError("CONVERSATION_RECORD_TAMPERED")
        state = sealed["state"]
        if (type(state) is not dict or set(state) != {"schema", "turns"}
                or state["schema"] != _SCHEMA
                or type(state["turns"]) is not list
                or len(state["turns"]) > _TURNS):
            raise ValueError("CONVERSATION_RECORD_INVALID")
        ids = set()
        for row in state["turns"]:
            if (type(row) is not dict or set(row) != {
                    "id", "status", "review", "result", "error"} or
                    type(row["id"]) is not str or len(row["id"]) != _ID_LENGTH
                    or any(c not in "0123456789abcdef" for c in row["id"])
                    or row["id"] in ids or
                    row["status"] not in ("AUTHORIZED_NOT_SENT", "IN_FLIGHT",
                                         "RESPONSE_RECEIVED_UNTRUSTED",
                                         "FAILED_NO_AUTOMATIC_RETRY", "STALE_NOT_SENT")):
                raise ValueError("CONVERSATION_RECORD_INVALID")
            ids.add(row["id"])
            review = row["review"]
            if (type(review) is not dict or
                    review.get("review_sha256") != self._seal_review(review)):
                raise ValueError("CONVERSATION_REVIEW_TAMPERED")
        return state, raw

    @staticmethod
    def _seal_review(review):
        if type(review) is not dict:
            raise ValueError("CONVERSATION_REVIEW_INVALID")
        return engine._sha(engine._bytes(
            {k: v for k, v in review.items() if k != "review_sha256"}))

    def _save(self, state, old):
        raw = engine._bytes({"state": state, "sha256": engine._sha(engine._bytes(state))})
        pending = sum(row["status"] in ("AUTHORIZED_NOT_SENT", "IN_FLIGHT")
                      for row in state["turns"])
        if len(raw) + pending * _RESULT_RESERVE > _LIMIT or len(state["turns"]) > _TURNS:
            raise ValueError("CONVERSATION_RECORD_BUDGET")
        fd, name = tempfile.mkstemp(prefix=".conversation-", dir=self.chat.root)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            if old is None:
                if self._file.exists():
                    raise ValueError("CONVERSATION_CHANGED_OUTSIDE_PROCESS")
            elif engine._read(safe_local_path(self._file), _LIMIT) != old:
                raise ValueError("CONVERSATION_CHANGED_OUTSIDE_PROCESS")
            os.replace(name, self._file)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    def _select(self, text, locale, expected_revision, classification):
        if self.provider is None:
            raise ValueError("CONVERSATION_NO_LLM_CONFIGURED")
        if (type(text) is not str or not text.strip() or len(text) > 4000 or
                classification not in _CLASSES or type(classification) is not str):
            raise ValueError("CONVERSATION_SELECTED_TEXT_OR_CLASS_INVALID")
        if locale not in ("en", "ko", "zh-Hans", "ja") or type(locale) is not str:
            raise ValueError("CONVERSATION_LOCALE_INVALID")
        if (type(expected_revision) is not int or
                expected_revision != self.chat.snapshot()["revision"]):
            raise ValueError("CHAT_REVISION_CONFLICT")
        packet = prepare_host_request(text, locale)
        binding = {
            "schema": "societybox-local-conversation-review-v1",
            "text": text,
            "classification": classification,
            "locale": locale,
            "expected_revision": expected_revision,
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "host_packet_sha256": engine._sha(engine._bytes(packet)),
            "transfer_boundary_sha256": _boundary_code_sha256(),
            "first_message_sha256": packet["first_message_sha256"],
            "approval": "NOT_GRANTED",
            "scope": "ONE_SELECTED_TEXT_TO_EXPLICIT_LOOPBACK_ONLY",
        }
        binding["review_sha256"] = self._seal_review(binding)
        return binding

    def review(self, text, locale, *, expected_revision, classification):
        """Pure preflight: inspect transfer scope; NEVER contact the model."""
        return self._select(text, locale, expected_revision, classification)

    def begin(self, text, locale, *, expected_revision, classification,
              acknowledgment, confirmed):
        """Persist a single-use grant, still WITHOUT contacting the model."""
        with self._lock:
            checked = self._select(text, locale, expected_revision, classification)
            if confirmed is not True:
                raise ValueError("CONVERSATION_EXPLICIT_TRANSFER_CONFIRMATION_REQUIRED")
            if acknowledgment != checked["review_sha256"]:
                raise ValueError("CONVERSATION_TRANSFER_REVIEW_CHANGED")
            state, original = self._load()
            for previous in state["turns"]:
                if (previous["status"] == "AUTHORIZED_NOT_SENT" and
                        previous["review"]["expected_revision"] != expected_revision):
                    previous["status"] = "STALE_NOT_SENT"
                    previous["error"] = "CHAT_REVISION_CHANGED_BEFORE_TRANSFER"
            key = uuid.uuid4().hex
            if any(row["id"] == key for row in state["turns"]):
                raise ValueError("CONVERSATION_TURN_ID_COLLISION")
            state["turns"].append({
                "id": key, "review": checked, "status": "AUTHORIZED_NOT_SENT",
                "result": None, "error": None})
            self._save(state, original)
            return {"id": key, "status": "AUTHORIZED_NOT_SENT",
                    "transfer_authorization": "EXPLICIT_SELECTED_TEXT_ONLY",
                    "execution_authorization": "NOT_GRANTED"}

    def _row(self, state, key):
        if type(key) is not str or len(key) != _ID_LENGTH:
            raise ValueError("CONVERSATION_TURN_INVALID")
        for row in state["turns"]:
            if row["id"] == key:
                return row
        raise ValueError("CONVERSATION_TURN_UNKNOWN")

    def send(self, turn_id, acknowledgment, *, confirmed):
        """One actual local HTTP call. Failures are terminal, never auto-retried."""
        with self._lock:
            state, original = self._load()
            row = self._row(state, turn_id)
            if (confirmed is not True or row["status"] != "AUTHORIZED_NOT_SENT"
                    or acknowledgment != row["review"]["review_sha256"]):
                raise ValueError("CONVERSATION_SEND_NOT_AUTHORIZED")
            if any(turn["status"] == "IN_FLIGHT" for turn in state["turns"]):
                raise ValueError("CONVERSATION_HOST_CALL_IN_FLIGHT")
            review = row["review"]
            if (self.provider != review["provider"] or
                    self.model != review["model"] or
                    self.base_url != review["base_url"]):
                raise ValueError("CONVERSATION_CONFIG_CHANGED")
            if self.chat.snapshot()["revision"] != review["expected_revision"]:
                row["status"] = "STALE_NOT_SENT"
                row["error"] = "CHAT_REVISION_CHANGED_BEFORE_TRANSFER"
                self._save(state, original)
                raise ValueError("CONVERSATION_STALE_CHAT_REVISION")
            packet = prepare_host_request(review["text"], review["locale"])
            if (_boundary_code_sha256() != review["transfer_boundary_sha256"] or
                    engine._sha(engine._bytes(packet)) != review["host_packet_sha256"] or
                    packet["first_message_sha256"] != review["first_message_sha256"]):
                raise ValueError("CONVERSATION_HOST_INSTRUCTIONS_CHANGED")
            row["status"] = "IN_FLIGHT"
            self._save(state, original)
        try:
            result = call_loopback_host(
                packet, provider=self.provider, model=self.model,
                base_url=self.base_url, timeout=10, allow_local_host=True,
                include_untrusted_arguments=True)
        except (ValueError, ScopeError, OSError, RuntimeError):
            with self._lock:
                state, original = self._load()
                row = self._row(state, turn_id)
                if row["status"] == "IN_FLIGHT":
                    row["status"] = "FAILED_NO_AUTOMATIC_RETRY"
                    row["error"] = "LOCAL_HOST_DELIVERY_OR_RESPONSE_UNVERIFIED"
                    self._save(state, original)
            raise
        with self._lock:
            state, original = self._load()
            row = self._row(state, turn_id)
            if row["status"] != "IN_FLIGHT":
                raise ValueError("CONVERSATION_SEND_STATE_CHANGED")
            row["status"] = "RESPONSE_RECEIVED_UNTRUSTED"
            row["result"] = result
            self._save(state, original)
        return {"id": turn_id, "status": "RESPONSE_RECEIVED_UNTRUSTED",
                "claim_verification": "NOT_RUN", "tool_execution": "NOT_GRANTED"}

    def read(self, turn_id):
        with self._lock:
            state, _ = self._load()
            return deepcopy(self._row(state, turn_id))

    def list(self, *, offset=0, size=20):
        if (type(offset) is not int or offset < 0 or
                type(size) is not int or not 1 <= size <= 50):
            raise ValueError("CONVERSATION_PAGE_INVALID")
        with self._lock:
            state, _ = self._load()
            return {"schema": _SCHEMA, "total": len(state["turns"]),
                    "items": deepcopy(state["turns"][offset:offset+size])}

    def draft(self, turn_id, call_id):
        """Return inert model-authored candidate, never grant executable rights."""
        record = self.read(turn_id)
        if record["status"] != "RESPONSE_RECEIVED_UNTRUSTED":
            raise ValueError("CONVERSATION_MODEL_PROPOSAL_UNAVAILABLE")
        result = record["result"]
        if type(call_id) is not str:
            raise ValueError("CONVERSATION_MODEL_PROPOSAL_UNKNOWN")
        matching = [call for call in result["tool_calls"] if call["id"] == call_id
                    and call["name"] == "propose_engine_study"
                    and call["authorization"] == "NOT_GRANTED"]
        if len(matching) != 1:
            raise ValueError("CONVERSATION_MODEL_PROPOSAL_UNKNOWN")
        arguments = matching[0].get("untrusted_arguments", {})
        if type(arguments) is not dict or set(arguments) != {"study"}:
            raise ValueError("CONVERSATION_MODEL_PROPOSAL_INVALID")
        spec = deepcopy(arguments["study"])
        if (type(spec) is not dict or type(spec.get("source")) is not dict
                or set(spec["source"]) != {"id", "license", "kind", "basis"} or
                spec.get("schema") != "societybox-engine-study-v1"):
            raise ValueError("CONVERSATION_MODEL_PROPOSAL_INVALID")
        # Retain the model-authored fields for manual editing. Its source
        # assertions remain unverified declarations, not authenticated proof.
        basis = spec["source"]["basis"]
        if type(basis) is not str:
            raise ValueError("CONVERSATION_MODEL_PROPOSAL_INVALID")
        provenance = ("UNTRUSTED_MODEL_PROPOSAL; turn=" + turn_id +
                      "; result_sha256=" + engine._sha(engine._bytes(result)) +
                      "; operator must independently check rights and assumptions; ")
        if len(provenance + basis) > 2000:
            raise ValueError("CONVERSATION_MODEL_PROVENANCE_TOO_LONG")
        spec["source"]["basis"] = provenance + basis
        engine.review(spec, runtime=self.chat.runtime)
        return spec
