"""Content-bound research records, not source/identity authentication.

Independent Society Box implementation of the owner's investigation contracts.
No IO, code execution, network, authority ranking, or implicit claim promotion.
Per-call JSON bounds protect resources; they do not cap reference/API coverage.
"""
from collections import deque
from copy import deepcopy
import math
import re

from .workspace import ScopeError, canonical, digest

_SHA = re.compile(r"[0-9a-f]{64}\Z")
_EVIDENCE_KEYS = set(("schema evidence_id subject_sha256 producer operation authority "
                      "epistemic_state proof_kind scope coverage result parent_evidence_ids limitations").split())
_OBLIGATION_KEYS = set(("schema obligation_id claim_id subject_sha256 scope required_authority "
                        "required_proof_kind verifier_id dependencies unknown_ids blocking_reasons next_probe").split())
_STATES = {"OPEN", "INVESTIGATING", "BLOCKED", "CONTRADICTED", "RESOLVED"}


def _reject(reason):
    raise ScopeError("RESEARCH_" + reason)


def _json(value):
    count = 0

    def visit(item, depth):
        nonlocal count
        count += 1
        if count > 100000 or depth > 32:
            _reject("JSON_RESOURCE_BOUND")
        if item is None or type(item) in (str, bool, int):
            return
        if type(item) is float and math.isfinite(item):
            return
        if type(item) is dict and all(type(key) is str for key in item):
            for child in item.values():
                visit(child, depth + 1)
        elif type(item) is list:
            for child in item:
                visit(child, depth + 1)
        else:
            _reject("JSON_INVALID")
    visit(value, 0)
    try:
        raw = canonical(value)
    except (ValueError, TypeError, OverflowError, RecursionError, UnicodeError):
        _reject("JSON_INVALID")
    if len(raw) > 2 * 1024 * 1024:
        _reject("JSON_RESOURCE_BOUND")
    return raw


def _text(value):
    if type(value) is not str or not value.strip() or len(value) > 4096:
        _reject("TEXT_INVALID")


def _keys(value, keys):
    if type(value) is not dict or set(value) != keys:
        _reject("FIELDS_INVALID")


def _sha(value):
    if type(value) is not str or not _SHA.fullmatch(value):
        _reject("DIGEST_INVALID")


def _id(value, prefix):
    if type(value) is not str or not value.startswith(prefix):
        _reject("ID_INVALID")
    _sha(value[len(prefix):])


def _strings(values, prefix=None):
    if type(values) is not list:
        _reject("LIST_INVALID")
    for value in values:
        _id(value, prefix) if prefix else _text(value)
    if len(set(values)) != len(values):
        _reject("DUPLICATE_REFERENCE")


def _sealed(payload, key, prefix):
    body = deepcopy(payload)
    body[key] = prefix + digest(_json(body))
    return body


def _verify_seal(record, key, prefix):
    _id(record[key], prefix)
    content = {k: v for k, v in record.items() if k != key}
    if record[key] != prefix + digest(_json(content)):
        _reject("EVIDENCE_DIGEST" if key == "evidence_id" else "OBLIGATION_DIGEST")


def make_evidence(**fields):
    """Seal declared facts; callers must establish the asserted authority separately."""
    record = _sealed(dict(schema="societybox-evidence-v1", **fields), "evidence_id", "ev_")
    _validate_evidence(record)
    return record


def _validate_evidence(rec):
    _keys(rec, _EVIDENCE_KEYS)
    if rec["schema"] != "societybox-evidence-v1":
        _reject("EVIDENCE_SCHEMA")
    _verify_seal(rec, "evidence_id", "ev_")
    _sha(rec["subject_sha256"])
    _keys(rec["producer"], {"tool_id", "version", "code_sha256"})
    _sha(rec["producer"]["code_sha256"])
    for value in (rec["producer"]["tool_id"], rec["producer"]["version"], rec["operation"],
                  rec["authority"], rec["proof_kind"]):
        _text(value)
    if rec["epistemic_state"] not in ("OBSERVED", "DERIVED", "INFERRED", "UNKNOWN"):
        _reject("EPISTEMIC_STATE")
    if type(rec["scope"]) is not dict or not rec["scope"] or type(rec["result"]) is not dict:
        _reject("EVIDENCE_SCOPE")
    _keys(rec["coverage"], {"status", "unknown_dimensions"})
    if rec["coverage"]["status"] not in ("COMPLETE", "PARTIAL", "UNKNOWN"):
        _reject("COVERAGE_INVALID")
    _strings(rec["coverage"]["unknown_dimensions"])
    if rec["coverage"]["status"] == "COMPLETE" and rec["coverage"]["unknown_dimensions"]:
        _reject("COVERAGE_INVALID")
    _strings(rec["parent_evidence_ids"], "ev_")
    _strings(rec["limitations"])


class EvidenceLedger:
    """Copy-on-ingress/egress, atomic validated merges and closed parent DAGs."""
    def __init__(self, records=None):
        self._records = {}
        if records is not None:
            self.merge(records)

    def merge(self, records):
        if type(records) is not list:
            _reject("EVIDENCE_LIST")
        _json(records)
        staged = dict(self._records)
        for rec in records:
            _validate_evidence(rec)
            key = rec["evidence_id"]
            if key in staged and staged[key] != rec:
                _reject("EVIDENCE_CONFLICT")
            staged[key] = deepcopy(rec)
        # Iterative Kahn traversal avoids recursion limits on long valid chains.
        waiting, children = {}, {}
        for key, rec in staged.items():
            parents = rec["parent_evidence_ids"]
            if any(parent not in staged for parent in parents):
                _reject("EVIDENCE_PARENT_MISSING")
            waiting[key] = len(parents)
            for parent in parents:
                children.setdefault(parent, []).append(key)
        ready = [key for key, count in waiting.items() if count == 0]
        seen = 0
        while ready:
            key = ready.pop()
            seen += 1
            for child in children.get(key, []):
                waiting[child] -= 1
                if waiting[child] == 0:
                    ready.append(child)
        if seen != len(staged):
            _reject("EVIDENCE_PARENT_CYCLE")
        self._records = staged

    def get(self, evidence_id):
        if evidence_id not in self._records:
            _reject("EVIDENCE_REFERENCE_MISSING")
        return deepcopy(self._records[evidence_id])

    def records(self):
        return [deepcopy(self._records[key]) for key in sorted(self._records)]


def _matches(rec, *, authority, proof_kind, scope, subject=None):
    return (rec["authority"] == authority and rec["proof_kind"] == proof_kind
            and canonical(rec["scope"]) == canonical(scope) and (subject is None or rec["subject_sha256"] == subject)
            and rec["epistemic_state"] in ("OBSERVED", "DERIVED")
            and rec["coverage"]["status"] == "COMPLETE")


class UnknownRegistry:
    """Revision-linked questions. Registration is analyst intent, not an answer."""
    def __init__(self, evidence):
        self._evidence = evidence
        self._history = {}

    def register(self, question, scope, required_authority, required_proof_kind):
        for value in (question, required_authority, required_proof_kind):
            _text(value)
        if type(scope) is not dict or not scope:
            _reject("UNKNOWN_SCOPE")
        key = "unk_" + digest(_json({"question": question, "scope": scope}))
        if key in self._history:
            old = self._history[key][-1]
            if (old["required_authority"], old["required_proof_kind"]) != (required_authority, required_proof_kind):
                _reject("UNKNOWN_REQUIREMENT_CONFLICT")
            return deepcopy(old)
        rec = dict(schema="societybox-residual-unknown-v1", unknown_id=key,
                   question=question, scope=deepcopy(scope), required_authority=required_authority,
                   required_proof_kind=required_proof_kind, revision=1, previous_revision_digest=None,
                   status="OPEN", supporting_evidence_ids=[], contradicting_evidence_ids=[],
                   resolution_evidence_ids=[], recommended_probes=[], dependencies=[])
        rec["revision_digest"] = digest(_json(rec))
        self._history[key] = [rec]
        return deepcopy(rec)

    def revise(self, unknown_id, expected_revision_digest, **changes):
        old = self.current(unknown_id)
        if old["revision_digest"] != expected_revision_digest:
            _reject("UNKNOWN_REVISION_CONFLICT")
        if not set(changes) <= {"status", "supporting_evidence_ids", "contradicting_evidence_ids",
                                "resolution_evidence_ids", "recommended_probes", "dependencies"}:
            _reject("UNKNOWN_FIELDS")
        rec = deepcopy(old)
        rec.update(deepcopy(changes))
        if type(rec["status"]) is not str or rec["status"] not in _STATES:
            _reject("UNKNOWN_STATUS")
        for field in ("supporting_evidence_ids", "contradicting_evidence_ids", "resolution_evidence_ids"):
            _strings(rec[field], "ev_")
            for key in rec[field]:
                self._evidence.get(key)
        # Never erase counterevidence on a later revision, even when reopening.
        if not set(old["contradicting_evidence_ids"]) <= set(rec["contradicting_evidence_ids"]):
            _reject("UNKNOWN_COUNTEREVIDENCE_REMOVAL")
        _strings(rec["recommended_probes"])
        _strings(rec["dependencies"], "unk_")
        for key in rec["dependencies"]:
            self.current(key)
            if key == unknown_id or self._depends_on(key, unknown_id):
                _reject("UNKNOWN_DEPENDENCY_CYCLE")
        if rec["status"] == "RESOLVED":
            if (not self._resolution_supported(rec)
                    or any(not self.is_resolved(key) for key in rec["dependencies"])):
                _reject("UNKNOWN_RESOLUTION_UNSUPPORTED")
        elif rec["resolution_evidence_ids"]:
            _reject("UNKNOWN_RESOLUTION_STATE")
        rec["revision"] += 1
        rec["previous_revision_digest"] = old["revision_digest"]
        del rec["revision_digest"]
        rec["revision_digest"] = digest(_json(rec))
        self._history[unknown_id].append(rec)
        return deepcopy(rec)

    def _resolution_supported(self, rec):
        if not rec["resolution_evidence_ids"] or rec["contradicting_evidence_ids"]:
            return False
        records = self._evidence.records()
        for key in rec["resolution_evidence_ids"]:
            ev = self._evidence.get(key)
            if (not _matches(ev, authority=rec["required_authority"],
                             proof_kind=rec["required_proof_kind"], scope=rec["scope"])
                    or ev["result"].get("unknown_id") != rec["unknown_id"]
                    or ev["result"].get("outcome") != "PASS"):
                return False
            candidates = [r for r in records if r["result"].get("unknown_id") == rec["unknown_id"]
                          and r["producer"]["tool_id"] == ev["producer"]["tool_id"]]
            latest = _latest_chain(candidates, "unknown_id")
            if latest is None or latest["evidence_id"] != key:
                return False
        return True

    def is_resolved(self, unknown_id):
        """Effective closure rechecks latest evidence and all current dependencies."""
        pending, seen = [unknown_id], set()
        while pending:
            key = pending.pop()
            if key not in seen:
                seen.add(key)
                rec = self.current(key)
                if rec["status"] != "RESOLVED" or not self._resolution_supported(rec):
                    return False
                pending.extend(rec["dependencies"])
        return True

    def _depends_on(self, key, target):
        pending, seen = [key], set()
        while pending:
            node = pending.pop()
            if node == target:
                return True
            if node not in seen:
                seen.add(node)
                pending.extend(self._history[node][-1]["dependencies"])
        return False

    def current(self, unknown_id):
        if unknown_id not in self._history:
            _reject("UNKNOWN_REFERENCE_MISSING")
        return deepcopy(self._history[unknown_id][-1])

    def history(self, unknown_id):
        self.current(unknown_id)
        return deepcopy(self._history[unknown_id])

    def records(self):
        return [self.current(key) for key in sorted(self._history)]


def make_obligation(**fields):
    rec = _sealed(dict(schema="societybox-research-obligation-v1", **fields), "obligation_id", "obl_")
    _validate_obligation(rec)
    return rec


def _validate_obligation(rec):
    _keys(rec, _OBLIGATION_KEYS)
    if rec["schema"] != "societybox-research-obligation-v1":
        _reject("OBLIGATION_SCHEMA")
    _verify_seal(rec, "obligation_id", "obl_")
    _sha(rec["subject_sha256"])
    if type(rec["scope"]) is not dict or not rec["scope"]:
        _reject("OBLIGATION_SCOPE")
    for field in ("claim_id", "required_authority", "required_proof_kind", "verifier_id", "next_probe"):
        _text(rec[field])
    for field, prefix in (("dependencies", "obl_"), ("unknown_ids", "unk_"), ("blocking_reasons", None)):
        _strings(rec[field], prefix)


def _latest_verification(ob, evidence_by_verifier):
    # Lookup is built once for a coverage evaluation, rather than re-scanning
    # the entire evidence ledger for each obligation. Revision-chain validation
    # and latest-verifier semantics remain unchanged.
    candidates = evidence_by_verifier.get((ob["obligation_id"], ob["verifier_id"]), ())
    return _latest_chain(candidates, "obligation_id")


def _latest_chain(candidates, target_field):
    if not candidates:
        return None
    by_revision = {}
    for rec in candidates:
        result = rec["result"]
        _keys(result, {target_field, "revision", "previous_verification_id", "outcome"})
        revision = result["revision"]
        if (type(revision) is not int or revision < 1 or revision in by_revision
                or result["outcome"] not in ("PASS", "FAIL", "UNKNOWN", "BLOCKED")):
            _reject("VERIFICATION_CHAIN_INVALID")
        by_revision[revision] = rec
    previous = None
    for expected, revision in enumerate(sorted(by_revision), 1):
        rec = by_revision[revision]
        if revision != expected or rec["result"]["previous_verification_id"] != previous:
            _reject("VERIFICATION_CHAIN_INVALID")
        previous = rec["evidence_id"]
    return by_revision[max(by_revision)]


def evaluate_coverage(obligations, evidence, unknowns=None):
    """Exact same authority/kind/scope; latest verifier chain; no score averaging.

    Results validate declared records, not the identity or truth of a verifier.
    Trusted integrations must compute evidence themselves, never trust imported
    authority labels as proof. No verified obligations means UNKNOWN.
    """
    if type(obligations) is not list:
        _reject("OBLIGATION_LIST")
    _json(obligations)
    indexed = {}
    for ob in obligations:
        _validate_obligation(ob)
        if ob["obligation_id"] in indexed:
            _reject("OBLIGATION_DUPLICATE")
        indexed[ob["obligation_id"]] = ob
    for ob in obligations:
        if any(key not in indexed for key in ob["dependencies"]):
            _reject("OBLIGATION_DEPENDENCY_MISSING")
    records = evidence.records()
    evidence_by_verifier = {}
    for rec in records:
        key = rec["result"].get("obligation_id")
        # Imported JSON may contain non-string fields; those never match the
        # validated string IDs and must not cause an unhashable-key exception.
        if type(key) is str:
            evidence_by_verifier.setdefault(
                (key, rec["producer"]["tool_id"]), []).append(rec)
    # Build one dependency DAG instead of rescanning every pending obligation
    # at each resolution depth. Sparse fan-out and long chains stay linear.
    remaining = {key: len(ob["dependencies"]) for key, ob in indexed.items()}
    children = {}
    for key, ob in indexed.items():
        for dependency in ob["dependencies"]:
            children.setdefault(dependency, []).append(key)
    ready = deque(key for key in indexed if remaining[key] == 0)
    evaluated = {}
    while ready:
        key = ready.popleft()
        ob = indexed[key]
        latest = _latest_verification(ob, evidence_by_verifier)
        state = "UNKNOWN"
        reasons = []
        if latest is None:
            reasons.append("NO_VERIFICATION_EVIDENCE")
        elif not _matches(latest, authority=ob["required_authority"], proof_kind=ob["required_proof_kind"],
                          scope=ob["scope"], subject=ob["subject_sha256"]):
            reasons.append("LATEST_EVIDENCE_AUTHORITY_KIND_SCOPE_OR_COVERAGE_MISMATCH")
        else:
            state = latest["result"]["outcome"]
        if ob["blocking_reasons"]:
            state = "BLOCKED" if state != "FAIL" else "FAIL"
            reasons.extend(ob["blocking_reasons"])
        if any(evaluated[dep]["state"] != "PASS" for dep in ob["dependencies"]):
            reasons.append("DEPENDENCY_NOT_PASSED")
            if state == "PASS":
                state = "UNKNOWN"
        for unk in ob["unknown_ids"]:
            if unknowns is None:
                _reject("UNKNOWN_REGISTRY_REQUIRED")
            status = unknowns.current(unk)["status"]
            if not unknowns.is_resolved(unk):
                reasons.append("UNRESOLVED:" + unk)
                if state == "PASS":
                    state = "BLOCKED" if status == "BLOCKED" else "UNKNOWN"
        evaluated[key] = dict(obligation_id=key, claim_id=ob["claim_id"], state=state,
                              verification_evidence_id=None if latest is None else latest["evidence_id"],
                              reasons=reasons, next_probe=ob["next_probe"])
        for child in children.get(key, ()):
            remaining[child] -= 1
            if remaining[child] == 0:
                ready.append(child)
    if len(evaluated) != len(indexed):
        _reject("OBLIGATION_DEPENDENCY_CYCLE")
    rows = [evaluated[ob["obligation_id"]] for ob in obligations]
    result = dict(schema="societybox-claim-coverage-v1",
                  status="COMPLETE_FOR_DECLARED_OBLIGATIONS" if rows and all(r["state"] == "PASS" for r in rows) else "INCOMPLETE",
                  scope="RECORD_CONSISTENCY_NOT_VERIFIER_OR_SOURCE_AUTHENTICATION", obligations=rows)
    result["evaluation_sha256"] = digest(_json(result))
    return result
