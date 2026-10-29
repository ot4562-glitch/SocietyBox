"""Immutable, reproducible experiment receipts and verification."""
from pathlib import Path

from .models import compute
from .workspace import (
    ScopeError, admission, canonical, digest, get_root, load_json,
    put_json, retrieve, SHA
)

RUN_SCHEMA = "econscope-deterministic-run-v1"
IMPLEMENTATION = "econscope-toy-models-0.1.0a0"


def make_run(root, *, model, params, data_id=None):
    path, _, scope = get_root(root)
    raw = None
    item = None
    decision = "SYNTHETIC_INPUT_NO_EMPIRICAL_CLAIM"
    if data_id:
        raw, item, _ = retrieve(root, data_id)
        decision = admission(item, scope)
        if scope.get("as_of") and scope.get("strict_historical", True):
            if decision == "CLAIMED_RELEASE_AFTER_CUTOFF":
                raise ScopeError("FUTURE_DATA_EXCLUDED_FROM_SCOPE")
            if decision == "UNKNOWN_PUBLICATION_DATE":
                raise ScopeError("UNDATED_DATA_EXCLUDED_FROM_SCOPE")
    output = compute(model, raw, params)
    record = {
        "schema": RUN_SCHEMA, "implementation": IMPLEMENTATION,
        "scope": scope, "scope_sha256": digest(canonical(scope)),
        "model": model, "params": params,
        "data_id": data_id, "dataset_admission": decision,
        "result": output, "results_are_empirical": False,
        "provider_authenticated": False,
        "llm_used": False,
        "allow_execute_imported_code": False
    }
    identifier = digest(canonical(record))
    outfile = path / "runs" / (identifier + ".json")
    if outfile.is_symlink():
        raise ScopeError("RUN_SYMLINK_FORBIDDEN")
    if outfile.exists():
        existing = load_json(outfile)
        if existing != record:
            raise ScopeError("RUN_RECORD_MISMATCH")
    else:
        put_json(outfile, record)
    return {"run_id": identifier, "saved_at": str(outfile), "record": record}


def verify_run(root, run_id):
    if not isinstance(run_id, str) or not SHA.fullmatch(run_id):
        raise ScopeError("INVALID_RUN_ID")
    path, _, _ = get_root(root)
    record_file = path / "runs" / (run_id + ".json")
    record = load_json(record_file)
    if record.get("schema") != RUN_SCHEMA or record.get("implementation") != IMPLEMENTATION:
        raise ScopeError("RUN_FORMAT_OR_ENGINE_VERSION_INVALID")
    if digest(canonical(record)) != run_id:
        raise ScopeError("RUN_RECORD_HASH_MISMATCH")
    if digest(canonical(record["scope"])) != record["scope_sha256"]:
        raise ScopeError("SCOPE_RECEIPT_MISMATCH")
    raw = None
    if record.get("data_id"):
        raw, _, _ = retrieve(root, record["data_id"])
    independent = compute(record["model"], raw, record["params"])
    if independent != record["result"]:
        raise ScopeError("RUN_RECOMPUTATION_MISMATCH")
    return {"verified": True, "run_id": run_id,
            "checks": ["record_sha256", "scope_sha256", "imported_data_sha256", "model_recomputation"],
            "scope": record["scope"], "result": record["result"]}
