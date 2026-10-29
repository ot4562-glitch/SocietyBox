"""Quarantined data library, manifest, time-scope contract and safe workspace operations."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import re
from pathlib import Path

SCHEMA = "econscope-local-workspace-v1"
MAX_IMPORT_BYTES = 32 * 1024 * 1024
DATE = re.compile(r"\d{4}-\d{2}-\d{2}\Z")
SHA = re.compile(r"[0-9a-f]{64}\Z")


class ScopeError(ValueError):
    """A typed validation error; no imported code is executed."""

    def __init__(self, code):
        super().__init__(code)
        self.code = code


def canonical(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def parse_date(value):
    if value is None:
        return None
    if not isinstance(value, str) or not DATE.fullmatch(value):
        raise ScopeError("DATE_FORMAT_YYYY_MM_DD_REQUIRED")
    try:
        return dt.date.fromisoformat(value)
    except ValueError:
        raise ScopeError("DATE_INVALID") from None


def safe_dir(base: Path, name: str):
    dest = base / name
    if dest.is_symlink():
        raise ScopeError("WORKSPACE_SYMLINK_FORBIDDEN")
    if dest.exists() and not dest.is_dir():
        raise ScopeError("WORKSPACE_SUBPATH_NOT_DIRECTORY")
    dest.mkdir(parents=False, exist_ok=True)
    return dest


def put_json(path: Path, value):
    if path.is_symlink():
        raise ScopeError("SYMLINK_OUTPUT_FORBIDDEN")
    blob = canonical(value) + b"\n"
    tmp = path.with_name(path.name + ".new-" + os.urandom(6).hex())
    try:
        with tmp.open("xb") as f:
            f.write(blob)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def load_json(path: Path):
    if path.is_symlink() or not path.is_file():
        raise ScopeError("MANIFEST_NOT_FOUND_OR_SYMLINK")
    if path.stat().st_size > MAX_IMPORT_BYTES:
        raise ScopeError("MANIFEST_TOO_LARGE")
    with path.open("rb") as stream:
        raw = stream.read(MAX_IMPORT_BYTES + 1)
    if len(raw) > MAX_IMPORT_BYTES:
        raise ScopeError("MANIFEST_TOO_LARGE")
    try:
        def unique(pairs):
            record = {}
            for key, val in pairs:
                if key in record:
                    raise ScopeError("DUPLICATE_JSON_KEY")
                record[key] = val
            return record
        def reject_constant(value):
            raise ValueError("nonfinite JSON constant")

        def finite_float(value):
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("JSON number overflow")
            return number

        # Reject nonstandard constants and exponent overflow before downstream
        # review hashing or approval checks can obscure the input failure.
        return json.loads(raw.decode("utf-8-sig"), object_pairs_hook=unique,
                          parse_constant=reject_constant, parse_float=finite_float)
    except ScopeError:
        # Preserve the duplicate-key contract before catching ValueError.
        raise
    except (ValueError, RecursionError) as exc:
        # CPython also rejects excessive nesting and integer digit counts.
        # Keep its limits in force and expose the same typed input failure.
        raise ScopeError("INVALID_JSON") from exc


def initialize(root):
    path = Path(root).expanduser()
    if path.is_symlink():
        raise ScopeError("WORKSPACE_SYMLINK_FORBIDDEN")
    path = path.resolve()
    if path.exists() and not path.is_dir():
        raise ScopeError("WORKSPACE_ROOT_NOT_DIRECTORY")
    path.mkdir(parents=True, exist_ok=True)
    for d in ("objects", "runs", "ai_requests"):
        safe_dir(path, d)
    manifest = path / "manifest.json"
    scope = path / "scope.json"
    if manifest.exists() or scope.exists():
        if not (manifest.exists() and scope.exists()):
            raise ScopeError("PARTIAL_WORKSPACE_INITIALIZATION")
    else:
        put_json(manifest, {"schema": SCHEMA, "objects": {}})
        put_json(scope, {"schema": "econscope-scope-v1", "domain": "cross-disciplinary",
                          "as_of": None, "region": "unspecified", "question": None,
                          "strict_historical": True})
    return {"workspace": str(path), "status": "READY", "private_by_default": True}


def get_root(root):
    path = Path(root).expanduser()
    if path.is_symlink():
        raise ScopeError("WORKSPACE_SYMLINK_FORBIDDEN")
    path = path.resolve()
    for sub in ("objects", "runs", "ai_requests"):
        if not (path / sub).is_dir() or (path / sub).is_symlink():
            raise ScopeError("WORKSPACE_DIRECTORY_INVALID")
    manifest = load_json(path / "manifest.json")
    scope = load_json(path / "scope.json")
    if manifest.get("schema") != SCHEMA or not isinstance(manifest.get("objects"), dict):
        raise ScopeError("WORKSPACE_SCHEMA_INVALID")
    if scope.get("schema") != "econscope-scope-v1":
        raise ScopeError("SCOPE_SCHEMA_INVALID")
    return path, manifest, scope


def update_scope(root, *, as_of=None, domain="cross-disciplinary", region="unspecified", question=None, strict=True):
    path, _, _ = get_root(root)
    if domain not in ("economics", "sociology", "cross-disciplinary"):
        raise ScopeError("DOMAIN_INVALID")
    if not isinstance(region, str) or len(region) > 160:
        raise ScopeError("REGION_INVALID")
    if question is not None and (not isinstance(question, str) or len(question) > 2000):
        raise ScopeError("QUESTION_INVALID")
    parse_date(as_of)
    scope = {"schema": "econscope-scope-v1", "domain": domain, "as_of": as_of,
             "region": region, "question": question, "strict_historical": bool(strict)}
    put_json(path / "scope.json", scope)
    return scope


def ingest(root, source, *, published=None, license_label="UNKNOWN"):
    path, manifest, _ = get_root(root)
    src = Path(source).expanduser()
    if src.is_symlink() or not src.is_file():
        raise ScopeError("INPUT_NOT_REGULAR_FILE")
    n = src.stat().st_size
    if n > MAX_IMPORT_BYTES:
        raise ScopeError("INPUT_TOO_LARGE")
    parse_date(published)
    if not isinstance(license_label, str) or len(license_label) > 140:
        raise ScopeError("LICENSE_LABEL_INVALID")
    with src.open("rb") as f:
        blob = f.read(MAX_IMPORT_BYTES + 1)
    if len(blob) != n or len(blob) > MAX_IMPORT_BYTES:
        raise ScopeError("INPUT_CHANGED_DURING_READ")
    key = digest(blob)
    object_file = path / "objects" / (key + ".bin")
    if object_file.is_symlink():
        raise ScopeError("OBJECT_SYMLINK_FORBIDDEN")
    if not object_file.exists():
        try:
            with object_file.open("xb") as f:
                f.write(blob)
                f.flush()
                os.fsync(f.fileno())
        except FileExistsError:
            pass
    # Existing objects can be replaced or grow before verification. Bound the
    # read itself rather than trusting an earlier size observation.
    with object_file.open("rb") as f:
        stored = f.read(MAX_IMPORT_BYTES + 1)
    if len(stored) > MAX_IMPORT_BYTES or digest(stored) != key:
        raise ScopeError("OBJECT_TAMPER_DETECTED")
    if key not in manifest["objects"]:
        manifest["objects"][key] = {
            "id": key, "name": src.name[:160], "size": n,
            "extension": src.suffix.lower()[:16],
            "publication_date_claim": published, "publication_verified": False,
            "license_label": license_label,
            "execution_permission": "NEVER_EXECUTE_AS_CODE",
            "llm_upload_permission": "DENIED_BY_DEFAULT"
        }
        put_json(path / "manifest.json", manifest)
    return manifest["objects"][key]


def retrieve(root, key):
    path, manifest, scope = get_root(root)
    if not isinstance(key, str) or not SHA.fullmatch(key):
        raise ScopeError("INVALID_DATA_ID")
    item = manifest["objects"].get(key)
    if not isinstance(item, dict):
        raise ScopeError("DATA_ID_NOT_FOUND")
    obj = path / "objects" / (key + ".bin")
    if obj.is_symlink() or not obj.is_file() or obj.stat().st_size > MAX_IMPORT_BYTES:
        raise ScopeError("DATA_FILE_MISSING_OR_UNSAFE")
    with obj.open("rb") as f:
        blob = f.read(MAX_IMPORT_BYTES + 1)
    if len(blob) > MAX_IMPORT_BYTES or digest(blob) != key or len(blob) != item["size"]:
        raise ScopeError("DATA_HASH_MISMATCH")
    return blob, item, scope


def admission(item, scope):
    as_of = scope.get("as_of")
    if not as_of:
        return "NOT_HISTORICALLY_SCOPED"
    parse_date(as_of)
    date = item.get("publication_date_claim")
    if date is None:
        return "UNKNOWN_PUBLICATION_DATE"
    parse_date(date)
    if date > as_of:
        return "CLAIMED_RELEASE_AFTER_CUTOFF"
    return "CLAIMED_RELEASE_BY_CUTOFF_NOT_AUTHENTICATED"




# Bulk import is deliberately DATA-ONLY: skip hidden, credential-like and build folders.
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".ssh", "secrets"}
SKIP_NAME_WORDS = ("secret", "credential", "password", "apikey", "api_key", "token")


def ingest_directory(root, source_dir, *, dry_run=False, max_files=200, max_total_bytes=256*1024*1024):
    """Import researcher-selected tree without executing code or following directory symlinks.

    Preflight size/count before the first object write. No external paths are saved
    in the workspace. This is not an encrypted confidential storage service.
    """
    import os
    base = Path(source_dir).expanduser()
    if base.is_symlink() or not base.is_dir():
        raise ScopeError("IMPORT_DIRECTORY_INVALID")
    candidates, skipped = [], []
    total = 0
    for current, dirs, files in os.walk(base, topdown=True, followlinks=False):
        now = Path(current)
        dirs[:] = sorted(d for d in dirs
                         if not d.startswith(".") and d not in SKIP_DIRS
                         and not (now / d).is_symlink())
        for filename in sorted(files):
            f = now / filename
            relative = f.relative_to(base).as_posix()
            if filename.startswith(".") or any(w in filename.lower() for w in SKIP_NAME_WORDS):
                skipped.append({"file":relative,"reason":"HIDDEN_OR_POSSIBLE_SECRET"})
                continue
            if f.is_symlink() or not f.is_file():
                skipped.append({"file":relative,"reason":"SYMLINK_OR_NOT_REGULAR"})
                continue
            n=f.stat().st_size
            if n > MAX_IMPORT_BYTES:
                skipped.append({"file":relative,"reason":"FILE_EXCEEDS_32_MIB"})
                continue
            if len(candidates) >= max_files or total + n > max_total_bytes:
                raise ScopeError("IMPORT_COLLECTION_BUDGET_EXCEEDED")
            candidates.append((f,relative,n))
            total += n
    if dry_run:
        return {"status":"PREVIEW_ONLY","files":[{"file":r,"bytes":n} for _,r,n in candidates],
                "total_bytes":total,"skipped":skipped}
    imported=[]
    for f,relative,n in candidates:
        item=ingest(root,f)
        imported.append({"relative_label":relative,"data_id":item["id"],"bytes":n,
                         "publication_date_claim":"UNKNOWN","executed":False})
    return {"status":"IMPORTED_LOCAL_ONLY","imported":imported,
            "total_bytes":total,"skipped":skipped,
            "llm_uploads":0}


def list_inventory(root):
    _, manifest, scope = get_root(root)
    return {"schema": "econscope-inventory-v1",
            "scope": scope,
            "objects": [{**v, "historical_admission": admission(v, scope)}
                        for _, v in sorted(manifest["objects"].items())]}
