"""Bounded byte-level .mod review. Never parse, import or execute model code."""
import hashlib
import os
from pathlib import Path
import re
import stat

from .workspace import ScopeError

MAX_MOD_BYTES = 1024 * 1024
_DEVICE = re.compile(r"(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?", re.I)


def _snapshot(path):
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
        raise ScopeError("MOD_LINK_OR_REPARSE_FORBIDDEN")
    return info


def _identity(info):
    # Windows CPython 3.12 lstat/fstat expose different ctime semantics.
    # Compare ctime only between snapshots from the same API below.
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size,
            info.st_mtime_ns)


def review_mod(directory, relative_path, expected_sha256=None):
    """Review a snapshot, not a future executable path or an approval token.

    Path/identity checks detect ordinary changes; they are not an OS sandbox or
    a guarantee against concurrent hostile filesystem mutation.
    """
    if (not isinstance(relative_path, str) or not relative_path or
            len(relative_path) > 1024 or "\\" in relative_path):
        raise ScopeError("MOD_RELATIVE_PATH_INVALID")
    parts = relative_path.split("/")
    if (len(parts) > 16 or any(
            part in ("", ".", "..") or part.endswith((".", " ")) or
            _DEVICE.fullmatch(part) or
            any(ord(c) < 32 or ord(c) == 127 or c in ':<>"|?*' for c in part)
            for part in parts) or not parts[-1].lower().endswith(".mod")):
        raise ScopeError("MOD_RELATIVE_PATH_INVALID")
    if expected_sha256 is not None and (
            not isinstance(expected_sha256, str) or
            not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)):
        raise ScopeError("MOD_EXPECTED_SHA256_INVALID")
    # UNC/device roots may contact remote filesystems; intake is local only.
    directory = os.fspath(directory)
    if not isinstance(directory, str) or not directory or directory.startswith(("\\\\", "//")):
        raise ScopeError("MOD_LOCAL_DIRECTORY_REQUIRED")
    root = Path(os.path.abspath(directory))
    try:
        # Check ancestors before resolving; resolve() alone would hide links.
        for parent in reversed((root, *root.parents)):
            if not stat.S_ISDIR(_snapshot(parent).st_mode):
                raise ScopeError("MOD_DIRECTORY_INVALID")
        path = root
        for part in parts[:-1]:
            path = path / part
            if not stat.S_ISDIR(_snapshot(path).st_mode):
                raise ScopeError("MOD_DIRECTORY_INVALID")
        path = path / parts[-1]
        before = _snapshot(path)
        if not stat.S_ISREG(before.st_mode):
            raise ScopeError("MOD_REGULAR_FILE_REQUIRED")
        if before.st_size > MAX_MOD_BYTES:
            raise ScopeError("MOD_TOO_LARGE")
        with path.open("rb") as stream:
            opened = os.fstat(stream.fileno())
            if _identity(opened) != _identity(before):
                raise ScopeError("MOD_CHANGED_DURING_READ")
            raw = stream.read(MAX_MOD_BYTES + 1)
            after = os.fstat(stream.fileno())
        if len(raw) > MAX_MOD_BYTES:
            raise ScopeError("MOD_TOO_LARGE")
        final = _snapshot(path)
        if (len(raw) != before.st_size or _identity(after) != _identity(before) or
                after.st_ctime_ns != opened.st_ctime_ns or
                _identity(final) != _identity(before) or
                final.st_ctime_ns != before.st_ctime_ns):
            raise ScopeError("MOD_CHANGED_DURING_READ")
    except OSError:
        raise ScopeError("MOD_INPUT_UNREADABLE") from None
    return review_mod_bytes(raw, relative_path, expected_sha256)


def review_mod_bytes(raw, relative_path, expected_sha256=None):
    """Validate preserved bytes without reopening a source path."""
    if len(raw) > MAX_MOD_BYTES:
        raise ScopeError("MOD_TOO_LARGE")
    if expected_sha256 is not None and (
            not isinstance(expected_sha256, str) or
            not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)):
        raise ScopeError("MOD_EXPECTED_SHA256_INVALID")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeError:
        raise ScopeError("MOD_UTF8_REQUIRED") from None
    if not text.strip():
        raise ScopeError("MOD_TEXT_REQUIRED")
    if any((ord(c) < 32 and c not in "\t\r\n") or ord(c) == 127 for c in text):
        raise ScopeError("MOD_CONTROL_CHARACTER_FORBIDDEN")
    sha = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and expected_sha256 != sha:
        raise ScopeError("MOD_SHA256_MISMATCH")
    return {
        "schema": "societybox-mod-intake-v1",
        "status": "REVIEWED_BYTES_ONLY",
        "artifact": {"relative_path": relative_path, "bytes": len(raw),
                     "sha256": sha, "encoding": "UTF-8"},
        "syntax": "NOT_CHECKED", "includes_and_macros": "NOT_RESOLVED",
        "source_rights": "NOT_VERIFIED", "approval": "NOT_GRANTED",
        "native_adapter": "NOT_IMPLEMENTED", "execution": "NOT_RUN",
        "native_reference_parity": "NOT_RUN", "os_isolation": "NOT_ESTABLISHED",
        "execution_status": "BLOCKED",
        "blockers": ["NATIVE_MOD_ADAPTER_NOT_IMPLEMENTED",
                     "OS_ISOLATION_NOT_ESTABLISHED", "NATIVE_REFERENCE_PARITY_NOT_RUN",
                     "SOURCE_RIGHTS_REVIEW_REQUIRED", "RESEARCHER_APPROVAL_REQUIRED"],
        "limitations": [
            "Byte validation does not prove valid Dynare syntax, model safety or scientific validity.",
            "No model text, macro, include, command or reference solver was executed or resolved.",
            "This unsigned snapshot grants no execution, LLM tool or data transfer authority.",
            "A later execution must independently bind bytes and all dependencies in proven OS isolation.",
            "Filesystem checks are not isolation against concurrent hostile filesystem mutation.",
        ],
    }


def review_stored_mod(workspace, artifact_id, expected_sha256=None):
    """Review hash/size-verified bytes; manifest labels confer no authority."""
    return inspect_stored_mod_source(workspace, artifact_id, expected_sha256)[0]


def inspect_stored_mod_source(workspace, artifact_id, expected_sha256=None):
    """Return review and inert UTF-8 text from the same verified byte snapshot.

    The optional leading UTF-8 BOM is omitted from display; the receipt hashes
    all original bytes. No includes, macros or model directives are resolved.
    """
    from .workspace import retrieve
    raw, item, _ = retrieve(workspace, artifact_id)
    if item.get("extension") != ".mod":
        raise ScopeError("MOD_STORED_EXTENSION_REQUIRED")
    if (item.get("execution_permission") != "NEVER_EXECUTE_AS_CODE" or
            item.get("llm_upload_permission") != "DENIED_BY_DEFAULT"):
        raise ScopeError("MOD_STORED_PERMISSIONS_INVALID")
    name = item.get("name")
    if not isinstance(name, str) or not name or len(name) > 160:
        raise ScopeError("MOD_STORED_NAME_INVALID")
    receipt = review_mod_bytes(raw, name, expected_sha256)
    receipt["artifact"].update({"artifact_id": artifact_id,
                                "storage": "VERIFIED_WORKSPACE_BYTES",
                                "execution_permission": "NEVER_EXECUTE_AS_CODE",
                                "llm_upload_permission": "DENIED_BY_DEFAULT"})
    return receipt, raw.decode("utf-8-sig")
