"""Explicit local recomputation of portable declarative comparisons.

Uses the current engine, not an independently implemented solver or clean OS.
Never evaluates imported source code or restores private execution approval.
"""
from decimal import localcontext
from pathlib import Path
import platform
import stat
import sys
import tempfile

from .study_core import ENGINE, calculate, dataset_references, numerical_assurance, check_processing_rights
from .study_runs import (_verified_comparison_snapshot, _comparison_contents,
                         engine_digest, put_file)
from .workspace import (ScopeError, canonical, initialize, put_json, ingest,
                        get_root, admission, MAX_IMPORT_BYTES, digest, load_json)


def compute_wdi_replication(directory, cutoff):
    """Two declared descriptive analyses, checked against separately parsed raw GDP.

    Decimal computes the report; Fraction independently checks all observations,
    intermediate operands and differences. Neither implementation is external
    replication, source authentication or historical information-set evidence.
    """
    from decimal import Context, Decimal
    from fractions import Fraction
    import json
    from . import wdi_snapshot

    review = wdi_snapshot.review_snapshot(directory, cutoff)
    raw_values = {}
    for source in review['sources']:
        raw = wdi_snapshot._read(Path(directory).resolve(strict=True), source['file'])
        # A separate JSON numeric path: no production Decimal parser is reused.
        document = json.loads(raw, parse_float=str, parse_int=str)
        for row in document[1]:
            raw_values[row['countryiso3code'], row['date']] = Fraction(row['value'])
    rows = []
    for country in ('USA', 'KOR'):
        cells = {cell['year']: cell for cell in review['cells'] if cell['country'] == country}
        for year in wdi_snapshot.YEARS:
            cell = cells[year]
            previous = cells[str(int(year) - 1)]['value_decimal'] if year != '2019' else None
            base = cells['2019']['value_decimal']
            with localcontext(Context(prec=64)):
                gdp = Decimal(cell['value_decimal'])
                annual = format(gdp - Decimal(previous), 'f') if previous is not None else 'NOT_APPLICABLE'
                total = format(gdp - Decimal(base), 'f')
            expected = raw_values[country, year]
            if (Fraction(cell['value_decimal']) != expected or
                Fraction(base) != raw_values[country, '2019'] or
                Fraction(total) != expected - raw_values[country, '2019'] or
                (previous is not None and
                 (Fraction(previous) != raw_values[country, str(int(year) - 1)] or
                  Fraction(annual) != expected - raw_values[country, str(int(year) - 1)]))):
                raise ScopeError('WDI_INDEPENDENT_ARITHMETIC_MISMATCH')
            rows.append({'country': country, 'year': year, 'gdp': cell['value_decimal'],
                         'previous_gdp': previous if previous is not None else 'NOT_APPLICABLE',
                         'base_gdp': base, 'annual_change': annual, 'change_since_2019': total,
                         'unit': 'current_USD', 'source_sha256': cell['source_sha256']})
    disclosures = [
        'Vintage: CURRENT_RETRIEVAL_ONLY_NOT_HISTORICAL_AS_OF.',
        'Historical information set: NOT_ESTABLISHED.',
        'Publisher authentication: NOT_ESTABLISHED; public HTTPS local digest only.',
        'Causal identification / real growth / forecast skill: NOT_ESTABLISHED.',
        'Indicator-specific rights clearance: NOT_ESTABLISHED.',
        'External independent science / clean installed rerun: NOT_ESTABLISHED.',
        'Observation uncertainty: UNKNOWN; no confidence intervals estimated.',
        'CJK rendering: NOT_RUN. Rendered-page QA: NOT_RUN. ASCII PDF only.',
        'Arithmetic check: separate Fraction implementation in the same process.',
        'Analysis templates are authored descriptions; researcher approval NOT_ESTABLISHED.',
    ]
    # Each standalone CSV row carries the same uncertainty and evidence limits
    # as the report; its meaning survives separation from the JSON manifest.
    for row in rows:
        row['disclosures'] = ' | '.join(disclosures)
    return {'schema': 'societybox-wdi-replication-v1',
            'status': 'DESCRIPTIVE_ARITHMETIC_REPLICATION_PASS',
            'arithmetic_crosscheck': 'PASS_SEPARATE_FRACTION_IMPLEMENTATION',
            'analyses': {'annual_change': 'GDP(year) - GDP(previous year); 2019 NOT_APPLICABLE',
                         'change_since_2019': 'GDP(year) - GDP(2019)'},
            'rows': rows, 'sources': review['sources'],
            'provenance_sha256': review['provenance_sha256'],
            'retrieved_utc': review['retrieved_utc'], 'cutoff': review['cutoff'],
            'rights': review['rights'], 'disclosures': disclosures,
            'data_sent_to_LLM': False}


def comparison_source_plan(folder):
    """Build an explicit local data-map template without reading any source.

    Only verified portable records supply requirements. No original workspace
    paths, source cells or free-text evidence descriptions enter this plan.
    Empty paths deliberately require a researcher to select local files.
    """
    _, comparison, records = _verified_comparison_snapshot(folder)
    required = {}
    rights = {}
    cutoffs = {}
    for role, record in records.items():
        spec = record["spec"]
        rights[role] = spec["export_rights"]
        cutoffs[role] = {key: record["scope"].get(key)
                         for key in ("as_of", "strict_historical")}
        for key in sorted(dataset_references(spec)):
            declarations = required.setdefault(key, [])
            for evidence_id, evidence in sorted(spec["evidence"].items()):
                if evidence.get("data_id") == key:
                    declarations.append({"run_role": role, "evidence_id": evidence_id,
                                         "type": evidence["type"],
                                         "export_rights": evidence["export_rights"],
                                         **({"processing_rights": evidence["processing_rights"]}
                                            if "processing_rights" in evidence else {})})
    contents = _comparison_contents(records, comparison)
    publication_requirements = {}
    for key, declarations in sorted(required.items()):
        roles = sorted({entry["run_role"] for entry in declarations
                        if cutoffs[entry["run_role"]]["as_of"]
                        and cutoffs[entry["run_role"]]["strict_historical"]})
        publication_requirements[key] = {
            "format": "YYYY-MM-DD",
            "required_for_strict_cutoff": bool(roles),
            "must_be_on_or_before": min(cutoffs[role]["as_of"] for role in roles) if roles else None,
            "required_by_runs": roles,
            "verification": "USER_DECLARED_NOT_AUTHENTICATED",
        }
    return {"schema": "societybox-portable-source-plan-v1",
            "status": "SOURCE_SELECTION_REQUIRED" if required else "NO_EXTERNAL_DATA_REQUIRED",
            "bundle_sha256": digest(canonical({name: digest(blob) for name, blob in contents.items()})),
            "data_map": {key: {"path": "", "published": None} for key in sorted(required)},
            "publication_requirements": publication_requirements,
            "source_requirements": dict(sorted(required.items())),
            "study_export_rights": rights, "cutoffs": cutoffs,
            "path_resolution": "EXPLICIT_PATHS_RELATIVE_TO_CURRENT_DIRECTORY",
            "raw_data_included": False,
            "export_rights_status": "RESEARCHER_DECLARED_NOT_LEGALLY_VERIFIED",
            "publication_claims": "USER_DECLARED_NOT_AUTHENTICATED",
            "receipt_authenticity": "NOT_ESTABLISHED",
            "engine_reexecution": "NOT_RUN", "independent_clean_rerun": "NOT_RUN",
            "scope": "LOCAL_MATCHING_ENGINE_NOT_INDEPENDENT_REPLICATION"}


def recompute_comparison_bundle(folder, *, source_data=None, receipt_output=None):
    """Recompute locally; optionally preserve a separate, unauthenticated receipt."""
    target = Path(receipt_output) if receipt_output is not None else None
    if target is not None and Path(folder).resolve() in (target.resolve(), *target.resolve().parents):
        raise ScopeError("STUDY_RECOMPUTATION_RECEIPT_INSIDE_BUNDLE")
    snapshot = _verified_comparison_snapshot(folder)
    runtime = _recomputation_runtime()
    result = _recompute_snapshot(snapshot, source_data=source_data, runtime=runtime)
    if target is not None:
        claims = {key: {"published": entry.get("published")}
                  for key, entry in (source_data or {}).items()}
        payload = {**_receipt_binding(snapshot, runtime=runtime), "status": result["status"],
                   "reason": result.get("reason"), "source_claims": claims,
                   "verified_data_ids": result.get("verified_data_ids", [])}
        receipt = {"schema": "societybox-local-recomputation-v1", "payload": payload,
                   "payload_sha256": digest(canonical(payload))}
        put_file(target, canonical(receipt) + b"\n")
    return result


def _recomputation_runtime():
    """Actual local attempt context, distinct from the projected original runs."""
    from .study_capabilities import execution_capabilities
    return {"python_version": sys.version, "implementation": platform.python_implementation(),
            "python_major": sys.version_info.major, "python_minor": sys.version_info.minor,
            "platform": platform.platform(), "decimal_precision": 28,
            "capabilities": execution_capabilities()}


def _matching_runtime_declarations(runtime):
    """Untrusted JSON declarations must be integers, even when floats compare equal."""
    expected = {"python_major": sys.version_info.major,
                "python_minor": sys.version_info.minor, "decimal_precision": 28}
    return isinstance(runtime, dict) and all(
        type(runtime.get(key)) is int and runtime[key] == value
        for key, value in expected.items())


def _matching_numerical_assurance(runtime):
    # Canonical bytes distinguish equal-valued JSON floats/booleans from the
    # integer policy fields. This checks disclosure, not numerical accuracy.
    return canonical(runtime.get("numerical_assurance")) == canonical(numerical_assurance())


def _receipt_binding(snapshot, *, runtime=None):
    _, comparison, records = snapshot
    contents = _comparison_contents(records, comparison)
    runtime = _recomputation_runtime() if runtime is None else runtime
    return {"bundle_sha256": digest(canonical({name: digest(blob)
                for name, blob in contents.items()})),
            "engine_code_sha256": engine_digest(),
            "recomputation_code_sha256": digest(Path(__file__).read_bytes()),
            "runtime": runtime, "runtime_sha256": digest(canonical(runtime))}


def inspect_recomputation_receipt(folder, receipt):
    """Check bindings only. Anyone can rewrite and rehash this unsigned record."""
    return _inspect_receipt(_verified_comparison_snapshot(folder), receipt)


def _inspect_receipt(snapshot, receipt):
    try:
        saved = load_json(Path(receipt))
        payload = saved["payload"]
        binding = _receipt_binding(snapshot)
        if (set(saved) != {"schema", "payload", "payload_sha256"}
            or saved["schema"] != "societybox-local-recomputation-v1"
            or digest(canonical(payload)) != saved["payload_sha256"]
            or set(payload) != set(binding) | {"status", "reason", "source_claims", "verified_data_ids"}
            or any(payload[k] != v for k, v in binding.items())):
            raise ScopeError("STUDY_RECOMPUTATION_RECEIPT_BINDING_MISMATCH")
        records = snapshot[2]
        data_ids = set()
        for record in records.values():
            data_ids.update(dataset_references(record["spec"]))
        claims = payload["source_claims"]
        passed = payload["status"] == "ENGINE_RECOMPUTATION_PASS"
        if (payload["status"] not in ("ENGINE_RECOMPUTATION_PASS", "RECOMPUTATION_NOT_RUN")
            or not isinstance(claims, dict) or not set(claims) <= data_ids
            or any(not isinstance(v, dict) or set(v) != {"published"}
                   or (v["published"] is not None and not isinstance(v["published"], str))
                   for v in claims.values())
            or (passed and (payload["reason"] is not None or set(claims) != data_ids
                            or payload["verified_data_ids"] != sorted(data_ids)))
            or (not passed and (not isinstance(payload["reason"], str)
                                or payload["verified_data_ids"] != []))):
            raise ScopeError("STUDY_RECOMPUTATION_RECEIPT_OUTCOME_INVALID")
        if passed:
            from .study_capabilities import execution_capabilities
            capabilities = execution_capabilities()
            for record in records.values():
                check_processing_rights(record["spec"])
                scope = record["scope"]
                if scope.get("as_of") and scope.get("strict_historical"):
                    for claim in claims.values():
                        availability = admission({"publication_date_claim": claim["published"]}, scope)
                        if availability != "CLAIMED_RELEASE_BY_CUTOFF_NOT_AUTHENTICATED":
                            raise ScopeError("STUDY_RECOMPUTATION_RECEIPT_OUTCOME_INVALID")
                if (record["engine"] != ENGINE or record["engine_code_sha256"] != binding["engine_code_sha256"]
                    or canonical(record["runtime"].get("capabilities")) != canonical(capabilities)
                    or not _matching_runtime_declarations(record["runtime"])
                    or not _matching_numerical_assurance(record["runtime"])):
                    raise ScopeError("STUDY_RECOMPUTATION_RECEIPT_OUTCOME_INVALID")
        return {"status": "UNAUTHENTICATED_LOCAL_RECORD", "recorded_outcome": payload["status"],
                "receipt_authenticity": "NOT_ESTABLISHED", "independent_clean_rerun": "NOT_RUN",
                "scope": "LOCAL_MATCHING_ENGINE_NOT_INDEPENDENT_REPLICATION",
                "publication_claims": "USER_DECLARED_NOT_AUTHENTICATED",
                "payload_sha256": saved["payload_sha256"], "payload": payload}
    except ScopeError as exc:
        if str(exc).startswith("STUDY_RECOMPUTATION_RECEIPT"):
            raise
        raise ScopeError("STUDY_RECOMPUTATION_RECEIPT_INVALID") from None
    except (OSError, ValueError, KeyError, TypeError, AttributeError, RecursionError):
        raise ScopeError("STUDY_RECOMPUTATION_RECEIPT_INVALID") from None


def _recompute_snapshot(snapshot, *, source_data=None, runtime=None):
    """Read only explicitly mapped local sources; publication claims stay unverified.

    source_data maps each referenced SHA256 to {"path": local_path,
    "published": optional_YYYY_MM_DD}. No bundle-provided source path is used.
    """
    assurance, _, records = snapshot
    runtime = _recomputation_runtime() if runtime is None else runtime
    result = {**assurance, "status": "RECOMPUTATION_NOT_RUN",
              "recomputation_runtime": runtime,
              "recomputation_runtime_sha256": digest(canonical(runtime)),
              "scope": "LOCAL_MATCHING_ENGINE_NOT_INDEPENDENT_REPLICATION"}
    data_ids = set()
    for record in records.values():
        data_ids.update(dataset_references(record["spec"]))
    # Validate the complete map before any early NOT_RUN return. Receipt
    # serialization consumes these declarations even when processing is denied.
    # This checks shape only; no supplied path is opened or staged here.
    supplied = {} if source_data is None else source_data
    if (not isinstance(supplied, dict) or not set(supplied) <= data_ids
        or any(not isinstance(v, dict) or "path" not in v
               or not set(v) <= {"path", "published"}
               or not isinstance(v["path"], (str, Path))
               or (v.get("published") is not None and not isinstance(v["published"], str))
               for v in supplied.values())):
        raise ScopeError("STUDY_PORTABLE_SOURCE_MAP_INVALID")
    for record in records.values():
        try:
            check_processing_rights(record["spec"])
        except ScopeError as exc:
            return {**result, "reason": exc.code}
    current_digest = engine_digest()
    for record in records.values():
        if record["engine"] != ENGINE or record["engine_code_sha256"] != current_digest:
            return {**result, "reason": "STUDY_ENGINE_CODE_DRIFT"}
        recorded_runtime = record["runtime"]
        if not _matching_runtime_declarations(recorded_runtime):
            return {**result, "reason": "STUDY_RUNTIME_ENVIRONMENT_DIFFERENT"}
        # Preserve JSON types as well as values in the execution contract.
        if canonical(recorded_runtime.get("capabilities")) != canonical(runtime["capabilities"]):
            return {**result, "reason": "STUDY_EXECUTION_CAPABILITIES_DIFFERENT"}
        if not _matching_numerical_assurance(recorded_runtime):
            return {**result, "reason": "STUDY_NUMERICAL_POLICY_DIFFERENT"}
    missing = data_ids - set(supplied)
    if missing:
        return {**result, "reason": "STUDY_PORTABLE_SOURCE_DATA_REQUIRED",
                "missing_data_ids": sorted(missing)}
    checked = {}
    # Only our own freshly created temporary workspace is written. Bundle scope
    # remains data; no paths in it are used as destinations or source locations.
    with tempfile.TemporaryDirectory(prefix="societybox-recompute-") as temp:
        root = Path(temp)
        initialize(root)
        # All source bytes are copied and verified before either calculation.
        # Temporary copies prevent later owner-file edits from changing a run.
        for key, entry in sorted(supplied.items()):
            try:
                source = Path(entry["path"]).expanduser()
                if (source.is_symlink() or not source.is_file()
                    or getattr(source.lstat(), "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT):
                    raise ScopeError("STUDY_PORTABLE_SOURCE_FILE_UNAVAILABLE")
                if source.stat().st_size > MAX_IMPORT_BYTES:
                    raise ScopeError("INPUT_TOO_LARGE")
                item = ingest(root, source, published=entry.get("published"))
            except OSError:
                raise ScopeError("STUDY_PORTABLE_SOURCE_FILE_UNAVAILABLE") from None
            if item["id"] != key:
                raise ScopeError("STUDY_PORTABLE_SOURCE_HASH_MISMATCH")
        _, manifest, _ = get_root(root)
        for record in records.values():
            scope = record["scope"]
            for key in sorted(data_ids):
                availability = admission(manifest["objects"][key], scope)
                if (scope.get("as_of") and scope.get("strict_historical")
                    and availability in ("UNKNOWN_PUBLICATION_DATE", "CLAIMED_RELEASE_AFTER_CUTOFF")):
                    return {**result, "reason": "STUDY_DATA_NOT_AVAILABLE_AT_CUTOFF",
                            "data_id": key, "availability": availability}
        for role, record in records.items():
            put_json(root / "scope.json", record["scope"])
            try:
                with localcontext() as context:
                    context.prec = 28
                    recomputed = calculate(record["spec"], root, record["overrides"])
            except ScopeError as exc:
                # No complete paired rerun exists after an engine rejection.
                # Preserve the structured reason in a separate NOT_RUN receipt;
                # partial calculations never become verified runs or source IDs.
                return {**result, "reason": exc.code}
            # Canonical bytes also distinguish booleans from integer step values.
            # Compare every output, including parameters, source lineage and time.
            if canonical(recomputed) != canonical(record["results"]):
                raise ScopeError("STUDY_PORTABLE_RECOMPUTATION_MISMATCH")
            checked[role] = {"trajectory_rows": len(recomputed["trajectory"]),
                             "events": len(recomputed["events"])}
    return {**result, "status": "ENGINE_RECOMPUTATION_PASS",
            "engine_reexecution": "PASS", "checked_runs": checked,
            "verified_data_ids": sorted(data_ids),
            "publication_claims": "USER_DECLARED_NOT_AUTHENTICATED"}
