"""Content-addressed research receipts, byte-level reruns and evidence-first exports."""
from __future__ import annotations

import ast
import csv
from copy import deepcopy
from decimal import Decimal, localcontext
import io
import json
import os
from pathlib import Path
import platform
import stat
import sys

from .study_core import (SCHEMA, ENGINE, validate, calculate, number, review_datasets,
                         temporal_assurance, dataset_references, numerical_assurance,
                         numerical_context, check_processing_rights, value_in_bounds)
from .study_claims import build_claim_graph, validate_claim_graph, _project_claim_graph
from .study_capabilities import execution_capabilities
from .workspace import (
    ScopeError, SHA, canonical, digest, get_root, load_json, put_json, retrieve
)

RUN_SCHEMA = "societybox-study-receipt-v1"
RECEIPT_ASSURANCE = {
    "operator_approval": "CLI_EXPLICIT_FLAG_AND_REVIEW_DIGEST_NOT_AUTHENTICATED_IDENTITY",
    "research_assurance": "COMPUTATION_CHECKED_NOT_PEER_REVIEWED",
    "file_execution": "DATA_ONLY_NO_UNTRUSTED_SOURCE_CODE",
}
EXPLANATIONS = {
    "STUDY_DATA_PROCESSING_RIGHTS_DENIED_OR_UNKNOWN": "Every dataset needs an explicit permitted local-processing declaration. A source has missing, denied or unresolved rights. Resolve its rights and review the changed study before execution; export permission cannot grant processing rights.",
    "STUDY_SCHEMA_INVALID": "A study needs title, question, inference, evidence, model and explicit limitations.",
    "STUDY_STATES_REQUIRED": "Define at least one state variable with initial value or an approved data-column aggregation.",
    "STUDY_RULE_MUST_COVER_ALL_STATES": "Every state requires a researcher-authored transition expression, including unchanged variables.",
    "STUDY_EVIDENCE_REF_REQUIRED": "Every state, parameter, rule and event must cite an explicit evidence or assumption record.",
    "STUDY_EVIDENCE_TRUST_ESCALATION": "Claims of independently verified provenance must be confirmed outside self-authored study files.",
    "STUDY_DATA_NOT_AVAILABLE_AT_CUTOFF": "The data lack a dated, supported availability claim at this historical cutoff.",
    "STUDY_DECIMAL_LITERAL_NEEDS_SOURCED_PARAMETER": "Replace decimal literals in rules with named parameters carrying units and evidence refs.",
    "STUDY_VARIABLE_NAME_INVALID": "Use simple ASCII variable IDs; the units and study descriptions may use Korean or other language.",
    "STUDY_TIME_UNIT_INVALID": "Use step, day, month, quarter, or year, aligned with the temporal resolution of actual sources.",
    "STUDY_OVERRIDE_UNDECLARED": "Interventions must vary a declared parameter and create a new run; do not silently edit model definitions.",
    "STUDY_LIMITATIONS_REQUIRED": "Explicit limitations are required; an appealing simulation is not independent validation."
}


def diagnose(spec):
    try:
        result = validate(spec)
        check_processing_rights(spec)
        return {"status":"STRUCTURAL_CHECK_PASS", "review":result,
                "ready_for_execution":True,
                "evidence_verification":"NOT_PERFORMED",
                "external_validity":"NOT_ESTABLISHED"}
    except ScopeError as exc:
        return {"status":"NEEDS_RESEARCH_INPUT","reason":exc.code,
                "explanation":EXPLANATIONS.get(exc.code,
                    "The model is underspecified or unsupported by the conservative calculation language."),
                "ready_for_execution":False}


def engine_digest():
    root=Path(__file__).parent
    parts=("workspace.py","study_units.py","study_core.py","study_runs.py","study_claims.py",
           "study_capabilities.py","study_csv_input.py")
    return digest(b"".join((name.encode("ascii")+b"\x00"+(root/name).read_bytes()) for name in parts))


def _review_assurance(scope, data_ids):
    """Policy labels for the read-only review, never a grant of approval."""
    return {"authorization": "NOT_GRANTED", "execution": "NOT_RUN",
            "temporal_assurance": temporal_assurance(scope, data_ids),
            "numerical_assurance": numerical_assurance(),
            "data_checks": "REFERENCED_BYTES_VERIFIED_NOT_RIGHTS_OR_EXTERNAL_VALIDITY"}


def execution_review(root, spec, overrides=None):
    """Read-only content binding, not authorization or authenticated identity.

    Bind the full inventory conservatively: even unrelated inventory edits
    require review again. Runs/reports do not invalidate a review.
    """
    validate(spec)
    overrides = {} if overrides is None else overrides
    if not isinstance(overrides, dict) or set(overrides) - set(spec["model"]["parameters"]):
        raise ScopeError("STUDY_OVERRIDE_UNDECLARED")
    for value in overrides.values():
        number(value)
    path, manifest, scope = get_root(root)
    data_ids = review_datasets(root, spec, scope)
    context = {"schema": "societybox-execution-review-v1",
               "workspace": str(path), "spec_sha256": digest(canonical(spec)),
               "scope": scope, "inventory": manifest, "overrides": deepcopy(overrides),
               "engine_code_sha256": engine_digest(), "data_ids": sorted(data_ids),
               "capabilities": execution_capabilities()}
    return {"context": context, "review_sha256": digest(canonical(context)),
            **_review_assurance(scope, data_ids)}


def execute(root, spec, overrides=None, approved=False, review_sha256=None):
    if approved is not True:
        raise ScopeError("STUDY_OPERATOR_APPROVAL_REQUIRED")
    if not isinstance(review_sha256, str) or not SHA.fullmatch(review_sha256):
        raise ScopeError("STUDY_REVIEW_DIGEST_REQUIRED")
    # Review, calculate and persist the same detached proposal. Caller edits,
    # including edits restored before the post-check, must not affect this run.
    # The digest check below rejects any change captured while making the copy.
    spec = deepcopy(spec)
    overrides = deepcopy(overrides)
    reviewed = execution_review(root, spec, overrides)
    if review_sha256 != reviewed["review_sha256"]:
        raise ScopeError("STUDY_REVIEW_STALE_OR_CHANGED")
    path, _, scope = get_root(root)
    outcome = calculate(spec, root, overrides)
    if execution_review(root, spec, overrides) != reviewed:
        raise ScopeError("STUDY_REVIEW_CHANGED_DURING_EXECUTION")
    record = {
        "schema":RUN_SCHEMA,
        "engine":ENGINE,
        "engine_code_sha256":engine_digest(),
        "runtime": {"python_major":sys.version_info.major,
                    "python_minor":sys.version_info.minor,
                    "decimal_precision":numerical_assurance()["policy"]["precision"],
                    "numerical_assurance":numerical_assurance(),
                    "capabilities":reviewed["context"]["capabilities"],
                    "platform":platform.system()},
        "spec":spec,
        "spec_sha256":digest(canonical(spec)),
        "scope":scope,
        "scope_sha256":digest(canonical(scope)),
        "overrides":overrides or {},
        "results":outcome,
        "execution_review":reviewed,
        **RECEIPT_ASSURANCE,
    }
    ident = digest(canonical(record))
    file=path/"runs"/(ident+".json")
    if file.is_symlink():
        raise ScopeError("STUDY_RUN_SYMLINK_FORBIDDEN")
    if file.exists():
        if load_json(file) != record:
            raise ScopeError("STUDY_RUN_EXISTING_RECORD_DIFFERS")
    else:
        put_json(file,record)
    return {"run_id":ident, "event_count":len(outcome["events"]),
            "final_state":outcome["trajectory"][-1]["state"],
            "record_path":str(file),
            "status":"CONDITIONAL_COMPUTATION_NOT_RESEARCH_VALIDATED"}


def verify(root, run_id):
    verified, _ = _verified_run_snapshot(root, run_id)
    return verified


def _verified_run_snapshot(root, run_id):
    """Return verification and the exact in-memory receipt that passed it."""
    if not isinstance(run_id,str) or not SHA.fullmatch(run_id):
        raise ScopeError("STUDY_RUN_ID_INVALID")
    path, _, scope=get_root(root)
    try:
        record=load_json(path/"runs"/(run_id+".json"))
    except ScopeError as exc:
        if exc.code == "MANIFEST_NOT_FOUND_OR_SYMLINK":
            raise ScopeError("STUDY_RUN_RECEIPT_UNAVAILABLE") from None
        raise
    except FileNotFoundError:
        # The receipt may disappear between load_json's checks and its open.
        raise ScopeError("STUDY_RUN_RECEIPT_UNAVAILABLE") from None
    if not isinstance(record, dict) or record.get("schema")!=RUN_SCHEMA:
        raise ScopeError("STUDY_RUN_RECEIPT_SCHEMA_INVALID")
    if digest(canonical(record))!=run_id:
        raise ScopeError("STUDY_RUN_RECEIPT_TAMPERED")
    if engine_digest()!=record.get("engine_code_sha256"):
        raise ScopeError("STUDY_ENGINE_CODE_DRIFT")
    # A matching content hash does not establish proposal container types.
    # Reject absent/scalar/array payloads before indexing or reading references.
    if not isinstance(record.get("spec"), dict):
        raise ScopeError("STUDY_RUN_SPEC_INVALID")
    if digest(canonical(record["spec"]))!=record.get("spec_sha256"):
        raise ScopeError("STUDY_MODEL_TAMPERED")
    if not isinstance(record.get("scope"), dict):
        raise ScopeError("STUDY_RUN_SCOPE_INVALID")
    if digest(canonical(record["scope"]))!=record.get("scope_sha256"):
        raise ScopeError("STUDY_SCOPE_RECEIPT_TAMPERED")
    if scope!=record["scope"]:
        raise ScopeError("STUDY_SCOPE_CHANGED")
    # None is an API default, not a valid recorded override object. Validate
    # before binding lookup even if an attacker has rehashed both records.
    if not isinstance(record.get("overrides"), dict):
        raise ScopeError("STUDY_RUN_OVERRIDES_INVALID")
    runtime = record.get("runtime")
    # Receipts are untrusted JSON, even with a matching content hash. Validate
    # the container and version types before comparing metadata: JSON floats
    # and booleans can compare equal to Python integers.
    if (not isinstance(runtime, dict) or
        type(runtime.get("python_major")) is not int or
        type(runtime.get("python_minor")) is not int or
        runtime.get("python_major")!=sys.version_info.major or
        runtime.get("python_minor")!=sys.version_info.minor or
        runtime.get("platform")!=platform.system()):
        raise ScopeError("STUDY_RUNTIME_ENVIRONMENT_DIFFERENT")
    expected_capabilities = execution_capabilities()
    binding = record.get("execution_review", {})
    if not isinstance(binding, dict) or not isinstance(binding.get("context"), dict):
        raise ScopeError("STUDY_EXECUTION_REVIEW_INCONSISTENT")
    context = binding.get("context", {})
    # Capability declarations are JSON contracts: 0/0.0 must not stand in for
    # false, even when a changed receipt and review have both been rehashed.
    if (canonical(record["runtime"].get("capabilities")) != canonical(expected_capabilities) or
            canonical(context.get("capabilities")) != canonical(expected_capabilities) or
            binding.get("review_sha256") != digest(canonical(context))):
        raise ScopeError("STUDY_EXECUTION_CAPABILITIES_DIFFERENT")
    # Compare the historical review to the recorded proposal, not today's
    # workspace inventory. Hashes alone cannot detect contradictory bindings
    # when both the review digest and the receipt ID have been recomputed.
    # Rehashed receipts remain untrusted input. Reference discovery assumes
    # validated evidence entries and dataset IDs; validate before traversing.
    validate(record["spec"])
    expected_binding = {
        "schema": "societybox-execution-review-v1",
        "spec_sha256": record["spec_sha256"],
        "overrides": record["overrides"],
        "scope": record["scope"],
        "engine_code_sha256": record["engine_code_sha256"],
        "data_ids": dataset_references(record["spec"]),
    }
    if any(key not in context or context[key] != value
           for key, value in expected_binding.items()):
        raise ScopeError("STUDY_EXECUTION_REVIEW_INCONSISTENT")
    # A matching proposal digest does not validate adjacent assurance labels.
    # Derive their permitted values from the recorded scope and engine policy;
    # the retained pre-execution review must still say NOT_GRANTED/NOT_RUN.
    expected_assurance = _review_assurance(record["scope"], expected_binding["data_ids"])
    if (any(key not in binding or canonical(binding[key]) != canonical(value)
            for key, value in expected_assurance.items()) or
            any(record.get(key) != value for key, value in RECEIPT_ASSURANCE.items())):
        raise ScopeError("STUDY_EXECUTION_REVIEW_INCONSISTENT")
    expected_numerical = numerical_assurance()
    precision = record["runtime"].get("decimal_precision")
    # Check the recorded runtime policy before executing the rerun. JSON floats
    # and booleans are not valid precision declarations, even if equal in Python.
    # Canonical comparison also enforces types inside the nested policy and
    # retained review above, rather than accepting 28.0 == 28 or False == 0.
    if (type(precision) is not int or
            precision != expected_numerical["policy"]["precision"] or
            canonical(record["runtime"].get("numerical_assurance")) != canonical(expected_numerical)):
        raise ScopeError("STUDY_NUMERICAL_POLICY_DIFFERENT")
    # Reject missing/non-object result payloads before doing any recalculation.
    # Objects still require exact comparison with the actual engine output.
    recorded_results = record.get("results")
    if not isinstance(recorded_results, dict):
        raise ScopeError("STUDY_RUN_RESULTS_INVALID")
    re_result=calculate(record["spec"],root,record["overrides"])
    # JSON result types are part of the engine contract: Python equality
    # otherwise accepts 0 == False and 1 == 1.0 in model-time coordinates.
    if canonical(re_result)!=canonical(recorded_results):
        raise ScopeError("STUDY_RECALCULATION_FAILED")
    return {"verified":True,"run_id":run_id,
            "scope":"COMPUTATIONAL_REEXECUTION_NOT_EMPIRICAL_OR_CAUSAL_VALIDATION",
            "checks":["receipt_hash","engine_code_hash","model_spec_hash",
                      "scope_hash","runtime","execution_review_binding",
                      "source_data_hash","reexecution"]}, record


def claim_graph(root, run_id, candidate=None):
    """Only expose lineage after independent receipt reexecution succeeds."""
    # Project and validate against the exact receipt that passed reexecution.
    # A second disk read could discard that snapshot during a concurrent edit.
    _, record = _verified_run_snapshot(root, run_id)
    graph = build_claim_graph(record, run_id)
    if candidate is not None:
        validate_claim_graph(candidate, record, run_id)
    return graph


def safe_md(s):
    """Literal inline text for CommonMark/GFM; original text stays in the receipt.

    Escape ampersands before HTML delimiters so supplied character references
    cannot change the displayed text. Flatten line breaks to keep the generated
    report structure, and escape inline markup (including GFM strikethrough).
    """
    if not isinstance(s,str):
        s=str(s)
    entities = {"&": "&amp;", "<": "&lt;", ">": "&gt;"}
    markup = "\\`*_{}[]()#!|~"
    # CommonMark/GFM cannot use intraword underscores as emphasis delimiters.
    # Keep policy codes and variable names readable in the Markdown source.
    return "".join(c if c == "_" and 0 < i < len(s) - 1
                   and s[i - 1].isalnum() and s[i + 1].isalnum() else
                   entities[c] if c in entities else
                   "\\" + c if c in markup else
                   " " if c in "\r\n" else c for i, c in enumerate(s))


def render_markdown(record, run_id):
    spec=record["spec"]
    model=spec["model"]
    result=record["results"]
    lines=[
        "# "+safe_md(spec["title"]),
        "",
        "Research question: "+safe_md(spec["question"]),
        "",
        "**Claim scope:** "+spec["inference"]+" — CONCLUSION ABOUT A DECLARED SIMULATION, not externally validated, not actual human history or a verified future prediction.",
        "",
        "## Evidence and assumptions (listed BEFORE results)",
        "",
        "Every item is a researcher/assistant declaration; the software has not authenticated any publisher or established the theory as true.",
        "Researcher text is displayed literally, with line breaks replaced by spaces; original text is retained in the run receipt. Markdown and HTML formatting from study fields is not interpreted.",
        ""
    ]
    for key,entry in sorted(spec["evidence"].items()):
        extra=[]
        if entry.get("locator"): extra.append("Locator: "+safe_md(entry["locator"]))
        if entry.get("data_id"): extra.append("Local file SHA256: "+entry["data_id"])
        lines += ["- **"+safe_md(key)+" / "+entry["type"]+"**: "+safe_md(entry["description"])+
                  (". "+"; ".join(extra) if extra else "")]
    lines += ["", "## Temporal assurance", "",
              "Only dataset-level user-declared publication timing can be checked. "
              "These statuses describe the recorded execution scope; they do not establish leakage-free historical prediction.", ""]
    lines += ["- " + key + ": " + safe_md(value) for key, value in
              temporal_assurance(record["scope"], dataset_references(spec)).items()]
    lines += ["", "## Execution capabilities", "",
              "Recorded tool context, not authenticated approval or measured network isolation. Data classification UNSPECIFIED is not permission to process sensitive data.", "",
              safe_md(json.dumps(record["runtime"].get("capabilities", {"status": "NOT_RECORDED"}), sort_keys=True)), ""]
    lines += ["", "## Numerical assurance", "",
              "Recorded fixed execution policy, not per-run error measurement. Rounded arithmetic does not establish numerical accuracy, convergence or stability.", "",
              safe_md(json.dumps(record["runtime"].get("numerical_assurance", {"status": "NOT_RECORDED"}), sort_keys=True)), ""]
    lines += ["Observed Decimal diagnostics (sticky flags across initialization, transitions and detectors; "
              "not error magnitudes or operation counts):", "",
              safe_md(json.dumps(result.get("numerical_diagnostics", {"status": "NOT_RECORDED"}), sort_keys=True)), ""]
    lines+=["","## Researcher-specified mechanisms and calculation","",
            "All transition expressions below came from the study, not from a fixed Society Box economic theory.",
            "All state transitions read the previous step's state simultaneously; no rule reads another rule's newly computed state in the same step.",
            "This is computational step ordering, not an established physical or causal time lag.", ""]
    for state,rule in sorted(model["rules"].items()):
        lines+=["- **"+state+" next step** = "+safe_md(rule["expr"])+
                ". Basis: "+", ".join(rule["evidence"])+
                ". Unit label: "+safe_md(model["states"][state]["unit"])]
    lines+=["","Calculation conditions: time unit = "+model["time_unit"]+
            ", steps = "+str(model["steps"])+
            ", parameters = "+safe_md(json.dumps(result["parameters"],ensure_ascii=False,sort_keys=True))+".",
            "",
            "## Conditional events and their preceding evidence",
            "",
            "Detectors use the post-transition state at steps 1 through "+str(model["steps"])+". "
            "Step 0 is initialization only; event detectors are not evaluated there. "
            "A satisfied detector is recorded at every matching step, not only on first crossing.",
            ""]
    if not result["events"]:
        lines+=["No event detector was satisfied at the evaluated post-transition steps 1 through "+str(model["steps"])+
                ". This does not assess the initial state or imply the event cannot occur outside the declared model.",""]
    for index, ev in enumerate(result["events"]):
        lines+=[f'<a id="claim-event-{index}"></a>',
                "### "+safe_md(ev["label"])+" (model step "+str(ev["step"])+")",
                "",
                "**Prior basis:** "+", ".join(ev["refs"])+".",
                "",
                "**Calculation:** the investigator-defined event test "+
                safe_md(ev["detector"])+" evaluated true with "+
                safe_md(json.dumps(ev["trigger_state"],ensure_ascii=False,sort_keys=True))+".",
                "",
                "**Conditional finding:** on the above **unverified assumptions and this model's calculations**, "+
                "the specified condition was met within the model at step "+
                str(ev["step"])+". It is not a confirmed real-world event.",
                ""]
    keys=sorted(model["states"])
    lines+=['<a id="computed-state-table"></a>', "## Computed state table","",
            "| Step | "+" | ".join(keys)+" |",
            "|---|"+"|".join("---" for _ in keys)+"|"]
    for t in result["trajectory"]:
        lines+=["| "+str(t["step"])+" | "+" | ".join(t["state"][k] for k in keys)+" |"]
    lines+=["","## Explicit study limitations",""]
    lines+=["- "+safe_md(x) for x in spec["limitations"]]
    lines+=["- Source dates and publisher identities were not independently checked.",
            "- Symbolic unit algebra was checked; physical unit meaning, model identification, robustness, calibration and out-of-sample assessment were not independently validated.",
            "- Results are contingent on the provided model; reproducible calculation does not prove external validity.",
            "",
            "## Reproduction evidence","",
            "Run ID: "+run_id,
            "",
            "Model SHA256: "+record["spec_sha256"],
            "",
            "Engine code SHA256: "+record["engine_code_sha256"],
            "",
            "ClaimGraph SHA256: "+digest(canonical(build_claim_graph(record,run_id))),
            "",
            "The replication bundle includes claim_graph.json: state cells and event anchors link through model definitions and effective parameters to declared evidence. This is computational lineage, not source authentication or causal validation.",
            "",
            "Reproduce using the unchanged data and engine: societybox study-verify --workspace WORKSPACE --run-id "+run_id,
            "",
            "The full declarative model is embedded in the run receipt. Privacy and redistribution rights require separate review.",
            ""]
    return "\n".join(lines)


def _validate_comparison_pair(baseline_id, alternative_id):
    """Reject invalid selections before reading or reexecuting either run."""
    if baseline_id == alternative_id:
        raise ScopeError("STUDY_COMPARE_REQUIRES_DIFFERENT_RUNS")
    for run_id in (baseline_id, alternative_id):
        if not isinstance(run_id, str) or not SHA.fullmatch(run_id):
            raise ScopeError("STUDY_RUN_ID_INVALID")


def compare(root, baseline_id, alternative_id, candidate=None):
    """Reexecute both runs and project exact paired computational lineage."""
    _validate_comparison_pair(baseline_id, alternative_id)
    _, a = _verified_run_snapshot(root, baseline_id)
    _, b = _verified_run_snapshot(root, alternative_id)
    result = _compare_records(a, b, baseline_id, alternative_id)
    if candidate is not None and canonical(candidate) != canonical(result):
        raise ScopeError("STUDY_COMPARISON_MISMATCH")
    return result


def _compare_records(a, b, baseline_id, alternative_id):
    """Derive a contrast; caller determines the input assurance level."""
    if baseline_id == alternative_id:
        raise ScopeError("STUDY_COMPARE_REQUIRES_DIFFERENT_RUNS")
    if (a["spec_sha256"]!=b["spec_sha256"] or
        a["scope_sha256"]!=b["scope_sha256"] or
        a["engine_code_sha256"]!=b["engine_code_sha256"]):
        raise ScopeError("STUDY_COMPARE_NOT_SAME_RESEARCH_DESIGN")
    parameters_a = a["results"]["parameters"]
    parameters_b = b["results"]["parameters"]
    changed = [key for key in sorted(parameters_a)
               if Decimal(parameters_a[key]) != Decimal(parameters_b[key])]
    if not changed:
        raise ScopeError("STUDY_COMPARE_PARAMETERS_NOT_CHANGED")
    av=a["results"]["trajectory"][-1]["state"]
    bv=b["results"]["trajectory"][-1]["state"]
    # Exact subtraction, even when opposite-signed 28-digit results need 29
    # digits. Own exponent bounds, traps and formatting as well as precision;
    # ambient Decimal settings must not change these content-addressed bytes.
    def difference(before, after):
        left, right = Decimal(before), Decimal(after)
        precision = max(left.adjusted(), right.adjusted()) - min(
            left.as_tuple().exponent, right.as_tuple().exponent) + 2
        if precision > 4096:
            raise ScopeError("STUDY_COMPARE_EXACT_DELTA_PRECISION_UNSUPPORTED")
        with localcontext(numerical_context()) as context:
            context.prec = max(28, precision)
            return str(right - left)

    graphs = [_project_claim_graph(record, ident)
              for record, ident in ((a, baseline_id), (b, alternative_id))]
    references = {role: {"run_id": ident, "claim_graph_sha256": digest(canonical(graph))}
                  for role, ident, graph in zip(
                      ("baseline", "alternative"), (baseline_id, alternative_id), graphs)}
    final_step = a["results"]["trajectory"][-1]["step"]
    deltas = {key: difference(av[key], bv[key]) for key in sorted(av)}
    lineage = {
        "schema": "societybox-paired-comparison-lineage-v1",
        "scope": "COMPUTATIONAL_CONTRAST_NOT_CAUSAL_IDENTIFICATION",
        "runs": references,
        "spec_sha256": a["spec_sha256"],
        "engine_code_sha256": a["engine_code_sha256"],
        "time_unit": a["spec"]["model"]["time_unit"],
        "parameter_contrasts": {
            key: {"baseline": parameters_a[key], "alternative": parameters_b[key],
                  "changed": key in changed, "graph_node": "parameter/" + key}
            for key in sorted(parameters_a)},
        "final_cells": {
            key: {"baseline": av[key], "alternative": bv[key], "delta": deltas[key],
                  "operation": "alternative_minus_baseline",
                  "unit": a["spec"]["model"]["states"][key]["unit"],
                  "step": final_step, "graph_node": f"cell/{final_step}/{key}"}
            for key in sorted(av)},
        "event_counts": {
            role: {"count": len(record["results"]["events"]),
                   "graph_nodes": [f"event/{i}" for i in range(len(record["results"]["events"]))]}
            for role, record in (("baseline", a), ("alternative", b))}}
    result = {"comparison_scope":"SAME_USER_AUTHORED_MODEL_PARAMETER_INTERVENTION",
            "baseline_run_id":baseline_id,"alternative_run_id":alternative_id,
            "baseline_overrides":a["overrides"],"alternative_overrides":b["overrides"],
            "final_deltas":deltas,
            "event_count_baseline":len(a["results"]["events"]),
            "event_count_alternative":len(b["results"]["events"]),
            "lineage": lineage,
            "limitations":"Computed model contrasts are not real-world causal effects."}
    return result


def put_file(path, data, *, staging_directory=None):
    """Publish complete bytes atomically, without replacing an existing file.

    Stage on the destination filesystem and link only after flush/fsync. Bundle
    callers stage in their parent directory so abandoned temporary files cannot
    contaminate the portable artifact or block a retry. No abandoned files are
    removed automatically. A
    failed or interrupted write cannot expose a partial final artifact. Filesystems
    without hard-link support fail closed; replacement would lose conflict safety.
    """
    if path.is_symlink():
        raise ScopeError("STUDY_EXPORT_SYMLINK_FORBIDDEN")
    if path.exists():
        if path.read_bytes()!=data:
            raise ScopeError("STUDY_EXPORT_CONTENT_CONFLICT")
        return
    temporary = (path.parent if staging_directory is None else staging_directory) / (
        path.name + ".new-" + os.urandom(12).hex())
    created = False
    try:
        with temporary.open("xb") as stream:
            created = True
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            # A competing writer owns the destination. Never replace its bytes.
            if path.is_symlink():
                raise ScopeError("STUDY_EXPORT_SYMLINK_FORBIDDEN")
            if path.read_bytes() != data:
                raise ScopeError("STUDY_EXPORT_CONTENT_CONFLICT")
    finally:
        if created:
            temporary.unlink(missing_ok=True)


def check_export_rights(spec):
    """Fail closed on declarations, without claiming legal verification.

    Raw bytes are never exported. Their declaration is required to distinguish
    raw-data restrictions from permissions for metadata and computed outputs.
    Missing/malformed rights do not prevent private calculation or local reports.
    """
    kinds = {"study_text", "source_metadata", "derived_results", "raw_data"}

    def permitted(rights):
        if not isinstance(rights, dict) or set(rights) != kinds:
            return False
        for kind, entry in rights.items():
            if (not isinstance(entry, dict) or set(entry) != {"classification", "allow_export"}
                or entry["classification"] not in ("PUBLIC", "CONFIDENTIAL", "UNKNOWN")
                or type(entry["allow_export"]) is not bool):
                return False
            if kind != "raw_data" and (entry["classification"] != "PUBLIC" or not entry["allow_export"]):
                return False
        return True

    if not permitted(spec.get("export_rights")):
        raise ScopeError("STUDY_EXPORT_RIGHTS_REQUIRED_OR_DENIED")
    cleared_data = set()
    data_ids = {v["initial"]["data_id"] for v in spec["model"]["states"].values()
                if isinstance(v["initial"], dict)}
    for evidence in spec["evidence"].values():
        if evidence["type"] != "USER_ASSUMPTION" or "export_rights" in evidence:
            if not permitted(evidence.get("export_rights")):
                raise ScopeError("STUDY_EXPORT_RIGHTS_REQUIRED_OR_DENIED")
        if "data_id" in evidence:
            cleared_data.add(evidence["data_id"])
            data_ids.add(evidence["data_id"])
    if not data_ids <= cleared_data:
        raise ScopeError("STUDY_EXPORT_RIGHTS_REQUIRED_OR_DENIED")
    return sorted(data_ids)


def _preflight_bundle_target(target, contents):
    """Reject mixed/private destinations and conflicts before adding any files."""
    if target.is_symlink():
        raise ScopeError("STUDY_EXPORT_SYMLINK_FORBIDDEN")
    if target.exists():
        if not target.is_dir() or {p.name for p in target.iterdir()} - set(contents):
            raise ScopeError("STUDY_EXPORT_CONTENT_CONFLICT")
        for name, blob in contents.items():
            file = target / name
            if file.is_symlink():
                raise ScopeError("STUDY_EXPORT_SYMLINK_FORBIDDEN")
            if file.exists() and (not file.is_file() or file.read_bytes() != blob):
                raise ScopeError("STUDY_EXPORT_CONTENT_CONFLICT")


def replication_bundle(root, run_id):
    verified, record = _verified_run_snapshot(root, run_id)
    path,_,_=get_root(root)
    spec=record["spec"]
    data_ids=check_export_rights(spec)
    # Reexecution checks today's bytes; source declarations belong to the
    # saved review. Never substitute today's names, dates or license claims.
    inventory = record["execution_review"]["context"].get("inventory")
    objects = inventory.get("objects") if isinstance(inventory, dict) else None
    data_files = []
    for key in data_ids:
        item = objects.get(key) if isinstance(objects, dict) else None
        if (not isinstance(item, dict) or not isinstance(item.get("name"), str)
                or not isinstance(item.get("license_label", "UNKNOWN"), str)
                or (item.get("publication_date_claim") is not None
                    and not isinstance(item["publication_date_claim"], str))):
            raise ScopeError("STUDY_BUNDLE_RECORDED_DATA_MANIFEST_INVALID")
        data_files.append({"id": key, "name": item["name"],
                           "publication_claim": item.get("publication_date_claim"),
                           "license_label": item.get("license_label", "UNKNOWN"),
                           "publisher_authenticated": False, "raw_data_included": False})
    dirname=path/"replication_bundles"
    if dirname.is_symlink(): raise ScopeError("STUDY_EXPORT_SYMLINK_FORBIDDEN")
    target=dirname/run_id
    if target.is_symlink(): raise ScopeError("STUDY_EXPORT_SYMLINK_FORBIDDEN")
    records=record["results"]["trajectory"]
    variables=sorted(spec["model"]["states"])
    output=io.StringIO(newline="")
    writer=csv.writer(output,lineterminator="\n")
    writer.writerow(["model_step"]+variables)
    for row in records:
        writer.writerow([row["step"]]+[row["state"][x] for x in variables])
    public_rights="NO_RAW_DATA_INCLUDED_REVIEW_LICENSE" if data_ids else "NO_EXTERNAL_DATA_DEPENDENCY"
    manifest_doc={"schema":"societybox-replication-manifest-v1",
        "run_id":run_id,"model_sha256":record["spec_sha256"],
        "engine_sha256":record["engine_code_sha256"],
        "data_files":data_files,
        "data_availability":public_rights,
        "export_rights_status":"RESEARCHER_DECLARED_NOT_LEGALLY_VERIFIED",
        "raw_receipt_included":False,
        "execution_environment":record["runtime"],
        "simulation_condition":"NOT_EMPIRICALLY_VALIDATED",
        "reproduce_instructions":"Initialize an empty local workspace, restore legally available input files by SHA, then run societybox study-run with study.json and approved model under the pinned engine."}
    rr=render_markdown(record,run_id)
    readme="\n".join([
        "# Society Box replication bundle",
        "",
        "This is a **computationally verified conditional study**, not an AEA-certified deposit or a peer-reviewed article.",
        "",
        "## Data and Code Availability Statement",
        "",
        ("This simulation requires no original research data." if not data_ids else
         "Raw researcher data are NOT included. Data-access arrangements, original sources and sharing rights must be documented before public release."),
        "",
        "See manifest.json for model/run/dataset SHA256 and environment information.",
        "Source names, publication claims and license labels in manifest.json are declarations retained from the recorded execution review, not today's inventory or authenticated publisher/legal evidence.",
        "Study source: study.json; output: trajectory.csv, events.json, report.md, claim_graph.json.",
        "run_record.json is an exported projection, not the original content-addressed receipt. Private execution-review workspace/inventory context is omitted; verify the original only in its local workspace.",
        "Export permissions and confidentiality classifications are researcher declarations, not legal clearance or anonymization checks. Raw bytes remain excluded regardless of declared permission.",
        "",
        "To reproduce in a fresh workspace: install the pinned Society Box engine; initialize a workspace; provide matching source-data hashes when needed; execute study.json with reviewed parameters, then compare trajectory.csv and events.json.",
        "",
        "The current package does not include an independently pinned full software environment image or any claim of DOI.",
        "No journal or university has reviewed this bundle.",
        ""
    ])
    exported_record={k:v for k,v in record.items() if k != "execution_review"}
    exported_record["schema"]="societybox-exported-study-record-v1"
    exported_record["original_run_id"]=run_id
    exported_record["private_execution_review_included"]=False
    contents={
       "study.json":canonical(spec)+b"\n",
       "run_record.json":canonical(exported_record)+b"\n",
       "trajectory.csv":output.getvalue().encode("utf-8"),
       "events.json":canonical(record["results"]["events"])+b"\n",
       "claim_graph.json":canonical(build_claim_graph(record,run_id))+b"\n",
       "manifest.json":canonical(manifest_doc)+b"\n",
       "report.md":rr.encode("utf-8"),
       "README.md":readme.encode("utf-8")
    }
    _preflight_bundle_target(target, contents)
    dirname.mkdir(exist_ok=True)
    target.mkdir(exist_ok=True)
    for filename,blob in contents.items():
        put_file(target/filename,blob,staging_directory=dirname)
    return {"path":str(target),"run_id":run_id,
            "files":{k:digest(v) for k,v in sorted(contents.items())},
            "raw_data_included":False,"data_access_status":public_rights,
            "study_status":verified["scope"]}


def comparison_bundle(root, baseline_id, alternative_id):
    """Export a deterministic, declaration-gated computational contrast.

    Both rights checks precede output creation, including single-run exports.
    This does not establish legal clearance or an independent clean rerun.
    """
    _validate_comparison_pair(baseline_id, alternative_id)
    path, _, _ = get_root(root)
    records = {}
    for role, ident in (("baseline", baseline_id), ("alternative", alternative_id)):
        _, records[role] = _verified_run_snapshot(root, ident)
        check_export_rights(records[role]["spec"])
    # Project records and contrast from the exact same reexecuted snapshots.
    # Reloading receipts (including through compare) can mix different bytes
    # under one original run ID if the workspace changes during verification.
    comparison = _compare_records(records["baseline"], records["alternative"],
                                  baseline_id, alternative_id)
    comparison_id = digest(canonical(comparison))
    contents = _comparison_contents(records, comparison)
    folder = path / "comparison_bundles"
    if folder.is_symlink():
        raise ScopeError("STUDY_EXPORT_SYMLINK_FORBIDDEN")
    target = folder / comparison_id
    if target.is_symlink():
        raise ScopeError("STUDY_EXPORT_SYMLINK_FORBIDDEN")
    _preflight_bundle_target(target, contents)
    folder.mkdir(exist_ok=True)
    target.mkdir(exist_ok=True)
    for name, blob in contents.items():
        put_file(target / name, blob, staging_directory=folder)
    return {"path": str(target), "comparison_id": comparison_id,
            "files": {name: digest(blob) for name, blob in sorted(contents.items())},
            "raw_data_included": False, "independent_clean_rerun": "NOT_RUN"}


def _comparison_contents(records, comparison):
    """Canonical v1 bundle bytes from supplied records, without I/O or execution."""
    baseline_id, alternative_id = comparison["baseline_run_id"], comparison["alternative_run_id"]
    comparison_id = digest(canonical(comparison))
    contents = {"comparison.json": canonical(comparison) + b"\n",
                "study.json": canonical(records["baseline"]["spec"]) + b"\n"}
    for role, ident in (("baseline", baseline_id), ("alternative", alternative_id)):
        record = records[role]
        projection = {k: v for k, v in record.items() if k != "execution_review"}
        projection.update(schema="societybox-exported-study-record-v1",
                          original_run_id=ident, private_execution_review_included=False)
        contents[role + "_record.json"] = canonical(projection) + b"\n"
        contents[role + "_claim_graph.json"] = canonical(_project_claim_graph(record, ident)) + b"\n"
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["variable", "unit", "model_step", "baseline", "alternative", "alternative_minus_baseline"])
    for variable, cell in sorted(comparison["lineage"]["final_cells"].items()):
        writer.writerow([variable, cell["unit"], cell["step"], cell["baseline"], cell["alternative"], cell["delta"]])
    contents["final_deltas.csv"] = output.getvalue().encode("utf-8")
    contents["README.md"] = (
        "# Society Box paired computational comparison\n\n"
        "Read study.json evidence and assumptions first, then the run graphs and records, "
        "then comparison.json and final_deltas.csv. Deltas are exact alternative minus baseline.\n\n"
        "This is a conditional model contrast, not empirical evidence or causal identification. "
        "Export rights are researcher declarations, not legally verified clearance. "
        "Raw data and private execution-review context are excluded. Exported records are projections, "
        "not original content-addressed receipts.\n\n"
        "Independent clean-environment rerun: NOT_RUN. To attempt reproduction, obtain the matching "
        "engine and legally available source data identified in study.json, initialize a fresh workspace, "
        "review study.json separately with each record's overrides, explicitly approve and execute each, "
        "then compare results and regenerate the contrast. Original receipt IDs depend on private context "
        "and are not expected to match. No environment image or source data is bundled.\n"
    ).encode("utf-8")
    manifest = {"schema": "societybox-comparison-manifest-v1", "comparison_id": comparison_id,
                "runs": comparison["lineage"]["runs"],
                "files": {name: digest(blob) for name, blob in sorted(contents.items())},
                "raw_data_included": False, "private_execution_review_included": False,
                "export_rights_status": "RESEARCHER_DECLARED_NOT_LEGALLY_VERIFIED",
                "independent_clean_rerun": "NOT_RUN"}
    contents["manifest.json"] = canonical(manifest) + b"\n"
    return contents


def verify_comparison_bundle(folder):
    """Read-only v1 artifact consistency; never execute code or access source data.

    This cannot authenticate original IDs after private context was removed.
    Consistently rewritten records/results are not independently detected.
    """
    return _verified_comparison_snapshot(folder)[0]


def _verified_comparison_snapshot(folder):
    """Return assurance, contrast and records from validated in-memory bytes."""
    names = {"study.json", "baseline_record.json", "alternative_record.json",
             "baseline_claim_graph.json", "alternative_claim_graph.json",
             "comparison.json", "final_deltas.csv", "README.md", "manifest.json"}
    try:
        folder = Path(folder)
        if (folder.is_symlink() or not folder.is_dir()
            or getattr(folder.lstat(), "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
            or {p.name for p in folder.iterdir()} != names):
            raise ScopeError("STUDY_COMPARISON_BUNDLE_FILES_INVALID")
        blobs = {}
        for name in sorted(names):
            file = folder / name
            if (file.is_symlink() or not file.is_file()
                or getattr(file.lstat(), "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT):
                raise ScopeError("STUDY_COMPARISON_BUNDLE_FILES_INVALID")
            # Bound offline input allocation; no manifest-controlled file paths.
            with file.open("rb") as stream:
                blob = stream.read(32 * 1024 * 1024 + 1)
            if len(blob) > 32 * 1024 * 1024:
                raise ScopeError("STUDY_COMPARISON_BUNDLE_FILE_TOO_LARGE")
            blobs[name] = blob
        manifest = json.loads(blobs["manifest.json"])
        hashes = {name: digest(blob) for name, blob in blobs.items() if name != "manifest.json"}
        if manifest["files"] != hashes:
            raise ScopeError("STUDY_COMPARISON_BUNDLE_HASH_MISMATCH")
        records, ids = {}, []
        fields = {"schema", "engine", "engine_code_sha256", "runtime", "spec", "spec_sha256",
                  "scope", "scope_sha256", "overrides", "results", "operator_approval",
                  "research_assurance", "file_execution", "original_run_id",
                  "private_execution_review_included"}
        for role in ("baseline", "alternative"):
            record = json.loads(blobs[role + "_record.json"])
            if (set(record) != fields or record["schema"] != "societybox-exported-study-record-v1"
                or record["private_execution_review_included"] is not False
                or not isinstance(record["original_run_id"], str)
                or not SHA.fullmatch(record["original_run_id"])):
                raise ScopeError("STUDY_COMPARISON_BUNDLE_RECORD_INVALID")
            validate(record["spec"])
            check_export_rights(record["spec"])
            _validate_portable_parameters(record)
            _validate_portable_trajectory(record)
            _validate_portable_initial_states(record)
            _validate_portable_source_lineage(record)
            _validate_portable_events(record)
            if (record["spec_sha256"] != digest(canonical(record["spec"]))
                or record["scope_sha256"] != digest(canonical(record["scope"]))):
                raise ScopeError("STUDY_COMPARISON_BUNDLE_RECORD_INVALID")
            records[role] = record
            ids.append(record["original_run_id"])
        comparison = _compare_records(records["baseline"], records["alternative"], *ids)
        expected = _comparison_contents(records, comparison)
        if blobs != expected:
            raise ScopeError("STUDY_COMPARISON_BUNDLE_CONTENT_MISMATCH")
        assurance = {"status": "ARTIFACT_CONSISTENCY_PASS",
                "comparison_id": digest(canonical(comparison)),
                "final_deltas": comparison["final_deltas"],
                "engine_reexecution": "NOT_RUN", "independent_clean_rerun": "NOT_RUN",
                "original_receipt_authenticity": "NOT_ESTABLISHED",
                "external_validity": "NOT_ESTABLISHED"}
        return assurance, comparison, records
    except ScopeError:
        raise
    except (OSError, ValueError, ArithmeticError, KeyError, TypeError, IndexError, AttributeError,
            SyntaxError, RecursionError, OverflowError):
        raise ScopeError("STUDY_COMPARISON_BUNDLE_INVALID") from None


def _validate_portable_parameters(record):
    """Bind effective parameter strings to defaults and explicit interventions.

    No transitions or source reads are needed for this declaration check. It
    does not authenticate the overrides or prove they generated the trajectory.
    """
    declared = record["spec"]["model"]["parameters"]
    overrides = record["overrides"]
    parameters = record["results"].get("parameters")
    if (not isinstance(overrides, dict) or set(overrides) - set(declared)
        or not isinstance(parameters, dict) or set(parameters) != set(declared)):
        raise ScopeError("STUDY_COMPARISON_BUNDLE_PARAMETERS_INVALID")
    try:
        with localcontext(numerical_context()):
            expected = {key: str(number(overrides.get(key, entry["value"])))
                        for key, entry in declared.items()}
            # Match the actual engine serialization, including Decimal scale;
            # Python numeric equality must not accept JSON int/bool results.
            if canonical(parameters) != canonical(expected):
                raise ScopeError("STUDY_COMPARISON_BUNDLE_PARAMETERS_INVALID")
    except (ScopeError, ArithmeticError):
        raise ScopeError("STUDY_COMPARISON_BUNDLE_PARAMETERS_INVALID") from None


def _validate_portable_trajectory(record):
    """Check declared cell coverage, timing, numeric types and state bounds.

    Rebuilding hashes/graphs cannot make missing or repeated model steps a
    coherent trajectory. This structural check does not authenticate results.
    The study schema must already have been validated by the caller.
    """
    model, results = record["spec"]["model"], record["results"]
    rows = results.get("trajectory")
    if (type(results.get("steps")) is not int
        or results["steps"] != model["steps"]
        or results.get("time_unit") != model["time_unit"]
        or not isinstance(rows, list) or len(rows) != model["steps"] + 1):
        raise ScopeError("STUDY_COMPARISON_BUNDLE_TRAJECTORY_INVALID")
    for step, row in enumerate(rows):
        if (not isinstance(row, dict) or set(row) != {"step", "state"}
            or type(row["step"]) is not int or row["step"] != step
            or not isinstance(row["state"], dict)
            or set(row["state"]) != set(model["states"])):
            raise ScopeError("STUDY_COMPARISON_BUNDLE_TRAJECTORY_INVALID")
        for key, value in row["state"].items():
            if type(value) is not str:
                raise ScopeError("STUDY_COMPARISON_BUNDLE_TRAJECTORY_INVALID")
            try:
                # Execution emits decimal strings, never JSON numeric scalars.
                # Reuse its finite/magnitude and declared state-bound contract,
                # owning parser traps/flags. This does not recompute the model.
                with localcontext(numerical_context()):
                    value_in_bounds(number(value), model["states"][key])
            except (ScopeError, ArithmeticError):
                raise ScopeError("STUDY_COMPARISON_BUNDLE_TRAJECTORY_INVALID") from None


def _validate_portable_initial_states(record):
    """Bind scalar step-zero cells to the engine's declared initialization.

    Trajectory structure/types must already be checked. Dataset aggregations
    need mapped-source recomputation and are deliberately not inferred here.
    This declaration check does not authenticate results or run transitions.
    """
    initial_state = record["results"]["trajectory"][0]["state"]
    with localcontext(numerical_context()):
        for key, entry in record["spec"]["model"]["states"].items():
            if not isinstance(entry["initial"], dict):
                if initial_state[key] != str(number(entry["initial"])):
                    raise ScopeError("STUDY_COMPARISON_BUNDLE_INITIAL_STATE_INVALID")


def _validate_portable_source_lineage(record):
    """Bind initializer provenance without reading or authenticating sources.

    Dataset row counts and publication claims remain recorded observations;
    their truth requires source reexecution. Content-addressed IDs, initializer
    declarations and the engine's unverified-publisher label are checkable here.
    """
    states = record["spec"]["model"]["states"]
    lineage = record["results"].get("source_lineage")
    error = "STUDY_COMPARISON_BUNDLE_SOURCE_LINEAGE_INVALID"
    if not isinstance(lineage, dict) or set(lineage) != set(states):
        raise ScopeError(error)
    for key, entry in states.items():
        initial, observed = entry["initial"], lineage[key]
        if not isinstance(initial, dict):
            expected = {"kind": "USER_ASSUMPTION", "refs": entry["evidence"]}
        else:
            if (not isinstance(observed, dict)
                or type(observed.get("rows")) is not int
                or not 1 <= observed["rows"] <= 10000):
                raise ScopeError(error)
            availability = observed.get("availability")
            allowed = {"NOT_HISTORICALLY_SCOPED"}
            if record["scope"].get("as_of"):
                allowed = {"UNKNOWN_PUBLICATION_DATE", "CLAIMED_RELEASE_AFTER_CUTOFF",
                           "CLAIMED_RELEASE_BY_CUTOFF_NOT_AUTHENTICATED"}
                if record["scope"].get("strict_historical"):
                    allowed = {"CLAIMED_RELEASE_BY_CUTOFF_NOT_AUTHENTICATED"}
            if not isinstance(availability, str) or availability not in allowed:
                raise ScopeError(error)
            expected = {"data_id": initial["data_id"], "column": initial["column"],
                        "aggregate": initial["aggregate"], "rows": observed["rows"],
                        "availability": availability, "publisher_verified": False,
                        "sha256": initial["data_id"]}
        if canonical(observed) != canonical(expected):
            raise ScopeError(error)


def _validate_portable_events(record):
    """Bind observations to declared detectors and recorded post-step cells.

    This checks event structure and lineage only, without evaluating predicates
    or establishing that all and only triggered events were recorded.
    """
    model, results = record["spec"]["model"], record["results"]
    events = results.get("events")
    if not isinstance(events, list) or len(events) > 600:
        raise ScopeError("STUDY_COMPARISON_BUNDLE_EVENTS_INVALID")
    definitions = {event["id"]: (index, event)
                   for index, event in enumerate(model["events"])}
    previous = (0, -1)
    for event in events:
        if (not isinstance(event, dict)
            or set(event) != {"step", "event_id", "label", "detector", "refs",
                              "trigger_state", "event_scope"}
            or type(event["step"]) is not int
            or not 1 <= event["step"] <= model["steps"]
            or not isinstance(event["event_id"], str)
            or event["event_id"] not in definitions):
            raise ScopeError("STUDY_COMPARISON_BUNDLE_EVENTS_INVALID")
        index, definition = definitions[event["event_id"]]
        position = (event["step"], index)
        names = {node.id for node in ast.walk(ast.parse(definition["when"], mode="eval"))
                 if isinstance(node, ast.Name)} & set(model["states"])
        expected = {"step": event["step"], "event_id": definition["id"],
                    "label": definition["label"], "detector": definition["when"],
                    "refs": definition["evidence"],
                    "trigger_state": {key: results["trajectory"][event["step"]]["state"][key]
                                      for key in sorted(names)},
                    "event_scope": "INSIDE_DECLARED_MODEL_ONLY"}
        if position <= previous or canonical(event) != canonical(expected):
            raise ScopeError("STUDY_COMPARISON_BUNDLE_EVENTS_INVALID")
        previous = position


def report(root, run_id):
    v,record=_verified_run_snapshot(root,run_id)
    path,_,_=get_root(root)
    contents=render_markdown(record,run_id)
    folder=path/"study_reports"
    if folder.is_symlink():
        raise ScopeError("STUDY_REPORT_FOLDER_SYMLINK")
    folder.mkdir(exist_ok=True)
    filename=folder/(run_id+".md")
    if filename.is_symlink():
        raise ScopeError("STUDY_REPORT_SYMLINK")
    if filename.exists():
        if filename.read_text(encoding="utf-8")!=contents:
            raise ScopeError("STUDY_REPORT_CONFLICT")
    else:
        try:
            put_file(filename, contents.replace("\n", os.linesep).encode("utf-8"))
        except ScopeError as exc:
            code = {"STUDY_EXPORT_SYMLINK_FORBIDDEN": "STUDY_REPORT_SYMLINK",
                    "STUDY_EXPORT_CONTENT_CONFLICT": "STUDY_REPORT_CONFLICT"}.get(exc.code)
            if code is None:
                raise
            raise ScopeError(code) from exc
    return {"file":str(filename),"run_id":run_id,
            "sha256":digest(filename.read_bytes()),
            "verification":v["scope"],
            "output_type":"EVIDENCE_FIRST_RESEARCH_DRAFT_NOT_PEER_REVIEWED"}
