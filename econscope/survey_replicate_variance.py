"""Independent exact replicate-weight variance arithmetic for declared surveys.

Public statistical formulas only; not SUDAAN parity, a validated survey
sampling design, causal evidence, or a confidence interval. Explicitly separate
survey replicate weights from regression analytic weights and avoid using an
unavailable proprietary runtime or data.
"""
from fractions import Fraction as F
from hashlib import sha256
from pathlib import Path
import re

from .research_evidence import make_evidence
from .workspace import ScopeError, canonical, digest

_NUMBER = re.compile(r"-?(?:0|[1-9][0-9]{0,9})(?:/[1-9][0-9]{0,9})?\Z")
_SHA = re.compile(r"[a-f0-9]{64}\Z")
SCHEMA = "societybox-survey-replicate-mean-v1"
_METHODS = {"BRR_DECLARED", "DELETE_ONE_JACKKNIFE_DECLARED"}


def _bad(code):
    raise ScopeError("SURVEY_REPLICATE_" + code)


def _fraction(value):
    if type(value) is not str or not _NUMBER.fullmatch(value):
        _bad("NUMBER_INVALID")
    return F(value)


def _parse(q):
    if (type(q) is not dict or set(q) != {
            "schema", "values", "base_weights", "replicate_weights",
            "method", "unit", "source_declared_sha256", "rights"}
            or q["schema"] != SCHEMA or type(q["method"]) is not str
            or q["method"] not in _METHODS):
        _bad("SCHEMA_INVALID")
    rights = q["rights"]
    if type(rights) is not dict or set(rights) != {"local_processing", "report_export"}:
        _bad("RIGHTS_INVALID")
    if rights["local_processing"] != "PERMITTED" or rights["report_export"] != "PERMITTED":
        _bad("RIGHTS_REQUIRED")
    if (type(q["unit"]) is not str or not 1 <= len(q["unit"]) <= 100
            or any(ord(c) < 32 for c in q["unit"])):
        _bad("UNIT_INVALID")
    if type(q["source_declared_sha256"]) is not str or not _SHA.fullmatch(q["source_declared_sha256"]):
        _bad("SOURCE_INVALID")
    values, weights, reps = q["values"], q["base_weights"], q["replicate_weights"]
    if type(values) is not list or not 2 <= len(values) <= 500:
        _bad("SAMPLE_BUDGET")
    n = len(values)
    if (type(weights) is not list or len(weights) != n
            or type(reps) is not list or not 2 <= len(reps) <= 200
            or n * len(reps) > 25000
            or any(type(rep) is not list or len(rep) != n for rep in reps)):
        _bad("WEIGHT_DIMENSIONS")
    y = [_fraction(v) for v in values]
    w = [_fraction(v) for v in weights]
    rw = [[_fraction(v) for v in rep] for rep in reps]
    if (any(value <= 0 for value in w)
            or any(value < 0 for rep in rw for value in rep)
            or any(not any(value > 0 for value in rep) for rep in rw)):
        _bad("WEIGHT_INVALID")
    return y, w, rw


def _weighted_mean(y, weights):
    return sum((v * w for v, w in zip(y, weights)), F(0)) / sum(weights)


def validate(q):
    """Validate declared inputs without calculating or authenticating design."""
    _parse(q)


def calculate(q):
    y, weights, reps = _parse(q)
    source_code_sha256 = sha256(Path(__file__).read_bytes()).hexdigest()
    theta = _weighted_mean(y, weights)
    estimates = [_weighted_mean(y, rep) for rep in reps]
    count = len(estimates)
    if q["method"] == "BRR_DECLARED":
        # Classical BRR point-estimate center; correctness requires that the
        # supplied replicate design be balanced, which is NOT authenticated.
        variance = sum(((value - theta)**2 for value in estimates), F(0)) / count
        expanded = (sum((v*v for v in estimates), F(0))
                    - 2*theta*sum(estimates, F(0)) + count*theta*theta) / count
    else:
        # Delete-one jackknife center is the mean of replicate estimates, NOT
        # the full-sample estimate. No automatic stratum/PSU claim.
        center = sum(estimates, F(0)) / count
        variance = F(count-1, count) * sum(((v-center)**2 for v in estimates), F(0))
        pairwise = sum(((estimates[i]-estimates[j])**2
                        for i in range(count) for j in range(i+1, count)), F(0))
        expanded = F(count-1, count*count) * pairwise
    if variance != expanded or variance < 0:
        _bad("EXACT_ORACLE_PARITY")
    if sha256(Path(__file__).read_bytes()).hexdigest() != source_code_sha256:
        _bad("CODE_CHANGED_DURING_RUN")
    subject = digest(canonical(q))
    ev = make_evidence(
        subject_sha256=subject,
        producer={"tool_id": "societybox.replicate_mean", "version": "prototype-1",
                  "code_sha256": source_code_sha256},
        operation="EXACT_DECLARED_SURVEY_REPLICATE_MEAN_VARIANCE",
        authority="LOCAL_EXACT_ARITHMETIC",
        epistemic_state="DERIVED", proof_kind="EXACT_REPLICATE_ALGEBRAIC_IDENTITY",
        scope={"method": q["method"], "unit": q["unit"],
               "sample_count": len(y), "replicate_count": count},
        coverage={"status": "PARTIAL",
                  "unknown_dimensions": ["survey_sample_design",
                                         "replicate_balance_or_delete_one_design",
                                         "source_and_population_authentication",
                                         "statistical_confidence_coverage"]},
        result={"point_exact": str(theta), "variance_exact": str(variance)},
        parent_evidence_ids=[],
        limitations=["Input and replicate weights are unverified researcher declarations",
                     "BRR balance / strata / PSU / finite population corrections not validated",
                     "Variance arithmetic is not itself a certified design-based standard error"])
    return {"schema": "societybox-survey-replicate-result-v1",
            "status": "DECLARED_DESIGN_ARITHMETIC_ONLY",
            "method": q["method"], "unit": q["unit"], "input_sha256": subject,
            "point_exact": str(theta), "variance_exact": str(variance),
            "replicate_points_exact": list(map(str, estimates)),
            "exact_oracle": "MATCHED_EXPANDED_OR_PAIRWISE_IDENTITY",
            "design_authentication": "NOT_ESTABLISHED",
            "source_authentication": "NOT_ESTABLISHED",
            "confidence_intervals": "NOT_COMPUTED_DESIGN_NOT_VERIFIED",
            "native_execution": "NOT_RUN", "evidence": ev}


def verify(q, result):
    if type(result) is not dict or canonical(calculate(q)) != canonical(result):
        _bad("RECOMPUTATION_MISMATCH")
    return {"status": "RECOMPUTED", "scope": "SAME_CODE_DECLARED_DATA_ONLY"}
