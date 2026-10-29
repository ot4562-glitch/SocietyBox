"""Bounded one-regressor OLS inverse fit with exact arithmetic and forward replay.

Pure, read-only, no packages, no LLM, no source transfer or arbitrary execution.
This reconstructs *coefficients conditional on the researcher-declared sample
and model*. It cannot recover the original published regression specification,
identify causes, or prove historical information availability from observations.
Classical or HC0–HC3 errors are explicit choices, with assumptions NOT VERIFIED.
"""
import csv
import io
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import re

from .workspace import ScopeError, canonical

_SCHEMA = "societybox-ols-reconstruction-v1"
_FIELDS = {
    "schema", "data_kind", "source_declared_sha256", "dependent_variable",
    "explanatory_variable", "dependent_unit", "explanatory_unit", "intercept",
    "variance_estimator", "rows",
}
_ROW_FIELDS = {"observation_id", "x", "y"}
_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,39}\Z", re.ASCII)
_UNIT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_ ./%-]{0,63}\Z", re.ASCII)
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z", re.ASCII)
_DIGEST = re.compile(r"[a-f0-9]{64}\Z", re.ASCII)
_DECIMAL = re.compile(r"-?(?:0|[1-9][0-9]{0,15})(?:\.[0-9]{1,12})?\Z", re.ASCII)
MAX_OBSERVATIONS = 256
MAX_ABS_VALUE = 10 ** 16


def _fail(code):
    raise ScopeError("IMFE_OLS_" + code)


def _decimal_fraction(value):
    if type(value) is not str or len(value) > 40 or not _DECIMAL.fullmatch(value):
        _fail("DATA_INVALID")
    result = Fraction(value)
    if abs(result) > MAX_ABS_VALUE:
        _fail("DATA_INVALID")
    return result


def _sqrt_decimal(value):
    """Display-only approximation; all estimators and assertions are rational."""
    if value < 0:
        _fail("ARITHMETIC_INVALID")
    if value == 0:
        return "0"
    with localcontext() as ctx:
        ctx.prec = 60
        fraction = Decimal(value.numerator) / Decimal(value.denominator)
        return str(fraction.sqrt())


def _fraction(value):
    return str(value)


def reconstruct_ols(spec):
    """Fit y_i = intercept + slope * x_i with *declared*, complete observations.

    Blocks rank deficiency, incomplete rows, unsupported SE estimators and
    accidental implicit choices; does not certify data rights or estimation
    assumptions. Outputs exact coefficients/predictions/residuals, not p-values.
    """
    if type(spec) is not dict or set(spec) not in (_FIELDS, _FIELDS | {"sample_selection"}):
        _fail("SPEC_INVALID")
    if (spec.get("schema") != _SCHEMA or
            spec.get("data_kind") not in ("SYNTHETIC", "DECLARED_PUBLIC_OBSERVATIONS") or
            type(spec.get("source_declared_sha256")) is not str or
            not _DIGEST.fullmatch(spec["source_declared_sha256"]) or
            type(spec.get("intercept")) is not bool or spec["intercept"] is not True or
            spec.get("variance_estimator") not in ("CLASSICAL_HOMOSKEDASTIC", "HC0", "HC1", "HC2", "HC3")):
        _fail("SPEC_INVALID")
    for key in ("dependent_variable", "explanatory_variable"):
        if type(spec[key]) is not str or not _NAME.fullmatch(spec[key]):
            _fail("SPEC_INVALID")
    if spec["dependent_variable"] == spec["explanatory_variable"]:
        _fail("SPEC_INVALID")
    for key in ("dependent_unit", "explanatory_unit"):
        if type(spec[key]) is not str or not _UNIT.fullmatch(spec[key]):
            _fail("SPEC_INVALID")
    rows = spec.get("rows")
    if type(rows) is not list or not 3 <= len(rows) <= MAX_OBSERVATIONS:
        _fail("SPEC_INVALID")
    ids, x, y, row_ids = set(), [], [], []
    for row in rows:
        if type(row) is not dict or row.keys() != _ROW_FIELDS:
            _fail("DATA_INVALID")
        rid = row.get("observation_id")
        if type(rid) is not str or not _ID.fullmatch(rid):
            _fail("DATA_INVALID")
        if rid in ids:
            _fail("ROW_ID_DUPLICATE")
        ids.add(rid)
        row_ids.append(rid)
        x.append(_decimal_fraction(row["x"]))
        y.append(_decimal_fraction(row["y"]))
    input_count = len(x)
    selection = spec.get("sample_selection")
    if "sample_selection" in spec:
        if (type(selection) is not dict or
                selection.keys() != {"observation_ids", "rationale"} or
                type(selection["rationale"]) is not str or
                not selection["rationale"].strip() or len(selection["rationale"]) > 500 or
                any(ord(c) < 32 for c in selection["rationale"])):
            _fail("SAMPLE_INVALID")
        chosen = selection["observation_ids"]
        if (type(chosen) is not list or not 3 <= len(chosen) <= input_count or
                any(type(v) is not str for v in chosen) or
                len(set(chosen)) != len(chosen) or not set(chosen) <= ids):
            _fail("SAMPLE_INVALID")
        selected = set(chosen)
    else:
        selected = ids
    excluded = [rid for rid in row_ids if rid not in selected]
    sample = [(rid, a, b) for rid, a, b in zip(row_ids, x, y) if rid in selected]
    row_ids, x, y = map(list, zip(*sample))
    n = len(x)
    mean_x, mean_y = sum(x) / n, sum(y) / n
    sxx = sum((a - mean_x) ** 2 for a in x)
    if sxx <= 0:
        _fail("REGRESSOR_NOT_IDENTIFIABLE")
    sxy = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    slope, intercept = sxy / sxx, mean_y - (sxy / sxx) * mean_x
    fitted = [intercept + slope * a for a in x]
    residuals = [b - a for a, b in zip(fitted, y)]
    sse = sum(err * err for err in residuals)
    tss = sum((b - mean_y) ** 2 for b in y)
    df = n - 2
    variance = sse / df
    stderr_slope_sq = variance / sxx
    stderr_intercept_sq = variance * (Fraction(1, n) + mean_x ** 2 / sxx)
    covariance = -variance * mean_x / sxx
    leverages = [Fraction(1, n) + (a - mean_x) ** 2 / sxx for a in x]
    estimator = spec["variance_estimator"]
    if estimator != "CLASSICAL_HOMOSKEDASTIC":
        if estimator in ("HC2", "HC3") and any(h == 1 for h in leverages):
            _fail("LEVERAGE_ONE")
        power = {"HC0": 0, "HC1": 0, "HC2": 1, "HC3": 2}[estimator]
        weights = [e ** 2 / (1 - h) ** power for e, h in zip(residuals, leverages)]
        scale = Fraction(n, df) if estimator == "HC1" else Fraction(1)
        # Influence of each y_i on intercept/slope. Exact sandwich covariance.
        slope_influence = [(a - mean_x) / sxx for a in x]
        intercept_influence = [Fraction(1, n) - mean_x * v for v in slope_influence]
        stderr_slope_sq = scale * sum(w * v ** 2 for w, v in zip(weights, slope_influence))
        stderr_intercept_sq = scale * sum(w * v ** 2 for w, v in zip(weights, intercept_influence))
        covariance = scale * sum(w * a * b for w, a, b in
                                 zip(weights, intercept_influence, slope_influence))
    # Independent normal-equation invariants, fail rather than report a
    # misleading result should future refactors break the fit.
    if sum(residuals) != 0 or sum(a * err for a, err in zip(x, residuals)) != 0:
        _fail("NORMAL_EQUATIONS_FAILED")
    if any(fit + err != observed for fit, err, observed in zip(fitted, residuals, y)):
        _fail("FORWARD_REPLAY_FAILED")
    csv_stream = io.StringIO(newline="")
    writer = csv.writer(csv_stream, lineterminator="\n")
    writer.writerow(["observation_id", "x_exact", "y_exact", "fitted_exact", "residual_exact", "leverage_exact"])
    # Validated IDs and numeric fractions exclude formula-leading characters
    # other than numeric minus signs; arbitrary researcher prose is not exported.
    writer.writerows([rid, *map(str, values)] for rid, *values in
                     zip(row_ids, x, y, fitted, residuals, leverages))
    return {
        "schema": "societybox-ols-reconstruction-result-v1",
        "status": "READY",
        "analysis": "ONE_REGRESSOR_LINEAR_OLS_WITH_INTERCEPT",
        "assurance": "DECLARED_DATA_DETERMINISTIC_EXACT_FIT_ONLY",
        "data_kind": spec["data_kind"],
        "source_declared_sha256": spec["source_declared_sha256"],
        "input_contract_sha256": hashlib.sha256(canonical(spec)).hexdigest(),
        "source_authentication": "NOT_ESTABLISHED",
        "historical_vintage_assurance": "NOT_ESTABLISHED",
        "researcher_information_set": "NOT_ESTABLISHED",
        "causal_identification": "NOT_ESTABLISHED",
        "original_regression_spec_recovery": "NOT_ESTABLISHED",
        "upstream_native_parity": "NOT_RUN",
        "estimation_execution": "EXACT_RATIONAL_LOCAL_ONLY",
        "variance_estimator": estimator,
        "standard_error_assumptions": ("IID_HOMOSKEDASTICITY_NOT_VERIFIED" if estimator == "CLASSICAL_HOMOSKEDASTIC"
                                      else "INDEPENDENT_ERRORS_EXOGENEITY_AND_ASYMPTOTICS_NOT_VERIFIED"),
        "variance_exact_meaning": "SSE_OVER_RESIDUAL_DF_NOT_ROBUST_COVARIANCE",
        "covariance_parameter_order": ["intercept", "slope"],
        "covariance_exact": [[str(stderr_intercept_sq), str(covariance)],
                             [str(covariance), str(stderr_slope_sq)]],
        "sample": {"input_observation_count": input_count,
                   "included_observation_ids": row_ids,
                   "excluded_observation_ids": excluded,
                   "selection_rationale": selection["rationale"] if selection else "ALL_DECLARED_ROWS",
                   "missing_policy": "REJECT_ALL_INPUT_MISSING_VALUES"},
        "fit_trace": {"algorithm": "EXACT_CENTERED_OLS", "rank": 2,
                      "mean_x_exact": str(mean_x), "mean_y_exact": str(mean_y),
                      "sxx_exact": str(sxx), "sxy_exact": str(sxy)},
        "back_test": "NOT_RUN",
        "residual_csv": csv_stream.getvalue(),
        "p_values": "NOT_CALCULATED",
        "dependent_variable": spec["dependent_variable"],
        "explanatory_variable": spec["explanatory_variable"],
        "dependent_unit": spec["dependent_unit"],
        "explanatory_unit": spec["explanatory_unit"],
        "observation_count": n,
        "degrees_of_freedom": df,
        "coefficient_exact": {"intercept": _fraction(intercept), "slope": _fraction(slope)},
        "standard_errors_decimal": {
            "intercept": _sqrt_decimal(stderr_intercept_sq),
            "slope": _sqrt_decimal(stderr_slope_sq),
        },
        "variance_exact": _fraction(variance),
        "sse_exact": _fraction(sse),
        "r_squared_exact": None if tss == 0 else _fraction(1 - sse / tss),
        "r_squared_status": "UNDEFINED_CONSTANT_RESPONSE" if tss == 0 else "DEFINED",
        "fitted_exact": [_fraction(v) for v in fitted],
        "residual_exact": [_fraction(v) for v in residuals],
        "normal_equations": "EXACT_ZERO_RESIDUAL_MOMENTS",
        "forward_replay": "EXACT_IDENTITY_PER_ROW",
    }
