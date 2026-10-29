"""Bounded automatic inverse-hypothesis screening over reviewed daily observations.

Runs the existing exact rational lagged-regression engine, not imported research
scripts. Training-only comparisons are exploratory and do not identify hidden
causes, causal directions, historical vintages or statistical significance.
"""
from fractions import Fraction
import hashlib
import math

from .feature_catalog import describe_feature
from .mechanism_dynamic import reconstruct_dynamic, reconstruct_dynamic_screen_batch
from .workspace import ScopeError, canonical

SCHEMA = "societybox-inverse-screen-v1"
RESULT_SCHEMA = "societybox-inverse-screen-result-v1"
_MAX_BYTES = 131072


def _reject(reason):
    raise ScopeError("IMFE_SCREEN_" + reason)


def _validate_request(payload):
    if type(payload) is not dict or set(payload) != {"schema", "study", "search", "rights"} or payload["schema"] != SCHEMA:
        _reject("SCHEMA_INVALID")
    if payload["rights"] != {"local_processing": "PERMITTED", "report_export": "PERMITTED"}:
        _reject("PROCESSING_RIGHTS_REQUIRED")
    data, search = payload["study"], payload["search"]
    if (type(data) is not dict or set(data) != {"schema", "data_kind", "source_declared_sha256",
            "target", "series_units", "training_end_day", "observations"} or
            type(search) is not dict or set(search) != {"lags", "include_target_history"}):
        _reject("SCHEMA_INVALID")
    # Check mapping keys before canonical JSON sorting: malformed Python callers
    # could otherwise escape with TypeError instead of a scoped rejection.
    units = data["series_units"]
    if type(units) is not dict or any(type(name) is not str for name in units):
        _reject("SERIES_INVALID")
    lags = search["lags"]
    if (type(lags) is not list or not 1 <= len(lags) <= 3 or
            any(type(v) is not int or not 1 <= v <= 7 for v in lags) or
            sorted(set(lags)) != lags or type(search["include_target_history"]) is not bool):
        _reject("SEARCH_INVALID")
    try:
        if len(canonical(payload)) > _MAX_BYTES:
            _reject("INPUT_TOO_LARGE")
    except (TypeError, ValueError, OverflowError, RecursionError):
        _reject("INPUT_INVALID")
    return data, lags


def _candidate(name, features):
    return {"candidate_id": name, "features": features}


def _single_candidates(data, lags, include_target_history):
    names = sorted(data["series_units"])
    if type(data.get("target")) is not str or data["target"] not in names:
        _reject("SERIES_INVALID")
    considered = [name for name in names if name != data["target"] or include_target_history]
    if not considered:
        _reject("SERIES_INVALID")
    return [_candidate("screen_%02d" % (i + 1), [{"series": name, "lag": lag}])
            for i, (name, lag) in enumerate((name, lag) for name in considered for lag in lags)]


def _residual_proxies(data, chosen, fit, max_lag):
    """Retrospective training residual correlations; not usable as forecast inputs."""
    if fit is None or fit["status"] != "FIT":
        return []
    rows = data["observations"]
    end_index = next(i for i, row in enumerate(rows) if row["day"] == data["training_end_day"])
    residuals = [Fraction(v) for v in fit["training_residuals_exact"]]
    if len(residuals) != end_index - max_lag + 1:
        _reject("RESIDUAL_ALIGNMENT_INVALID")
    included = {f["series"] for f in chosen["features"]} | {data["target"]}
    mean_e = sum(residuals) / len(residuals)
    ve = sum((e - mean_e) ** 2 for e in residuals)
    findings = []
    for name in sorted(set(data["series_units"]) - included):
        values = [Fraction(row["values"][name]) for row in rows[max_lag:end_index + 1]]
        mean_x = sum(values) / len(values)
        vx = sum((v - mean_x) ** 2 for v in values)
        covariance = sum((e - mean_e) * (x - mean_x) for e, x in zip(residuals, values))
        # Only this displayed correlation is approximate; fits and ranking are exact.
        coefficient = None
        if ve > 0 and vx > 0:
            coefficient = format(max(-1.0, min(1.0, float(covariance / ve) /
                                math.sqrt(float(vx / ve)))), ".8g")
        findings.append({"observed_series": name, "training_residual_correlation": coefficient,
                         "mode": "SAME_DAY_EX_POST_PATTERN_NOT_A_FORECAST_FEATURE",
                         "latent_identity": "UNKNOWN", "causality": "NOT_ESTABLISHED"})
    return findings


def screen_inverse_hypotheses(payload):
    """Discover bounded lag hypotheses without selecting on holdout outcomes.

    Evaluates every declared lag on the SAME train window via an explicit anchor,
    so candidates with different lag lengths have comparable training scores.
    A two-feature joint candidate is also evaluated from the two best singles.
    """
    data, lags = _validate_request(payload)
    from .inverse_investigation import _code_receipt, build_inverse_investigation
    code_hashes = _code_receipt()
    singles = _single_candidates(data, lags, payload["search"]["include_target_history"])
    if len(singles) > 24:
        _reject("SEARCH_BUDGET")
    max_lag = max(lags)
    anchor = _candidate("window_anchor", [{"series": singles[0]["features"][0]["series"],
                                           "lag": max_lag}])
    results, fitted = [], {}
    # The exact estimator validates the entire generated list and observation
    # information set only once. Legacy API calls retain their four-candidate
    # bound; this internal batch does not skip per-feature release checks.
    def evaluate(batch, *, screen_batch=False):
        query = dict(data, candidates=batch + [anchor])
        report = (reconstruct_dynamic_screen_batch(query) if screen_batch
                  else reconstruct_dynamic(query))
        if report["training"]["common_max_lag"] != max_lag:
            _reject("SAMPLE_WINDOW_MISMATCH")
        for candidate, fit in zip(batch, report["candidates"][:len(batch)]):
            record = {"candidate_id": candidate["candidate_id"],
                      "features": candidate["features"], "status": fit["status"]}
            if fit["status"] == "FIT":
                record.update(training_score_exact=fit["selection_score_exact"],
                              training_sse_exact=fit["training_sse_exact"],
                              training_degrees_of_freedom=fit["residual_degrees_of_freedom"])
                fitted[candidate["candidate_id"]] = (candidate, fit, report)
            results.append(record)
    evaluate(singles, screen_batch=True)

    ranked_singles = sorted((Fraction(r["training_score_exact"]), len(r["features"]), r["candidate_id"])
                            for r in results if r["status"] == "FIT")
    if len(ranked_singles) >= 2:
        one = fitted[ranked_singles[0][2]][0]
        two = fitted[ranked_singles[1][2]][0]
        joint = _candidate("joint_top_two", one["features"] + two["features"])
        if joint["features"][0] != joint["features"][1]:
            evaluate([joint])

    ranked_all = sorted((Fraction(r["training_score_exact"]), len(r["features"]), r["candidate_id"])
                        for r in results if r["status"] == "FIT")
    winner = ranked_all[0][2] if ranked_all else None
    chosen, fit, run = fitted[winner] if winner is not None else (None, None, None)
    result = {
        "schema": RESULT_SCHEMA, "status": "READY" if winner else "INDETERMINATE",
        "product_feature": describe_feature("dynamic_inverse"),
        "input_sha256": hashlib.sha256(canonical(payload)).hexdigest(),
        "source_declared_sha256": data["source_declared_sha256"],
        "source_authentication": "NOT_ESTABLISHED",
        "processing_rights": "OPERATOR_DECLARED_NOT_AUTHENTICATED",
        "candidate_count": len(results),
        "ranking_rule": "TRAINING_ONLY_SSE_PER_RESIDUAL_DF_WITH_COMMON_LAG_WINDOW",
        "common_max_lag": max_lag,
        "candidates": results,
        "selected_candidate_id": winner,
        "selected_features": [] if chosen is None else chosen["features"],
        "selected_coefficients_exact": None if fit is None else fit["coefficients_exact"],
        "selected_training_residuals_exact": None if fit is None else fit["training_residuals_exact"],
        "holdout": None if fit is None else {
            "evaluation": run["evaluation"],
            "model_mse_exact": fit["holdout_mse_exact"],
            "naive_previous_target_mse_exact": run["baseline"]["holdout_mse_exact"],
            "rows": fit["holdout"],
            "was_used_for_selection": False},
        "residual_pattern_candidates": _residual_proxies(data, chosen, fit, max_lag),
        "assurance": {
            "method": "BOUNDED_NO_LLM_EXACT_LAGGED_OLS_EXPLORATORY_SCREEN",
            "automatic_candidate_generation": True,
            "historical_release_proof": "RESEARCHER_DECLARED_UNVERIFIED",
            "multiple_comparisons": "UNCORRECTED_EXPLORATORY_HYPOTHESIS_SEARCH",
            "independent_oracle": "EXACT_CRAMER_ARITHMETIC_SAME_PROCESS_NOT_EXTERNAL_REPLICATION",
            "latent_identity": "UNKNOWN",
            "causal_identification": "NOT_ESTABLISHED",
            "hidden_cause_discovered": False,
            "future_values_known": False,
            "residual_proxy_forecast_use": "PROHIBITED",
            "llm": "NO_LLM",
        },
    }
    from .inverse_numeric_verifier import verify_computed_screen
    result["numeric_verification"] = verify_computed_screen(payload, result)
    if code_hashes != _code_receipt():
        raise ScopeError("INVESTIGATION_CODE_CHANGED_DURING_CALCULATION")
    result["investigation"] = build_inverse_investigation(payload, result, code_hashes)
    return result
