"""Bounded evidence-first daily inverse-pattern probe for Society Box IMFE.

This is an initial, deterministic, NO_LLM/NO_NETWORK research component. It
discovers *candidate* shared innovations and a predeclared one-day lag relation
from user-declared complete DAILY observations; it does not discover a named
hidden cause or prove structural identification, vintage or forecast skill.

REA inspiration is the evidence/unknown/verification workflow only; no REA
implementation, tool, code or original Evidence schema is imported or copied.
"""
from datetime import date, timedelta
from decimal import Decimal
import hashlib
import math
import re

from .workspace import ScopeError, canonical

SCHEMA = "societybox-daily-latent-probe-v1"
RESULT = "societybox-daily-latent-probe-result-v1"
_NAMES = re.compile(r"[a-z][a-z0-9_]{0,39}\Z", re.ASCII)
_DIGEST = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_NUMBER = re.compile(r"-?(?:0|[1-9][0-9]{0,14})(?:\.[0-9]{1,8})?\Z", re.ASCII)
_DAYS = re.compile(r"\d{4}-\d{2}-\d{2}\Z", re.ASCII)
_FIELDS = {"schema", "data_kind", "source_declared_sha256",
           "data_frequency", "historical_release_proof", "series",
           "training_end_day", "observations"}
_SERIES = {"series_id", "unit", "role"}
_ROW = {"day", "values"}
MIN_DAYS, MAX_DAYS = 55, 512
MIN_TRAIN, MIN_HOLDOUT = 30, 14
MAX_SOURCE_BYTES = 128 * 1024


def _bad(reason):
    raise ScopeError("IMFE_DAILY_" + reason)


def _number(v):
    if type(v) is not str or len(v) > 40 or not _NUMBER.fullmatch(v):
        _bad("NUMBER_INVALID")
    # Validate bounds before the float conversion can round 1e12+epsilon
    # back into the allowed interval.
    if abs(Decimal(v)) > Decimal("1e12"):
        _bad("NUMBER_INVALID")
    n = float(v)
    if not math.isfinite(n):
        _bad("NUMBER_INVALID")
    return n


def _mean(values):
    return math.fsum(values) / len(values)


def _covariance(a, b):
    ma, mb = _mean(a), _mean(b)
    return math.fsum((x-ma)*(y-mb) for x,y in zip(a,b))


def _pearson(a, b):
    aa, bb = _covariance(a,a), _covariance(b,b)
    if aa <= 1e-18 or bb <= 1e-18:
        return None
    result = _covariance(a,b) / math.sqrt(aa*bb)
    return max(-1.0,min(1.0,result))


def _fmt(value):
    if not math.isfinite(value):
        _bad("NUMERICAL_NONFINITE")
    return format(value, ".12g")


def _id(payload):
    return "ev_" + hashlib.sha256(canonical(payload)).hexdigest()


def _validate(spec):
    if type(spec) is not dict or spec.keys() != _FIELDS or spec["schema"] != SCHEMA:
        _bad("SPEC_INVALID")
    if (spec["data_kind"] not in ("SYNTHETIC", "DECLARED_PUBLIC_DAILY") or
            spec["data_frequency"] != "DAILY" or
            spec["historical_release_proof"] != "NOT_ESTABLISHED" or
            type(spec["source_declared_sha256"]) is not str or
            not _DIGEST.fullmatch(spec["source_declared_sha256"])):
        _bad("SPEC_INVALID")
    series = spec["series"]
    if type(series) is not list or len(series) != 2:
        _bad("SERIES_INVALID")
    roles, ids = {}, set()
    for item in series:
        if (type(item) is not dict or item.keys() != _SERIES or
                item["role"] not in ("TARGET","DRIVER") or
                type(item["series_id"]) is not str or
                not _NAMES.fullmatch(item["series_id"]) or
                type(item["unit"]) is not str or
                not 1 <= len(item["unit"]) <= 64 or
                any(ord(ch) < 32 for ch in item["unit"]) or
                item["series_id"] in ids or item["role"] in roles):
            _bad("SERIES_INVALID")
        roles[item["role"]] = item["series_id"]
        ids.add(item["series_id"])
    if set(roles) != {"TARGET", "DRIVER"}:
        _bad("SERIES_INVALID")
    rows = spec["observations"]
    if type(rows) is not list or not MIN_DAYS <= len(rows) <= MAX_DAYS:
        _bad("OBSERVATION_LIMIT")
    try:
        if type(spec["training_end_day"]) is not str or not _DAYS.fullmatch(spec["training_end_day"]):
            _bad("DAY_INVALID")
        train_end = date.fromisoformat(spec["training_end_day"])
    except ValueError:
        _bad("DAY_INVALID")
    observations = []
    for row in rows:
        if (type(row) is not dict or row.keys() != _ROW or
                type(row["day"]) is not str or not _DAYS.fullmatch(row["day"]) or
                type(row["values"]) is not dict or
                row["values"].keys() != ids):
            _bad("ROW_INVALID")
        try:
            day = date.fromisoformat(row["day"])
        except ValueError:
            _bad("DAY_INVALID")
        if observations and day != observations[-1][0] + timedelta(days=1):
            _bad("DAILY_CONTINUITY_REQUIRED")
        y = _number(row["values"][roles["TARGET"]])
        x = _number(row["values"][roles["DRIVER"]])
        observations.append((day, y, x))
    # A valid ISO date at year 9999 can still overflow when projecting T+1.
    if observations[-1][0] == date.max:
        _bad("DAY_INVALID")
    if train_end not in [r[0] for r in observations]:
        _bad("TRAINING_END_NOT_OBSERVED")
    train_idx = next(i for i,obs in enumerate(observations) if obs[0]==train_end)
    if train_idx+1 < MIN_TRAIN or len(rows)-train_idx-1 < MIN_HOLDOUT:
        _bad("TRAIN_HOLDOUT_LIMIT")
    # Bound input in addition to row counts. No source filename or raw dataset
    # content leaves this local function.
    if len(canonical(spec)) > MAX_SOURCE_BYTES:
        _bad("INPUT_TOO_LARGE")
    return roles, observations, train_idx


def investigate_daily_innovations(spec):
    """Evaluate observed innovations, a candidate latent proxy, and lagged fit.

    x[t-1]-x[t-2] predicts y[t]-y[t-1]; training and holdout disjoint.
    Same-day common-innovation proxy is retrospective only and cannot be
    passed to a future-origin forecaster as an observed latent variable.
    """
    roles, rows, cutoff = _validate(spec)
    days = [v[0] for v in rows]
    y = [v[1] for v in rows]
    x = [v[2] for v in rows]
    dy = [y[i]-y[i-1] for i in range(1,len(y))]
    dx = [x[i]-x[i-1] for i in range(1,len(x))]
    # Regression uses day t's dependent innovation and ONLY prior-day x
    # innovation. Cutoff day belongs to training, next day to holdout.
    regress_x = [x[t-1]-x[t-2] for t in range(2,cutoff+1)]
    regress_y = [y[t]-y[t-1] for t in range(2,cutoff+1)]
    sxx = _covariance(regress_x,regress_x)
    model_supported = sxx > 1e-15
    if model_supported:
        beta = _covariance(regress_x,regress_y)/sxx
        intercept = _mean(regress_y)-beta*_mean(regress_x)
    else:
        beta,intercept=0.0,0.0
    # Holdout predictions never use y[t] or x[t] for target day t.
    cells = []
    for t in range(cutoff+1,len(rows)):
        baseline=y[t-1]
        candidate=y[t-1]+intercept+beta*(x[t-1]-x[t-2]) if model_supported else baseline
        cells.append({"forecast_origin_day":days[t-1].isoformat(),
                      "target_day":days[t].isoformat(),
                      "forecast_horizon_days":1,
                      "baseline_random_walk":_fmt(baseline),
                      "candidate_lagged":_fmt(candidate) if model_supported else None,
                      "observed_target":_fmt(y[t]),
                      "candidate_abs_error":_fmt(abs(y[t]-candidate)) if model_supported else None,
                      "baseline_abs_error":_fmt(abs(y[t]-baseline)),
                      "availability":"DECLARED_PREVIOUS_DAY_ONLY_NOT_RELEASE_ATTESTED"})
    baseline_mae = _mean([abs(y[t]-y[t-1]) for t in range(cutoff+1,len(rows))])
    candidate_mae = _mean([abs(y[t]-(y[t-1]+intercept+beta*(x[t-1]-x[t-2])))
                           for t in range(cutoff+1,len(rows))]) if model_supported else None
    # Contemporaneous innovations are *separate* from the lagged forecast.
    contemporaneous = _pearson(dx[:cutoff],dy[:cutoff])
    ex_post_proxy = []
    if contemporaneous is not None and abs(contemporaneous)>=0.55:
        mx,my = _mean(dx[:cutoff]),_mean(dy[:cutoff])
        sd_x=math.sqrt(_covariance(dx[:cutoff],dx[:cutoff])/cutoff)
        sd_y=math.sqrt(_covariance(dy[:cutoff],dy[:cutoff])/cutoff)
        for t in range(max(1,len(rows)-5),len(rows)):
            zx=(dx[t-1]-mx)/sd_x
            zy=(dy[t-1]-my)/sd_y
            f=(zx + math.copysign(1,contemporaneous)*zy)/2
            ex_post_proxy.append({"day":days[t].isoformat(),
                                  "arbitrary_scale_value":_fmt(f),
                                  "uses_same_day_target":True,
                                  "mode":"EX_POST_UNIDENTIFIED_PROXY_ONLY"})
    # Observation IDs commit to the *actual supplied bytes*, not just the
    # user-declared upstream hash; a renamed/edited daily row cannot retain
    # the same Evidence ID.
    input_digest=hashlib.sha256(canonical(spec)).hexdigest()
    observation_evidence = {"authority":"DECLARED_LOCAL_OBSERVATIONS",
                            "input_sha256":input_digest,
                            "epistemic_state":"observed",
                            "source_sha256":spec["source_declared_sha256"],
                            "coverage":"COMPLETE_SUPPLIED_CONTIGUOUS_DAYS_ONLY",
                            "data_points":len(rows),
                            "source_authentication":"NOT_ESTABLISHED"}
    observation_id=_id(observation_evidence)
    numerical_evidence={"authority":"DETERMINISTIC_LOCAL_ARITHMETIC",
                        "epistemic_state":"derived",
                        "supports":[observation_id],
                        "training_end_day":spec["training_end_day"],
                        "holdout_count":len(cells),
                        "model_fit_status":"FIT" if model_supported else "RANK_DEFICIENT",
                        "baseline_mae":_fmt(baseline_mae),
                        "candidate_mae":_fmt(candidate_mae) if model_supported else None}
    numeric_id=_id(numerical_evidence)
    proposal_evidence={"authority":"PATTERN_BASED_HYPOTHESIS",
                       "epistemic_state":"inferred",
                       "supports":[observation_id,numeric_id],
                       "status":("SHARED_INNOVATION_PROXY_CANDIDATE"
                                 if ex_post_proxy else "INSUFFICIENT_COMMON_SIGNAL"),
                       "cannot_name_underlying_cause":True}
    proposal_id=_id(proposal_evidence)
    last=len(rows)-1
    future_day=(days[last]+timedelta(days=1)).isoformat()
    next_day = (_fmt(y[last]+intercept+beta*(x[last]-x[last-1]))
                if model_supported else None)
    return {
        "schema":RESULT,"status":"READY" if model_supported else "INDETERMINATE",
        "analysis":"TWO_DAILY_SERIES_FIXED_ONE_DAY_LAG_HYPOTHESIS",
        "data_kind":spec["data_kind"],
        "input_sha256":input_digest,
        "source_declared_sha256":spec["source_declared_sha256"],
        "training":{"start_day":days[0].isoformat(),
                    "end_day":days[cutoff].isoformat(),
                    "observations":cutoff+1,
                    "lagged_regression_pairs":len(regress_y)},
        "holdout":{"start_day":days[cutoff+1].isoformat(),
                   "end_day":days[-1].isoformat(),"observations":len(cells),
                   "evaluation":"SIMULATED_RETROSPECTIVE_ONE_STEP",
                   "baseline_mae":_fmt(baseline_mae),
                   "candidate_mae":_fmt(candidate_mae) if model_supported else None,
                   "comparative_claim":(
                      "LOWER_MAE_ON_THIS_HOLDOUT_NO_SIGNIFICANCE_CLAIM"
                      if model_supported and candidate_mae < baseline_mae else
                      "NO_EMPIRICAL_ADVANTAGE_ESTABLISHED")},
        "model":{"family":"PREDECLARED_LAGGED_DIFFERENCE_OLS",
                 "intercept":_fmt(intercept) if model_supported else None,
                 "lag_coefficient":_fmt(beta) if model_supported else None,
                 "fit_status":"FIT" if model_supported else "RANK_DEFICIENT",
                 "specification_discovery":"NOT_AUTOMATIC"},
        "innovation_pattern":{"contemporaneous_training_correlation":
                              _fmt(contemporaneous) if contemporaneous is not None else None,
                              "common_latent_proxy_status":proposal_evidence["status"],
                              "proxy_samples_ex_post":ex_post_proxy,
                              "latent_identity":"UNKNOWN",
                              "causal_direction":"NOT_ESTABLISHED"},
        "holdout_forecasts":cells,
        "next_day_projection":{
             "forecast_origin_day":days[last].isoformat(),"target_day":future_day,
             "model_conditional_point":next_day,
             "prediction_status":"UNCALIBRATED_ONE_DAY_EXTRAPOLATION"
                                  if model_supported else "NOT_ESTIMABLE",
             "actual_future_value":"UNKNOWN",
             "long_horizon_forecast":"NOT_IMPLEMENTED"},
        "evidence":[{"evidence_id":observation_id,**observation_evidence},
                    {"evidence_id":numeric_id,**numerical_evidence},
                    {"evidence_id":proposal_id,**proposal_evidence}],
        "residual_unknowns":[
          {"question":"What real-world variable or mechanism generated the common innovation?",
           "status":"OPEN","needed":"INDEPENDENT_OBSERVABLES_OR_IDENTIFICATION_RESTRICTIONS"},
          {"question":"Were all inputs available at each historical forecast origin?",
           "status":"OPEN","needed":"AUTHENTICATED_RELEASE_AND_ACCESS_VINTAGES"},
          {"question":"Does this relationship forecast a new regime or longer horizon?",
           "status":"OPEN","needed":"PROSPECTIVE_OUT_OF_SAMPLE_REGIME_AND_CALIBRATION_TESTS"}],
        "llm_investigation_brief":{
           "automatic_model_call":False,
           "sharing_policy":"LOCAL_OR_SEPARATELY_APPROVED_ONLY",
           "raw_rows_included":False,
           "question":"Propose multiple falsifiable latent economic mechanisms behind the co-movement and lagged innovations; distinguish correlation, confounding and causal identification.",
           "training_only_correlation":_fmt(contemporaneous)
                                      if contemporaneous is not None else None,
           "observed_lagged_fit":"TRAIN_ONLY" if model_supported else "UNIDENTIFIABLE",
           "required_followups":["independent proxies","timing/release tests",
                                 "alternative mechanisms","negative controls",
                                 "regime stability and origin-wise backtest"]},
        "assurance":{"empirical_source_authenticated":False,
                     "historical_as_of_verified":False,
                     "hidden_cause_discovered":False,
                     "causal_identification":"NOT_ESTABLISHED",
                     "latent_factor_unique":"NOT_ESTABLISHED",
                     "independent_external_oracle":"NOT_RUN",
                     "out_of_sample_test":"DECLARED_WINDOW_ONLY",
                     "future_value_known":False,
                     "llm_host_consumption":"NOT_RUN",
                     "REA_SOURCE_IMPORTED":False},
    }
