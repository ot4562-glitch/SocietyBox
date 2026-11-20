"""Exact stratified-PSU Taylor linearization for a declared weighted mean.

The one-stage with-replacement (WR) first-stage PSU approximation is
conditional on a correctly declared probability sampling design. The method
is not a PSU design authenticator, an FPC, a normal/t calibration, an estimate
of causal effects, or a substitute for replicate-weight estimators. Domain
analysis retains the complete sampled PSU frame, including zero contributions
from sampled units outside the domain.
"""
from collections import defaultdict
from fractions import Fraction as F
import hashlib
import json
import re

from .execution_control import checkpoint

SCHEMA = "societybox-survey-stratified-psu-mean-v1"
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}\Z", re.ASCII)
_UNIT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_ ./%-]{0,63}\Z", re.ASCII)
_DIGEST = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_NUMBER = re.compile(
    r"-?(?:(?:0|[1-9][0-9]{0,11})(?:\.[0-9]{1,9})?|"
    r"(?:0|[1-9][0-9]{0,11})/[1-9][0-9]{0,11})\Z",
    re.ASCII,
)


def _fail(reason):
    raise ValueError("SURVEY_STRATIFIED_" + reason)


def _number(raw):
    if type(raw) is not str or not _NUMBER.fullmatch(raw):
        _fail("NUMBER_INVALID")
    try:
        return F(raw)
    except (ValueError, ZeroDivisionError, OverflowError):
        _fail("NUMBER_INVALID")


def _validate(spec):
    required = {"schema", "data_kind", "source_declared_sha256", "rights",
                "outcome_unit", "psu_sampling", "rows"}
    if type(spec) is not dict or set(spec) != required or spec["schema"] != SCHEMA:
        _fail("SCHEMA_INVALID")
    if type(spec["data_kind"]) is not str or spec["data_kind"] not in (
        "SYNTHETIC", "RESEARCHER_SUPPLIED"
    ):
        _fail("DATA_KIND_INVALID")
    if type(spec["source_declared_sha256"]) is not str or not _DIGEST.fullmatch(
        spec["source_declared_sha256"]
    ):
        _fail("SOURCE_DIGEST_REQUIRED")
    if spec["rights"] != {"local_processing": "PERMITTED", "report_export": "PERMITTED"}:
        _fail("RIGHTS_REQUIRED")
    if type(spec["outcome_unit"]) is not str or not _UNIT.fullmatch(spec["outcome_unit"]):
        _fail("UNIT_INVALID")
    if spec["psu_sampling"] != "WITH_REPLACEMENT_FIRST_STAGE_APPROXIMATION_DECLARED":
        _fail("DESIGN_DECLARATION_REQUIRED")
    rows = spec["rows"]
    if type(rows) is not list or not 4 <= len(rows) <= 2048:
        _fail("ROW_BUDGET")
    seen = set()
    groups = defaultdict(lambda: defaultdict(list))
    observations = 0
    domain_count = 0
    domain_psus = set()
    total_w = F(0)
    total_yw = F(0)
    for index, row in enumerate(rows):
        if index % 32 == 0:
            checkpoint()
        if type(row) is not dict or set(row) != {
            "unit", "stratum", "psu", "value", "population_weight", "in_domain"
        }:
            _fail("ROW_INVALID")
        unit, stratum, psu = row["unit"], row["stratum"], row["psu"]
        if any(type(value) is not str or not _ID.fullmatch(value)
               for value in (unit, stratum, psu)):
            _fail("IDENTIFIER_INVALID")
        if unit in seen:
            _fail("DUPLICATE_UNIT")
        seen.add(unit)
        if type(row["in_domain"]) is not bool:
            _fail("DOMAIN_INDICATOR_INVALID")
        value, weight = _number(row["value"]), _number(row["population_weight"])
        if weight <= 0:
            _fail("POSITIVE_WEIGHT_REQUIRED")
        groups[stratum][psu].append((value, weight, row["in_domain"]))
        observations += 1
        if row["in_domain"]:
            domain_count += 1
            domain_psus.add((stratum, psu))
            total_w += weight
            total_yw += weight * value
    if len(groups) > 64 or sum(map(len, groups.values())) > 512:
        _fail("DESIGN_BUDGET")
    if any(len(psus) < 2 for psus in groups.values()):
        _fail("LONELY_PSU_UNSUPPORTED")
    if not total_w:
        _fail("EMPTY_DOMAIN")
    if len(domain_psus) < 2:
        _fail("DOMAIN_PSU_SUPPORT_INSUFFICIENT")
    return groups, observations, domain_count, total_w, total_yw


def validate(spec):
    """Validate declarations and sample design shape, without fitting a mean."""
    _validate(spec)


def stratified_psu_mean(spec):
    """Exact ratio mean and stratified first-stage WR Taylor PSU variance.

    Let e_hi = sum_j w_hij * I(domain) * (y_hij - theta), including
    e_hi=0 for PSUs with no domain members. With m_h sampled PSUs per stratum,
    V = sum_h [m_h/(m_h-1) * sum_i (e_hi - mean_h(e))**2] / W_domain**2.

    The PSU totals are not released; only stratum PSU counts and the exact
    stratum variance contribution are returned. This is arithmetic under an
    assumed design, not a certified survey variance or inferential coverage.
    """
    groups, n, domain_n, total_w, total_yw = _validate(spec)
    point = total_yw / total_w
    variance = F(0)
    psu_count = 0
    strata = []
    for stratum in sorted(groups):
        checkpoint()
        totals = []
        for psu in sorted(groups[stratum]):
            checkpoint()
            totals.append(sum((weight * (y - point) for y, weight, active
                               in groups[stratum][psu] if active), F(0)))
        m = len(totals)
        psu_count += m
        mean_total = sum(totals, F(0)) / m
        component = F(m, m-1) * sum(((e - mean_total)**2 for e in totals), F(0)) / total_w**2
        if component < 0:
            _fail("NEGATIVE_VARIANCE")
        variance += component
        strata.append(dict(stratum=stratum, psu_count=m,
                           variance_component_exact=str(component)))
    if variance < 0:
        _fail("NEGATIVE_VARIANCE")
    input_sha = hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":"),
                                           ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()
    return dict(schema="societybox-survey-stratified-psu-mean-result-v1",
                status="DECLARED_DESIGN_ARITHMETIC_ASSUMPTIONS_UNVERIFIED",
                estimator="STRATIFIED_PSU_WR_TAYLOR_RATIO_MEAN",
                input_sha256=input_sha,
                source_declared_sha256=spec["source_declared_sha256"],
                source_authentication="NOT_ESTABLISHED",
                design_authentication="NOT_ESTABLISHED",
                inference_validity="NOT_ESTABLISHED",
                confidence_interval="NOT_CALCULATED",
                finite_population_correction="NOT_APPLIED",
                sampling_assumption="FIRST_STAGE_PSU_WITH_REPLACEMENT_APPROXIMATION_DECLARED",
                weight_semantics="DECLARED_POSITIVE_POPULATION_WEIGHTS_NOT_REGRESSION_ANALYTIC_WEIGHTS",
                variance_scope="TAYLOR_LINEARIZATION_CONDITIONAL_ON_DECLARED_DESIGN",
                original_upstream_parity="NOT_RUN",
                data_kind=spec["data_kind"], outcome_unit=spec["outcome_unit"],
                observations=n, domain_observations=domain_n,
                strata_count=len(groups), total_psus=psu_count,
                degrees_of_freedom=psu_count-len(groups),
                domain_population_weight_exact=str(total_w),
                weighted_mean_exact=str(point), variance_exact=str(variance),
                stratum_variance_components=strata)
