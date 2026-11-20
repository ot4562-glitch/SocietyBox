"""Exact bounded cohort-time Difference-in-Differences for declared balanced panels.

Computes *unconditional* panel ATT(g,t) comparisons using never-treated or
not-yet-treated controls. These are arithmetic contrasts, NOT causally
identified effects without independently established parallel trends,
no anticipation beyond the declared window, no spillovers, and stable groups.
No standard errors, covariate adjustment, pre-trend tests, or TWFE are supplied.

For cohort g and post period t, the untreated baseline is g-1-A, where A
is a declared maximum anticipation window. Controls have cohort > t+A, or
are never-treated. Unweighted unit changes are compared at t and baseline.
"""
from collections import defaultdict
from fractions import Fraction as F
import re

from .execution_control import checkpoint

_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}\Z", re.ASCII)
_UNIT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_ ./%-]{0,39}\Z", re.ASCII)
_NUM = re.compile(r"-?(?:0|[1-9][0-9]{0,11})(?:\.[0-9]{1,9})?\Z", re.ASCII)
_DIGEST = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)


def _fail(message):
    raise ValueError("STAGGERED_DID_" + message)


def _number(raw):
    if type(raw) is not str or not _NUM.fullmatch(raw):
        _fail("OUTCOME_DECIMAL_STRING_REQUIRED")
    return F(raw)


def _validate(spec):
    fields = {"schema", "data_kind", "source_declared_sha256", "rights",
              "outcome_unit", "period_unit", "comparison", "anticipation_periods", "rows"}
    if type(spec) is not dict or set(spec) != fields or spec["schema"] != "societybox-staggered-did-v1":
        _fail("SCHEMA_INVALID")
    if spec["data_kind"] not in ("SYNTHETIC", "RESEARCHER_SUPPLIED"):
        _fail("DATA_KIND_INVALID")
    if type(spec["source_declared_sha256"]) is not str or not _DIGEST.fullmatch(spec["source_declared_sha256"]):
        _fail("SOURCE_DIGEST_REQUIRED")
    if spec["rights"] != {"local_processing": "PERMITTED", "report_export": "PERMITTED"}:
        _fail("RIGHTS_REQUIRED")
    for name in ("outcome_unit", "period_unit"):
        if type(spec[name]) is not str or not _UNIT.fullmatch(spec[name]):
            _fail("UNITS_INVALID")
    comparison = spec["comparison"]
    if comparison not in ("NEVER_TREATED", "NOT_YET_TREATED") or type(comparison) is not str:
        _fail("COMPARISON_INVALID")
    anticipation = spec["anticipation_periods"]
    if type(anticipation) is not int or not 0 <= anticipation <= 10:
        _fail("ANTICIPATION_INVALID")
    rows = spec["rows"]
    if type(rows) is not list or not 4 <= len(rows) <= 4096:
        _fail("ROW_BUDGET")
    table, cohorts = {}, {}
    periods, units = set(), set()
    for i, row in enumerate(rows):
        if i % 64 == 0:
            checkpoint()
        if type(row) is not dict or set(row) != {"unit", "period", "first_treated_period", "treated", "outcome"}:
            _fail("ROW_INVALID")
        unit = row["unit"]
        if type(unit) is not str or not _ID.fullmatch(unit):
            _fail("UNIT_ID_INVALID")
        period, cohort = row["period"], row["first_treated_period"]
        if type(period) is not int or not 0 <= period <= 1000000:
            _fail("PERIOD_INVALID")
        if cohort is not None and (type(cohort) is not int or not 0 <= cohort <= 1000000):
            _fail("COHORT_INVALID")
        if type(row["treated"]) is not bool:
            _fail("TREATMENT_INVALID")
        if row["treated"] != (cohort is not None and period >= cohort):
            _fail("TREATMENT_REVERSAL_OR_COHORT_MISMATCH")
        key = (unit, period)
        if key in table:
            _fail("DUPLICATE_UNIT_PERIOD")
        if unit in cohorts and cohorts[unit] != cohort:
            _fail("INCONSISTENT_COHORT")
        cohorts[unit] = cohort
        table[key] = _number(row["outcome"])
        periods.add(period)
        units.add(unit)
    time = sorted(periods)
    if len(time) > 32 or len(units) > 256 or len(table) != len(time) * len(units):
        _fail("UNBALANCED_PANEL_UNSUPPORTED")
    # Do not materialize a potentially million-element range for sparse input.
    if any(t1 - t0 != 1 for t0, t1 in zip(time, time[1:])):
        _fail("NONCONSECUTIVE_PERIODS_UNSUPPORTED")
    if any(cohort is not None and cohort <= time[0] + anticipation for cohort in cohorts.values()):
        _fail("UNTREATED_BASELINE_MISSING")
    treated_cohorts = sorted({c for c in cohorts.values() if c is not None and c <= time[-1]})
    if not treated_cohorts:
        _fail("NO_TREATED_COHORT_IN_WINDOW")
    if any(cohort - anticipation - 1 < time[0] for cohort in treated_cohorts):
        _fail("UNTREATED_BASELINE_MISSING")
    return time, sorted(units), cohorts, table, treated_cohorts


def _average(values):
    return sum(values, F(0)) / len(values)


def cohort_time_att(spec):
    """Compute unweighted ATT(g,t) contrasts from a declared balanced panel.

    A blocked cohort-time cell is reported, not silently dropped. Event-time
    aggregates use positive cohort-size weights across supported cohorts only.
    Overall aggregate first averages supported periods within each cohort,
    then weights the cohort means by treated cohort sizes. The changing
    support makes these descriptive aggregates conditional on support.
    """
    periods, units, cohorts, table, treated_cohorts = _validate(spec)
    A, comparison = spec["anticipation_periods"], spec["comparison"]
    effects, grouped, by_event = [], defaultdict(list), defaultdict(list)
    for g in treated_cohorts:
        members = [u for u in units if cohorts[u] == g]
        baseline = g - A - 1
        for t in periods:
            if t < g:
                continue
            checkpoint()
            if comparison == "NEVER_TREATED":
                controls = [u for u in units if cohorts[u] is None]
            else:
                controls = [u for u in units if cohorts[u] is None or cohorts[u] > t + A]
            entry = dict(cohort=g, period=t, event_time=t - g,
                         baseline_period=baseline, treated_units=len(members),
                         control_units=len(controls), comparison=comparison,
                         status="COMPUTED_ASSUMPTIONS_UNVERIFIED" if controls else "BLOCKED",
                         blocker=None if controls else "NO_UNEXPOSED_CONTROLS_AT_PERIOD",
                         treated_change_exact=None, control_change_exact=None, att_exact=None)
            if controls:
                treated_change = _average([table[u, t] - table[u, baseline] for u in members])
                control_change = _average([table[u, t] - table[u, baseline] for u in controls])
                effect = treated_change - control_change
                entry.update(treated_change_exact=str(treated_change),
                             control_change_exact=str(control_change),att_exact=str(effect))
                grouped[g].append((t, effect, len(members)))
                by_event[t - g].append((g, effect, len(members)))
            effects.append(entry)
    if not grouped:
        _fail("NO_SUPPORTED_COHORT_TIME_COMPARISONS")

    events = []
    for event_time in sorted(by_event):
        entries = by_event[event_time]
        size = sum(n for _, _, n in entries)
        effect = sum((value*n for _, value, n in entries), F(0))/size
        events.append(dict(event_time=event_time,att_exact=str(effect),
                           contributing_cohorts=[g for g,_,_ in entries],
                           treated_units_in_support=size,
                           cohort_size_weights_exact=[str(F(n,size)) for _,_,n in entries]))

    cohort_means = []
    total = 0
    combined = F(0)
    for g in sorted(grouped):
        entries = grouped[g]
        size = entries[0][2]
        mean = _average([effect for _, effect, _ in entries])
        combined += mean*size
        total += size
        cohort_means.append(dict(cohort=g, mean_att_exact=str(mean),treated_units=size,
                                 supported_periods=[t for t,_,_ in entries]))
    return dict(schema="societybox-staggered-did-result-v1",
                status=("COMPUTED_ASSUMPTIONS_UNVERIFIED" if len(effects) == sum(map(len,grouped.values()))
                        else "PARTIAL_COMPUTATION_ASSUMPTIONS_UNVERIFIED"),
                estimand="UNCONDITIONAL_GROUP_TIME_PANEL_DID_CONTRAST_NOT_IDENTIFICATION_CERTIFICATE",
                comparison=comparison, anticipation_periods=A,
                source_declared_sha256=spec["source_declared_sha256"],
                source_authentication="NOT_ESTABLISHED", data_kind=spec["data_kind"],
                outcome_unit=spec["outcome_unit"], period_unit=spec["period_unit"],
                parallel_trends="NOT_VERIFIED", no_spillovers="NOT_VERIFIED",
                treatment_anticipation_assumption="DECLARED_NOT_VERIFIED",
                standard_errors="NOT_IMPLEMENTED", original_upstream_parity="NOT_RUN",
                balanced_panel_required=True, unit_weights="EQUAL_WITHIN_GROUP",
                aggregation="COHORT_SIZE_WEIGHTED_WITHIN_SUPPORTED_EVENT_TIME_OR_COHORT_MEANS",
                cohort_time=effects, event_time=events, cohort_means=cohort_means,
                overall_cohort_mean_att_exact=str(combined/total),
                overall_support_treated_units=total,
                supported_cells=sum(entry["att_exact"] is not None for entry in effects),
                blocked_cells=sum(entry["att_exact"] is None for entry in effects))
