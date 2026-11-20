"""Exact staggered-DiD pre-treatment one-period placebo trend contrasts.

These are descriptive, diagnostic differences, not a statistical test of
parallel trends. No causal identification, standard errors, p-values or
nominal coverage are inferred from zero (or small) placebo contrasts.
The input contract and treatment/anticipation validation are shared with
staggered_did.cohort_time_att in the existing PR #52.
"""
from collections import defaultdict
from fractions import Fraction as F

from .execution_control import checkpoint
from .staggered_did import _validate


def _average(values):
    return sum(values, F(0)) / len(values)


def pretrend_placebos(spec):
    """Unweighted cohort placebo changes on fully pre-exposure intervals.

    For treated cohort g and eligible period pair (t-1,t), require
    t <= g - A - 1 (strictly before possible treatment anticipation).
    Use never-treated or not-yet-treated-by-(t+A) controls, excluding
    the treated cohort itself even when it is currently unexposed.
    Every unsupported comparison is blocked, never treated as a zero effect.
    Lead-time aggregates have nonnegative cohort-size weights and explicitly
    changing cohort support; they are not event-study inference estimates.
    """
    periods, units, cohorts, values, treated_cohorts = _validate(spec)
    A, comparison = spec['anticipation_periods'], spec['comparison']
    cells, by_lead = [], defaultdict(list)
    for g in treated_cohorts:
        treated = [u for u in units if cohorts[u] == g]
        cutoff = g - A - 1
        for end in periods[1:]:
            if end > cutoff:
                break
            checkpoint()
            begin = end-1
            if comparison == 'NEVER_TREATED':
                controls = [u for u in units if cohorts[u] is None]
            else:
                controls = [u for u in units if cohorts[u] != g and
                            (cohorts[u] is None or cohorts[u] > end+A)]
            cell = dict(cohort=g, start_period=begin, end_period=end,
                        lead_to_treatment=g-end, anticipation_periods=A,
                        treated_units=len(treated), control_units=len(controls),
                        status='COMPUTED_ASSUMPTIONS_UNVERIFIED' if controls else 'BLOCKED',
                        blocker=None if controls else 'NO_UNEXPOSED_CONTROLS',
                        treated_change_exact=None, control_change_exact=None,
                        placebo_contrast_exact=None)
            if controls:
                y_treated = _average([values[u,end] - values[u,begin] for u in treated])
                y_controls = _average([values[u,end] - values[u,begin] for u in controls])
                contrast = y_treated-y_controls
                cell.update(treated_change_exact=str(y_treated),
                            control_change_exact=str(y_controls),
                            placebo_contrast_exact=str(contrast))
                by_lead[g-end].append((g,contrast,len(treated)))
            cells.append(cell)
    supported=sum(c['placebo_contrast_exact'] is not None for c in cells)
    lead_summary=[]
    for lead in sorted(by_lead):
        entries=by_lead[lead]
        total=sum(n for _,_,n in entries)
        result=sum((effect*n for _,effect,n in entries),F(0))/total
        lead_summary.append(dict(lead_to_treatment=lead,
                                 placebo_contrast_exact=str(result),
                                 cohorts=[g for g,_,_ in entries],
                                 cohort_weights_exact=[str(F(n,total)) for _,_,n in entries],
                                 treated_units_in_support=total))
    status=('NOT_EVALUABLE_NO_CLEAN_PREPERIOD_TRANSITIONS' if not cells else
            'NOT_EVALUABLE_NO_ELIGIBLE_CONTROLS' if not supported else
            'PARTIAL_COMPUTATION_ASSUMPTIONS_UNVERIFIED' if supported<len(cells) else
            'COMPUTED_ASSUMPTIONS_UNVERIFIED')
    return dict(schema='societybox-did-pretrend-placebo-result-v1',status=status,
                estimand='UNADJUSTED_ONE_PERIOD_PRETREATMENT_DIAGNOSTIC_NOT_CAUSAL_TEST',
                comparison=comparison,anticipation_periods=A,
                source_declared_sha256=spec['source_declared_sha256'],
                source_authentication='NOT_ESTABLISHED',
                parallel_trends='NOT_VERIFIED_EVEN_IF_PLACEBOS_ARE_ZERO',
                no_anticipation='DECLARED_NOT_VERIFIED',
                common_trend_inference='NOT_COMPUTED',
                p_values='NOT_COMPUTED',standard_errors='NOT_COMPUTED',
                original_upstream_parity='NOT_RUN',
                cohort_placebo_cells=cells, lead_time_aggregates=lead_summary,
                supported_cells=supported,blocked_cells=len(cells)-supported)
