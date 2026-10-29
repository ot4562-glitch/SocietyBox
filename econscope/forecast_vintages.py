"""Bind daily forecast calendars to declared revision histories, without fitting.

Legacy single-snapshot forecasts remain separate. In revision mode observations
contain dates only: there is no parallel set of numeric cells to accidentally
use instead of the approved ledger. Timestamp declarations are not attestation.
"""
from copy import deepcopy
from datetime import date, datetime, timedelta
from fractions import Fraction

from . import research_vintages as vintages
from .mechanism_vintage import _instant, _date
from .workspace import canonical, digest


def _comparison(parameters, instant):
    binding = parameters['revision_history']
    return dict(schema='societybox-vintage-comparison-v1', ledger=binding['ledger'],
                earlier_as_of=instant, later_as_of=instant, availability_mode=binding['availability_mode'])


def validation_screen(parameters):
    """Return zero placeholders ONLY for legacy schema validation, never a fit.

    No snapshot selection here. Execution materializes a new row array from
    each actual declared cutoff; the placeholder array is never fitted/scored.
    """
    try:
        binding = parameters['revision_history']; screen = parameters['screen']
        if type(binding) is not dict or set(binding) != {'ledger', 'availability_mode', 'origin_as_of', 'evaluation_as_of'}:
            raise ValueError('FORECAST_VINTAGE_BINDING_INVALID')
        origins = parameters['origins']; cutoffs = binding['origin_as_of']
        if (type(origins) is not list or not origins or any(type(day) is not str for day in origins)
                or type(cutoffs) is not dict or set(cutoffs) != set(origins)):
            raise ValueError('FORECAST_VINTAGE_ORIGIN_BINDINGS_REQUIRED')
        previous = None
        evaluation = _instant(binding['evaluation_as_of'])
        if datetime.fromisoformat(binding['evaluation_as_of'].replace('Z', '+00:00')).date() != _date(parameters['evaluation_as_of_day']):
            raise ValueError('FORECAST_VINTAGE_EVALUATION_DAY_MISMATCH')
        for day in origins:
            instant = _instant(cutoffs[day])
            if (datetime.fromisoformat(cutoffs[day].replace('Z', '+00:00')).date() != _date(day)
                    or instant > evaluation or (previous is not None and instant <= previous)):
                raise ValueError('FORECAST_VINTAGE_ORIGIN_INSTANT_INVALID')
            previous = instant
        vintages.validate(_comparison(parameters, binding['evaluation_as_of']))
        ledger = binding['ledger']; data = screen['study']; units = data['series_units']
        expected_kind = 'SYNTHETIC' if ledger['data_kind'] == 'SYNTHETIC' else 'DECLARED_PUBLIC_OBSERVATIONS'
        if data['data_kind'] != expected_kind or data['source_declared_sha256'] != digest(canonical(ledger)):
            raise ValueError('FORECAST_VINTAGE_LEDGER_BINDING_MISMATCH')
        if (screen['rights'] != ledger['rights'] or type(units) is not dict
                or set(units) != set(ledger['series']) or any(
                    ledger['series'][name]['unit'] != unit or ledger['series'][name]['frequency'] != 'day'
                    for name, unit in units.items())):
            raise ValueError('FORECAST_VINTAGE_SERIES_OR_RIGHTS_MISMATCH')
        calendar = data['observations']
        if (type(calendar) is not list or any(type(row) is not dict or set(row) != {'day'} for row in calendar)):
            raise ValueError('FORECAST_VINTAGE_DATE_ONLY_CALENDAR_REQUIRED')
        days = {_date(row['day']).isoformat() for row in calendar}
        for row in ledger['revisions']:
            if row['period_start'] != row['period_end'] or row['period_start'] not in days:
                raise ValueError('FORECAST_VINTAGE_DAILY_CALENDAR_MISMATCH')
            if (row['source_release_at'] is not None and
                    datetime.fromisoformat(row['source_release_at'].replace('Z', '+00:00')).date() < _date(row['period_end'])):
                raise ValueError('FORECAST_VINTAGE_OBSERVATION_RELEASE_INVALID')
        normalized = deepcopy(screen)
        normalized['study']['observations'] = [dict(day=row['day'],
            values={name: '0' for name in units}, available_on={name: row['day'] for name in units}) for row in calendar]
        return normalized
    except (KeyError, TypeError, AttributeError):
        raise ValueError('FORECAST_VINTAGE_BINDING_INVALID') from None


def prepare(parameters):
    """Validated immutable groups reused across approved origin selections."""
    return vintages._validate(_comparison(parameters, parameters['revision_history']['evaluation_as_of']))


def select(parameters, groups, instant, calendar_rows, day):
    binding = parameters['revision_history']
    snapshot = vintages._snapshot(groups, instant, binding['availability_mode'])
    vintages._verify_selection(_comparison(parameters, instant), snapshot)
    # Only the calendar information used at this origin belongs to its hash.
    # The full ledger stays in study.json, independently bound by the input SHA.
    snapshot['cells'] = [row for row in snapshot['cells'] if _date(row['period_end']) <= day]
    snapshot['unresolved'] = [row for row in snapshot['unresolved'] if _date(row['period_end']) <= day]
    selected = {(row['period_start'], row['series_id']): row for row in snapshot['cells']}
    materialized = []
    for observation_day, original_values, _ in calendar_rows:
        values, availability = {}, {}
        for name in original_values:
            cell = selected.get((observation_day.isoformat(), name))
            values[name] = None if cell is None else Fraction(cell['value_decimal'])
            # The exact gate was applied above. This date is an eligibility
            # marker for the existing daily fitter, not the release timestamp.
            availability[name] = date.max if cell is None else day
        materialized.append((observation_day, values, availability))
    return materialized, digest(canonical(snapshot)), selected


def binding_receipt(parameters):
    binding = parameters['revision_history']
    return dict(ledger_sha256=digest(canonical(binding['ledger'])), availability_mode=binding['availability_mode'],
        origin_as_of=deepcopy(binding['origin_as_of']), evaluation_as_of=binding['evaluation_as_of'],
        source_authentication='NOT_ESTABLISHED', historical_researcher_access='NOT_ESTABLISHED',
        data_values='ONLY_ORIGIN_SPECIFIC_LEDGER_SELECTIONS_NOT_CALENDAR_PLACEHOLDERS')


def example_parameters():
    """Authored toy history used by the existing forecast family in the UI."""
    days=[(date(2025,1,1)+timedelta(days=i)).isoformat() for i in range(24)]
    x=[i+(i%3)**2 for i in range(24)]; revisions=[]
    for i,day in enumerate(days):
        for name,value in (('x',x[i]),('y',3+2*x[max(0,i-1)])):
            revisions.append(dict(series_id=name,period_start=day,period_end=day,
                revision_id='first',revision_sequence=1,value_decimal=str(value),source_sha256=None,
                source_release_at=day+'T18:00:00Z',vendor_recorded_at=day+'T18:05:00Z',local_acquired_at=day+'T19:00:00Z'))
    for index,value,release in ((5,'20',days[17]),(16,'40',days[22])):
        row=deepcopy(next(r for r in revisions if r['series_id']=='y' and r['period_start']==days[index]))
        row.update(revision_id='revision',revision_sequence=2,value_decimal=value,
            source_release_at=release+'T18:00:00Z',vendor_recorded_at=release+'T18:05:00Z',local_acquired_at=release+'T19:00:00Z')
        revisions.append(row)
    ledger=dict(schema='societybox-revision-ledger-v1',data_kind='SYNTHETIC',
        source=dict(id='authored-vintage-forecast',license='CC0-1.0',
                    basis='Invented daily observations and revisions, not an actual provider or authenticated history.'),
        rights=dict(local_processing='PERMITTED',report_export='PERMITTED'),
        series={name:dict(unit='index',frequency='day',timezone='UTC',seasonal_adjustment='UNADJUSTED',
                         statistical_definition='Authored '+name) for name in ('x','y')},revisions=revisions)
    return dict(screen=dict(schema='societybox-inverse-screen-v1',
        study=dict(schema='societybox-dynamic-inverse-v1',data_kind='SYNTHETIC',source_declared_sha256=digest(canonical(ledger)),
            target='y',series_units={'x':'index','y':'index'},training_end_day=days[15],observations=[{'day':day} for day in days]),
        rights=deepcopy(ledger['rights']),search=dict(lags=[1,2],include_target_history=False)),
        coefficient_bounds={name:['-1000','1000'] for name in ('intercept','x[t-1]','x[t-2]')},
        preprocessing=dict(method='WINSORIZE_TRAIN_QUANTILES',lower_probability='1/4',upper_probability='3/4',
                           fit_scope='ORIGIN_AVAILABLE_COMMON_TRAINING_ROWS_ONLY'),
        origins=[days[15],days[18]],horizons=[1],evaluation_as_of_day=days[23],
        revision_history=dict(ledger=ledger,availability_mode='SOURCE_RELEASE_AS_OF',
            origin_as_of={days[i]:days[i]+'T20:00:00Z' for i in (15,18)},evaluation_as_of=days[23]+'T20:00:00Z'))
