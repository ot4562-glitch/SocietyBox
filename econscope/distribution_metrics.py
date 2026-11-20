"""Exact weighted income-distribution and FGT poverty accounting.

Population weights are researcher-declared *relative population weights*.
They do not verify representativeness, survey design, monetary comparability,
source authentication, causal poverty reduction, or inference uncertainty.
Nonnegative incomes are required: standard Gini and Lorenz shares would lose
usual [0,1] interpretation with negative totals/incomes.
"""
from fractions import Fraction as F
import re

from .execution_control import checkpoint

_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,39}\Z', re.ASCII)
_UNIT = re.compile(r'[A-Za-z0-9][A-Za-z0-9_ ./%-]{0,39}\Z', re.ASCII)
_NUMBER = re.compile(r'(?:0|[1-9][0-9]{0,11})(?:\.[0-9]{1,9})?\Z', re.ASCII)
_SHA = re.compile(r'[0-9a-f]{64}\Z', re.ASCII)


def _fail(reason):
    raise ValueError('DISTRIBUTION_METRICS_' + reason)


def _number(raw):
    if type(raw) is not str or not _NUMBER.fullmatch(raw):
        _fail('NONNEGATIVE_DECIMAL_STRING_REQUIRED')
    return F(raw)


def _validated(spec):
    fields={'schema','data_kind','source_declared_sha256','rights',
            'income_unit','poverty_line','rows'}
    if type(spec) is not dict or set(spec)!=fields or spec['schema']!='societybox-distribution-metrics-v1':
        _fail('SCHEMA_INVALID')
    if type(spec['data_kind']) is not str or spec['data_kind'] not in ('SYNTHETIC','RESEARCHER_SUPPLIED'):
        _fail('DATA_KIND_INVALID')
    if type(spec['source_declared_sha256']) is not str or not _SHA.fullmatch(spec['source_declared_sha256']):
        _fail('SOURCE_DIGEST_REQUIRED')
    if spec['rights']!={'local_processing':'PERMITTED','report_export':'PERMITTED'}:
        _fail('RIGHTS_REQUIRED')
    if type(spec['income_unit']) is not str or not _UNIT.fullmatch(spec['income_unit']):
        _fail('UNIT_INVALID')
    line=_number(spec['poverty_line'])
    if not line: _fail('POSITIVE_POVERTY_LINE_REQUIRED')
    rows=spec['rows']
    if type(rows) is not list or not 2<=len(rows)<=1024:
        _fail('ROW_BUDGET')
    data=[]; seen=set()
    for n,row in enumerate(rows):
        if n%64==0: checkpoint()
        if type(row) is not dict or set(row)!={'unit','income','population_weight'}:
            _fail('ROW_INVALID')
        unit=row['unit']
        if type(unit) is not str or not _ID.fullmatch(unit):
            _fail('UNIT_ID_INVALID')
        if unit in seen: _fail('UNIT_ID_DUPLICATE')
        seen.add(unit)
        income=_number(row['income']); weight=_number(row['population_weight'])
        if weight<=0: _fail('POSITIVE_POPULATION_WEIGHT_REQUIRED')
        data.append((income,unit,weight))
    return sorted(data),line


def distribution_metrics(spec):
    """Return exact Gini, Lorenz and FGT(0,1,2), plus nearest-rank median.

    FGT indices require a declared positive poverty line in the income unit;
    they are finite arithmetic measures, not valid sampling variance or
    policy impact estimates. Zero total income blocks Gini/Lorenz, while
    poverty accounting remains available.
    """
    data,line=_validated(spec)
    population_weight=sum((weight for _,_,weight in data),F(0))
    income_total=sum((income*weight for income,_,weight in data),F(0))
    weighted_mean=income_total/population_weight
    poor_weight=F(0)
    poverty_gap=F(0)
    poverty_severity=F(0)
    cumulative_weight=F(0)
    cumulative_income=F(0)
    median=None
    lorenz=([dict(population_share_exact='0',income_share_exact='0')]
            if income_total>0 else [])
    area_twice=F(0)
    previous_pop=previous_inc=F(0)
    for index,(income,_,weight) in enumerate(data):
        if index%64==0: checkpoint()
        if income<line:
            gap=(line-income)/line
            poor_weight+=weight
            poverty_gap+=weight*gap
            poverty_severity+=weight*gap*gap
        cumulative_weight+=weight
        cumulative_income+=income*weight
        if median is None and 2*cumulative_weight>=population_weight:
            median=income
        if income_total:
            pop=cumulative_weight/population_weight
            inc=cumulative_income/income_total
            area_twice+=(pop-previous_pop)*(inc+previous_inc)
            lorenz.append(dict(population_share_exact=str(pop),income_share_exact=str(inc)))
            previous_pop,previous_inc=pop,inc
    if cumulative_weight!=population_weight or cumulative_income!=income_total:
        _fail('INTERNAL_SUM_MISMATCH')
    gini=1-area_twice if income_total else None
    if gini is not None and not 0<=gini<=1:
        _fail('GINI_RANGE_INVALID')
    return dict(schema='societybox-distribution-metrics-result-v1',
                status='COMPUTED_ACCOUNTING_ONLY' if income_total else 'PARTIAL_ZERO_MEAN_GINI_UNDEFINED',
                source_declared_sha256=spec['source_declared_sha256'],
                source_authentication='NOT_ESTABLISHED',data_kind=spec['data_kind'],
                income_unit=spec['income_unit'],observations=len(data),
                weight_semantics='DECLARED_POSITIVE_RELATIVE_POPULATION_WEIGHTS_NOT_SURVEY_DESIGN_VALIDATION',
                income_comparability='NOT_VERIFIED',survey_sampling_uncertainty='NOT_COMPUTED',
                policy_causality='NOT_ESTABLISHED',original_upstream_parity='NOT_RUN',
                poverty_line_exact=str(line), total_population_weight_exact=str(population_weight),
                weighted_income_total_exact=str(income_total),weighted_mean_exact=str(weighted_mean),
                weighted_median_exact=str(median),median_convention='WEIGHTED_LOWER_NEAREST_RANK',
                gini_exact=None if gini is None else str(gini),
                gini_status='UNDEFINED_ZERO_MEAN' if gini is None else 'COMPUTED_NONNEGATIVE_INCOME_ONLY',
                lorenz=lorenz,
                fgt_0_headcount_exact=str(poor_weight/population_weight),
                fgt_1_gap_exact=str(poverty_gap/population_weight),
                fgt_2_severity_exact=str(poverty_severity/population_weight),
                poor_population_weight_exact=str(poor_weight),
                poverty_definition='STRICTLY_BELOW_LINE_NO_EQUIVALENCE_SCALE_OR_DEFLATOR')
