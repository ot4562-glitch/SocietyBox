"""Offline economic study review/run/verify. Data only; no upstream code intake.

Run requires the exact review digest and an explicit fallback or pinned upstream
choice in the study. Source licenses below are operator declarations, never clearance.
Portable verification recomputes with the same source tree, not a clean install
or externally independent scientific replication.
"""
import argparse
import csv
from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile

from .macro_solver import optional_macro
from .optimization_policy import optional_optimization
from .feature_catalog import describe_feature
from .execution_control import checkpoint

LOCALES = ('en', 'zh-Hans', 'ko', 'ja')
LIMIT = 65536
ARTIFACT_LIMIT = 2*1024*1024

# Implemented study entry points, not a product-wide coverage ceiling. UI and
# callers discover this same registry instead of maintaining smaller allowlists.
_STUDY_REQUIREMENTS = {
    'input_output': (), 'survey_replicate': (), 'vintage_comparison': (), 'consumption': (), 'allocation': (),
    'service_queue': ('sources',), 'service_staffing': ('sources',),
    'supply_network': ('sources',), 'supply_capacity': ('sources',), 'supply_cost': ('sources',),
    'mathematical_program': ('runtime',), 'dynamic_inverse': ('runtime',),
    'coefficient_identification': ('runtime',), 'forecast_validation': ('runtime',),
    'regression_sensitivity': ('runtime',), 'fixed_effects_regression': ('runtime',),
}


def study_families(locale='en'):
    return {family: dict(family=family, requires=list(requirements), **describe_feature(family, locale))
            for family, requirements in _STUDY_REQUIREMENTS.items()}


def _bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _read(path, limit=LIMIT):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError('STUDY_FILE_INVALID')
    with path.open('rb') as stream:
        raw = stream.read(limit+1)
    if len(raw) > limit:
        raise ValueError('STUDY_FILE_TOO_LARGE')
    return raw


def _parse(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('STUDY_DUPLICATE_KEY')
            result[key] = value
        return result
    def invalid(_):
        raise ValueError('STUDY_NONFINITE_NUMBER')
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def _text(value, cap):
    return type(value) is str and 0 < len(value.strip()) <= cap


def _code_hash():
    names = ('engine_study.py', 'research_review_entry.py', 'macro_solver.py',
             'optimization_policy.py', 'engine_prerequisites.py',
             'simpy_queue.py', 'simpy_queue_worker.py', 'simpy_staffing.py', 'strict_json.py',
             'network_paths.py', 'network_paths_worker.py', 'network_capacity.py', 'network_cost_flow.py',
             'highs_native.py', 'highs_native_worker.py', 'pyomo_model.py', 'pyomo_model_worker.py',
             'feature_catalog.py', 'inverse_native.py', 'inverse_screen.py', 'mechanism_dynamic.py',
             'inverse_numeric_verifier.py', 'inverse_investigation.py', 'research_evidence.py', 'workspace.py',
             'inverse_identification.py', 'forecast_validation.py', 'regression_sensitivity.py', 'regression_covariance_verifier.py',
             'input_output_impact.py', 'survey_replicate_variance.py', 'research_vintages.py', 'mechanism_vintage.py', 'forecast_vintages.py', 'forecast_preprocessing.py', 'fixed_effects.py', 'fixed_effects_verifier.py',
             'research_chat.py', 'research_chat_ui.py', 'research_chat_service.py', 'standalone_service.py', 'execution_control.py', 'research_jobs.py')
    return _sha(_bytes({name: _sha(Path(__file__).with_name(name).read_bytes()) for name in names}))


def review(spec, *, runtime=None):
    fields = {'schema', 'family', 'question', 'source', 'rights', 'execution', 'request'}
    if type(spec) is not dict or set(spec) != fields or spec['schema'] != 'societybox-engine-study-v1':
        raise ValueError('ENGINE_STUDY_SCHEMA_INVALID')
    if type(spec['family']) is not str or spec['family'] not in _STUDY_REQUIREMENTS:
        raise ValueError('ENGINE_STUDY_FAMILY_UNSUPPORTED')
    if len(_bytes(spec)) > LIMIT or not _text(spec['question'], 2000):
        raise ValueError('ENGINE_STUDY_SIZE_OR_QUESTION_INVALID')
    source = spec['source']
    if (type(source) is not dict or set(source) != {'id', 'license', 'kind', 'basis'} or
            not _text(source['id'], 200) or not _text(source['basis'], 2000) or
            source['license'] not in ('CC0-1.0', 'CC-BY-4.0', 'OPERATOR_OWNED') or
            source['kind'] not in ('SYNTHETIC', 'RESEARCHER_SUPPLIED')):
        raise ValueError('ENGINE_STUDY_SOURCE_DECLARATION_INVALID')
    if spec['rights'] != {'local_processing': 'PERMITTED', 'report_export': 'PERMITTED'}:
        raise ValueError('ENGINE_STUDY_RIGHTS_REQUIRED')
    staffing_family = spec['family'] == 'service_staffing'
    queue_family = spec['family'] in ('service_queue', 'service_staffing')
    capacity_family = spec['family'] == 'supply_capacity'
    cost_family = spec['family'] == 'supply_cost'
    network_family = spec['family'] in ('supply_network', 'supply_capacity', 'supply_cost')
    native_program = spec['family'] == 'mathematical_program'
    native_inverse = spec['family'] == 'dynamic_inverse'
    native_identification = spec['family'] == 'coefficient_identification'
    native_forecast = spec['family'] == 'forecast_validation'
    native_regression = spec['family'] == 'regression_sensitivity'
    native_fixed_effects = spec['family'] == 'fixed_effects_regression'
    highs_family = native_program or native_inverse or native_identification or native_forecast or native_regression or native_fixed_effects
    pyomo_program = native_program and type(spec['request']) is dict and spec['request'].get('engine') == 'Pyomo+HiGHS'
    exact_family = spec['family'] in ('input_output', 'survey_replicate', 'vintage_comparison')
    expected_mode = ('EXPLICIT_INDEPENDENT_EXACT' if exact_family else 'EXPLICIT_PINNED_PYOMO_HIGHS' if pyomo_program else
                     'EXPLICIT_PINNED_HIGHS' if highs_family else 'EXPLICIT_PINNED_NETWORKX' if network_family else
                     'EXPLICIT_PINNED_SIMPY' if queue_family else 'EXPLICIT_OFFLINE_FALLBACK')
    if spec['execution'] != expected_mode:
        raise ValueError('ENGINE_STUDY_EXECUTION_CHOICE_REQUIRED')
    pin = None
    if exact_family:
        from . import input_output_impact, survey_replicate_variance, research_vintages
        request = spec['request']
        model, core = {'input_output': ('static_leontief', input_output_impact),
                       'survey_replicate': ('declared_replicate_mean', survey_replicate_variance),
                       'vintage_comparison': ('declared_revision_comparison', research_vintages)}[spec['family']]
        if (type(request) is not dict or set(request) != {'engine', 'model', 'parameters'} or
                request['engine'] != 'SocietyBoxExact' or request['model'] != model):
            raise ValueError('ENGINE_STUDY_EXACT_REQUEST_INVALID')
        core.validate(request['parameters'])
        if spec['family'] == 'vintage_comparison':
            ledger = request['parameters']['ledger']
            if (ledger['data_kind'] != source['kind'] or ledger['source']['id'] != source['id']
                    or ledger['source']['license'] != source['license']):
                raise ValueError('ENGINE_STUDY_VINTAGE_SOURCE_MISMATCH')
    elif native_fixed_effects:
        from .fixed_effects import validate
        from .highs_native import runtime_pin
        request=spec['request']
        if (type(request) is not dict or set(request)!={'engine','model','parameters'} or
                request['engine']!='HiGHS+ExactFixedEffects' or request['model']!='declared_weighted_fixed_effects'):
            raise ValueError('ENGINE_STUDY_FIXED_EFFECTS_REQUEST_INVALID')
        validate(request['parameters'])
        if (request['parameters']['data_kind']=='SYNTHETIC') != (source['kind']=='SYNTHETIC'):
            raise ValueError('ENGINE_STUDY_FIXED_EFFECTS_SOURCE_KIND_MISMATCH')
        if runtime is None:raise ValueError('HIGHS_PINNED_RUNTIME_REQUIRED')
        pin=runtime_pin(runtime)
    elif native_regression:
        from .regression_sensitivity import validate
        from .highs_native import runtime_pin
        request = spec['request']
        if (type(request) is not dict or set(request) != {'engine', 'model', 'parameters'} or
                request['engine'] != 'HiGHS+ExactWLS' or request['model'] != 'declared_lagged_weighted_regression'):
            raise ValueError('ENGINE_STUDY_REGRESSION_REQUEST_INVALID')
        validate(request['parameters'])
        kind = request['parameters']['data']['data_kind']
        if (kind == 'SYNTHETIC') != (source['kind'] == 'SYNTHETIC'):
            raise ValueError('ENGINE_STUDY_REGRESSION_SOURCE_KIND_MISMATCH')
        if runtime is None: raise ValueError('HIGHS_PINNED_RUNTIME_REQUIRED')
        pin = runtime_pin(runtime)
    elif native_forecast:
        from .forecast_validation import validate
        from .highs_native import runtime_pin
        request = spec['request']
        if (type(request) is not dict or set(request) != {'engine', 'model', 'parameters'} or
                request['engine'] != 'HiGHS+ExactForecast' or request['model'] != 'declared_originwise_lag_forecast'):
            raise ValueError('ENGINE_STUDY_FORECAST_REQUEST_INVALID')
        validate(request['parameters'])
        kind = request['parameters']['screen']['study']['data_kind']
        if (kind == 'SYNTHETIC') != (source['kind'] == 'SYNTHETIC'):
            raise ValueError('ENGINE_STUDY_FORECAST_SOURCE_KIND_MISMATCH')
        if 'revision_history' in request['parameters']:
            declared=request['parameters']['revision_history']['ledger']['source']
            if declared['id'] != source['id'] or declared['license'] != source['license']:
                raise ValueError('ENGINE_STUDY_FORECAST_REVISION_SOURCE_MISMATCH')
        if runtime is None: raise ValueError('HIGHS_PINNED_RUNTIME_REQUIRED')
        pin = runtime_pin(runtime)
    elif native_identification:
        from .inverse_identification import validate
        from .highs_native import runtime_pin
        request = spec['request']
        if (type(request) is not dict or set(request) != {'engine', 'model', 'parameters'} or
                request['engine'] != 'HiGHS+ExactIdentification' or request['model'] != 'finite_prediction_equivalence'):
            raise ValueError('ENGINE_STUDY_IDENTIFICATION_REQUEST_INVALID')
        validate(request['parameters'])
        kind = request['parameters']['data']['data_kind']
        if (kind == 'SYNTHETIC') != (source['kind'] == 'SYNTHETIC'):
            raise ValueError('ENGINE_STUDY_IDENTIFICATION_SOURCE_KIND_MISMATCH')
        if runtime is None: raise ValueError('HIGHS_PINNED_RUNTIME_REQUIRED')
        pin = runtime_pin(runtime)
    elif native_inverse:
        from .inverse_native import validate
        from .highs_native import runtime_pin
        request = spec['request']
        if (type(request) is not dict or set(request) != {'engine', 'model', 'parameters'} or
                request['engine'] != 'HiGHS+ExactInverse' or request['model'] != 'lagged_regression_screen'):
            raise ValueError('ENGINE_STUDY_INVERSE_REQUEST_INVALID')
        validate(request['parameters'])
        kind = request['parameters']['screen']['study']['data_kind']
        if (kind == 'SYNTHETIC') != (source['kind'] == 'SYNTHETIC'):
            raise ValueError('ENGINE_STUDY_INVERSE_SOURCE_KIND_MISMATCH')
        if runtime is None: raise ValueError('HIGHS_PINNED_RUNTIME_REQUIRED')
        pin = runtime_pin(runtime)
    elif native_program:
        from .highs_native import validate, runtime_pin
        request = spec['request']
        if (type(request) is not dict or set(request) != {'engine', 'model', 'parameters'} or
                request['engine'] not in ('HiGHS', 'Pyomo+HiGHS') or request['model'] != 'bounded_mathematical_program'):
            raise ValueError('ENGINE_STUDY_HIGHS_REQUEST_INVALID')
        validate(request['parameters'])
        if runtime is None: raise ValueError('HIGHS_PINNED_RUNTIME_REQUIRED')
        pin = runtime_pin(runtime)
    elif spec['family'] == 'consumption':
        optional_macro(spec['request'])
    elif spec['family'] == 'allocation':
        optional_optimization(spec['request'])
    elif cost_family:
        from .network_cost_flow import validate
        request = spec['request']
        if (type(request) is not dict or set(request) != {'engine', 'model', 'parameters'} or
                request['engine'] != 'NetworkX' or request['model'] != 'directed_minimum_cost_flow'):
            raise ValueError('ENGINE_STUDY_COST_FLOW_REQUEST_INVALID')
        validate(request['parameters'])
    elif capacity_family:
        from .network_capacity import validate
        request = spec['request']
        if (type(request) is not dict or set(request) != {'engine', 'model', 'parameters'} or
                request['engine'] != 'NetworkX' or request['model'] != 'directed_supply_capacity'):
            raise ValueError('ENGINE_STUDY_CAPACITY_REQUEST_INVALID')
        validate(request['parameters'])
    elif network_family:
        from .network_paths import validate
        request = spec['request']
        if (type(request) is not dict or set(request) != {'engine', 'model', 'parameters'} or
                request['engine'] != 'NetworkX' or request['model'] != 'directed_delivery_paths'):
            raise ValueError('ENGINE_STUDY_NETWORK_REQUEST_INVALID')
        validate(request['parameters'])
    elif staffing_family:
        from .simpy_staffing import validate_study
        request = spec['request']
        if (type(request) is not dict or set(request) != {'engine', 'model', 'parameters'} or
                request['engine'] != 'SimPy' or request['model'] != 'finite_fifo_staffing'):
            raise ValueError('ENGINE_STUDY_STAFFING_REQUEST_INVALID')
        validate_study(request['parameters'])
    elif queue_family:
        from .simpy_queue import validate_spec
        request = spec['request']
        if (type(request) is not dict or set(request) != {'engine', 'model', 'parameters'} or
                request['engine'] != 'SimPy' or request['model'] != 'finite_fifo_queue'):
            raise ValueError('ENGINE_STUDY_QUEUE_REQUEST_INVALID')
        validate_spec(request['parameters'])
    else:
        raise ValueError('ENGINE_STUDY_FAMILY_UNSUPPORTED')
    binding = {'study_sha256': _sha(_bytes(spec)), 'engine_code_sha256': _code_hash()}
    if pin is not None: binding['runtime_sha256'] = _sha(_bytes(pin))
    if pyomo_program:
        from .pyomo_model import ARCHIVE_SHA256
        binding['modeling_archive_sha256'] = ARCHIVE_SHA256
    result = {'schema': 'societybox-engine-study-review-v1', 'binding': binding,
            'review_sha256': _sha(_bytes(binding)), 'approval': 'NOT_GRANTED',
            'source_clearance': 'OPERATOR_DECLARATION_NOT_VERIFIED',
            'llm': 'NO_LLM', 'native_execution': 'NOT_RUN', 'native_parity': 'NOT_RUN',
            'product_feature': describe_feature(spec['family']),
            'os_isolation': 'NOT_ATTESTED', 'scientific_acceptance': 'NOT_ESTABLISHED'}
    if pin is not None: result['runtime_pin'] = pin
    if native_forecast:
        parameters = spec['request']['parameters']
        result['forecast_plan_readiness'] = {
            'scope': 'VALIDATED_PLAN_AND_LAG_SUPPORT_ONLY_NO_ESTIMATION',
            'requested_cells': len(parameters['origins']) * len(parameters['horizons']),
            'horizons_without_declared_lag_support': [h for h in parameters['horizons'] if h > max(parameters['screen']['search']['lags'])],
            'historical_vintage_authentication': 'NOT_ESTABLISHED', 'probabilistic_calibration': 'NOT_RUN_POINT_FORECASTS_ONLY'}
        if 'revision_history' in parameters:
            from .forecast_vintages import binding_receipt
            result['forecast_plan_readiness']['revision_history'] = binding_receipt(parameters)
        if 'preprocessing' in parameters:
            result['forecast_plan_readiness']['preprocessing'] = dict(parameters['preprocessing'])
    if pyomo_program:
        from .pyomo_model import COMMIT, VERSION
        result['modeling_source_pin'] = {'name': 'Pyomo', 'commit': COMMIT, 'version': VERSION,
                                        'archive_sha256': ARCHIVE_SHA256,
                                        'source_policy': 'REVERIFY_PIN_AND_LICENSE_BEFORE_IMPORT'}
    if queue_family:
        from .simpy_queue import ARCHIVE_SHA256, COMMIT
        result['upstream_pin'] = {'archive_sha256': ARCHIVE_SHA256, 'commit': COMMIT,
                                  'source_policy': 'REVERIFY_PIN_AND_LICENSE_BEFORE_IMPORT'}
    if network_family:
        from .network_paths import ARCHIVE_SHA256, COMMIT
        result['upstream_pin'] = {'archive_sha256': ARCHIVE_SHA256, 'commit': COMMIT,
                                 'source_policy': 'REVERIFY_PIN_AND_LICENSE_BEFORE_IMPORT'}
    return result


def _calculate(spec, sources=None, runtime=None):
    if spec['family'] in ('input_output', 'survey_replicate', 'vintage_comparison'):
        return _exact_result(spec)
    if spec['family'] == 'regression_sensitivity':
        from .regression_sensitivity import run
        return run(spec['request']['parameters'], runtime)
    if spec['family'] == 'fixed_effects_regression':
        from .fixed_effects import run
        return run(spec['request']['parameters'], runtime)
    if spec['family'] == 'forecast_validation':
        from .forecast_validation import run
        return run(spec['request']['parameters'], runtime)
    if spec['family'] == 'supply_cost':
        from .network_cost_flow import run
        return run(spec['request']['parameters'], sources)
    if spec['family'] == 'coefficient_identification':
        from .inverse_identification import run
        return run(spec['request']['parameters'], runtime)
    if spec['family'] == 'dynamic_inverse':
        from .inverse_native import run
        return run(spec['request']['parameters'], runtime)
    if spec['family'] == 'mathematical_program':
        if spec['request']['engine'] == 'Pyomo+HiGHS':
            from .pyomo_model import run
            return run(spec['request']['parameters'], sources, runtime)
        from .highs_native import run
        return run(spec['request']['parameters'], runtime)
    if spec['family'] == 'supply_capacity':
        from .network_capacity import run
        return run(spec['request']['parameters'], sources)
    if spec['family'] == 'service_staffing':
        from .simpy_staffing import run_study
        if sources is None:
            raise ValueError('ENGINE_STUDY_PINNED_SOURCES_REQUIRED')
        raw = run_study(spec['request']['parameters'], sources)
        result = {k: v for k, v in raw.items() if k != 'runs'}
        result.update(engine='SIMPY_PINNED_FIFO_STAFFING', mode='NO_LLM',
                      native_execution='EXECUTED', native_parity='EXACT_MATCH_FIFO_INTERVAL_ORACLE',
                      os_isolation='NOT_ATTESTED',
                      runs=[_portable_queue(r) for r in raw['runs']])
        return result
    if spec['family'] == 'supply_network':
        from .network_paths import run as run_network
        return run_network(spec['request']['parameters'], sources)
    if spec['family'] == 'consumption':
        return optional_macro(spec['request'], allow_oracle=True)['fallback']
    if spec['family'] == 'service_queue':
        return _queue_result(spec['request']['parameters'], sources)
    return optional_optimization(spec['request'], allow_fallback=True)['fallback']


def _exact_result(spec):
    """Reuse the exact cores and the shared evidence contracts, without new ledgers."""
    from . import input_output_impact, survey_replicate_variance, research_vintages
    from .research_evidence import EvidenceLedger, UnknownRegistry, make_obligation, evaluate_coverage
    core = {'input_output': input_output_impact, 'survey_replicate': survey_replicate_variance,
            'vintage_comparison': research_vintages}[spec['family']]
    result = core.calculate(spec['request']['parameters'])
    evidence = result['evidence']
    ledger = EvidenceLedger([evidence]); registry = UnknownRegistry(ledger)
    unknowns, obligations = [], []
    for dimension in evidence['coverage']['unknown_dimensions']:
        scope = {'family': spec['family'], 'dimension': dimension,
                 'input_sha256': result['input_sha256']}
        unknown = registry.register('Establish ' + dimension, scope,
                                    'INDEPENDENT_EXTERNAL_REVIEW', 'SOURCE_AND_METHOD_VALIDATION')
        unknowns.append(unknown)
        obligations.append(make_obligation(
            claim_id=dimension, subject_sha256=result['input_sha256'], scope=scope,
            required_authority='INDEPENDENT_EXTERNAL_REVIEW', required_proof_kind='SOURCE_AND_METHOD_VALIDATION',
            verifier_id='external.' + dimension, dependencies=[], unknown_ids=[unknown['unknown_id']],
            blocking_reasons=[], next_probe='Obtain independent source/method evidence for ' + dimension))
    result.update(engine='SOCIETY_BOX_INDEPENDENT_EXACT', mode='NO_LLM', native_parity='NOT_RUN',
                  independent_verification=(result['oracle']['status'] if spec['family'] == 'input_output'
                                            else result['exact_oracle']),
                  evidence=ledger.records(), unknowns=unknowns, obligations=obligations,
                  coverage=evaluate_coverage(obligations, ledger, registry))
    return result


def _queue_result(parameters, sources):
    """Retain checked import hashes; omit disposable host paths for portability.

    run_queue checks real paths/hashes before this projection. Reverification
    launches that same fixed worker again; it never substitutes the oracle.
    """
    from .simpy_queue import run_queue
    if sources is None:
        raise ValueError('ENGINE_STUDY_PINNED_SOURCES_REQUIRED')
    raw = run_queue(parameters, sources)
    return _portable_queue(raw)


def _portable_queue(raw):
    """Project only disposable paths away, after the adapter verified provenance."""
    upstream = {k: v for k, v in raw['upstream'].items() if k not in ('import_files', 'interpreter')}
    upstream['import_file_sha256'] = {name: value['sha256']
                                    for name, value in raw['upstream']['import_files'].items()}
    return {'engine': 'SIMPY_PINNED_FIFO_QUEUE', 'mode': 'NO_LLM',
            'native_execution': 'EXECUTED', 'native_parity': 'EXACT_MATCH_FIFO_INTERVAL_ORACLE',
            'actual': raw['actual'], 'comparison': raw['comparison'], 'upstream': upstream,
            'prepared_file_sha256': raw['prepared_file_sha256'],
            'worker_sha256': raw['worker_sha256'], 'manifest_sha256': raw['manifest_sha256'],
            'source_files_unchanged': raw['source_files_unchanged'],
            'temporary_source_removed_on_return': raw['temporary_source_removed_on_return'],
            'child_exit': raw['child_exit'], 'os_isolation': 'NOT_ATTESTED',
            'assumptions': ['declared FIFO server count with finite total system capacity',
                            'arrivals precede completions at equal model ticks',
                            'declared arrival/service durations and waiting-cost rate',
                            'unverified researcher data; no causal or forecast inference'],
            'provenance_scope': 'Import paths verified by adapter; portable receipt retains relative file hashes.',
            'limits': raw['limits']}


def _chart(spec, result):
    if spec['family'] == 'vintage_comparison':
        return {'schema': 'societybox-engine-study-chart-v1', 'labels': [], 'values_exact': [],
                'unit': 'SERIES_SPECIFIC_UNITS_USE_REVISION_TABLE', 'source_id': spec['source']['id'],
                'quantity': 'DECLARED_REVISION_DIFFERENCES_NOT_A_COMMON_AXIS'}
    if spec['family'] in ('input_output', 'survey_replicate'):
        io_family = spec['family'] == 'input_output'
        return {'schema': 'societybox-engine-study-chart-v1',
                'labels': result['sector_names'] if io_family else [],
                'values_exact': result['gross_output_change_exact'] if io_family else [],
                'unit': result['unit'], 'source_id': spec['source']['id'],
                'quantity': ('conditional gross output change; direct plus indirect, no household closure'
                             if io_family else 'NO_COMMON_AXIS_MEAN_AND_VARIANCE_HAVE_DIFFERENT_UNITS')}
    if spec['family'] in ('regression_sensitivity','fixed_effects_regression'):
        # Different coefficient/covariance units must not share a quantitative
        # chart axis. Exact coefficients and all covariance entries are exported.
        return {'schema': 'societybox-engine-study-chart-v1', 'labels': [], 'values_exact': [],
                'unit': 'NO_COMMON_UNIT_COEFFICIENTS_IN_CSV', 'source_id': spec['source']['id'],
                'note': 'MIXED_PARAMETER_UNITS_USE_COEFFICIENT_AND_COVARIANCE_TABLES'}
    if spec['family'] == 'forecast_validation':
        data = spec['request']['parameters']['screen']['study']
        scored = [row for row in result['horizon_summaries'] if row['scored_pairs']]
        return {'schema': 'societybox-engine-study-chart-v1',
                'labels': [f"h{row['horizon']}_{model}" for row in scored for model in ('model_mse', 'last_available_target_mse')],
                'values_exact': [row[key] for row in scored for key in ('model_mse_exact', 'baseline_mse_exact')],
                'unit': data['series_units'][data['target']] + ' squared',
                'quantity': 'paired retrospective origin-wise MSE by horizon; not calibrated future skill',
                'unscored_horizons': [row['horizon'] for row in result['horizon_summaries'] if not row['scored_pairs']],
                'source_id': spec['source']['id']}
    if spec['family'] == 'supply_cost':
        feasible = result['actual']['status'] == 'OPTIMAL'
        return {'schema': 'societybox-engine-study-chart-v1',
                'labels': ['minimum_declared_total_cost'] if feasible else [],
                'values_exact': [str(result['actual']['cost'])] if feasible else [],
                'status': result['actual']['status'], 'unit': spec['request']['parameters']['cost_unit'],
                'quantity': 'declared one-commodity static minimum flow/circulation cost; not observed profit or welfare',
                'source_id': spec['source']['id']}
    if spec['family'] == 'coefficient_identification':
        nonunique = sum(Fraction(row['lower_exact']) != Fraction(row['upper_exact']) for row in result['ranges'])
        return {'schema': 'societybox-engine-study-chart-v1',
                'labels': ['design_rank', 'parameter_count', 'nullity', 'nonunique_coefficients'],
                'values_exact': list(map(str, (result['design_rank'], result['parameter_count'], result['nullity'], nonunique))),
                'unit': 'dimensionless finite-design counts',
                'quantity': 'rank/ambiguity diagnostics; not coefficient magnitudes or structural causal identification',
                'source_id': spec['source']['id']}
    if spec['family'] == 'dynamic_inverse':
        holdout = result['screen']['holdout']
        data = spec['request']['parameters']['screen']['study']
        return {'schema': 'societybox-engine-study-chart-v1',
                'labels': [] if holdout is None else ['selected_model_holdout_mse', 'previous_target_holdout_mse'],
                'values_exact': [] if holdout is None else [holdout['model_mse_exact'], holdout['naive_previous_target_mse_exact']],
                'unit': data['series_units'][data['target']] + ' squared',
                'quantity': 'same-target retrospective holdout losses; no authenticated historical-as-of or future forecast claim',
                'source_id': spec['source']['id']}
    if spec['family'] == 'mathematical_program':
        values = result['actual']['solution'] or []
        return {'schema': 'societybox-engine-study-chart-v1',
                'labels': ['x'+str(i) for i in range(len(values))],
                'values_decimal': [str(v) for v in values],
                'numeric_representation': 'NATIVE_BINARY64_ROUNDTRIP_DECIMAL_NOT_EXACT',
                'unit': 'researcher-declared decision units; unverified',
                'source_id': spec['source']['id']}
    if spec['family'] == 'supply_capacity':
        return {'schema': 'societybox-engine-study-chart-v1', 'labels': ['maximum_throughput'],
                'values_exact': [str(result['actual']['flow_value'])],
                'unit': 'declared common goods units/day',
                'quantity': 'maximum steady throughput, not observed output or economic loss',
                'source_id': spec['source']['id']}
    if spec['family'] == 'service_staffing':
        return {'schema': 'societybox-engine-study-chart-v1',
                'labels': ['servers_'+str(p['server_count']) for p in result['policies']],
                'values_exact': [str(p['total_cost']) for p in result['policies']],
                'unit': 'declared common cost units for entire workload',
                'quantity': 'waiting + rejection + staffing cost; supplied alternatives only',
                'source_id': spec['source']['id']}
    if spec['family'] == 'supply_network':
        if 'disrupted_node' in spec['request']['parameters']:
            actual = result['actual']
            return {'schema': 'societybox-engine-study-chart-v1',
                    'labels': ['newly_unreachable', 'previously_unreachable'],
                    'values_exact': [str(actual['newly_unreachable_demand_units_per_day']),
                                     str(actual['previously_unreachable_demand_units_per_day'])],
                    'unit': 'declared common demand units/day',
                    'quantity': 'supply access at risk; not realized economic loss',
                    'source_id': spec['source']['id']}
        reachable = {node: delay for node, delay in result['actual'].items() if delay is not None}
        return {'schema': 'societybox-engine-study-chart-v1',
                'labels': list(reachable), 'values_exact': [str(v) for v in reachable.values()],
                'unit': 'model ticks', 'quantity': 'minimum directed delivery delay',
                'unreachable_nodes': [n for n, v in result['actual'].items() if v is None],
                'source_id': spec['source']['id']}
    if spec['family'] == 'consumption':
        labels, values = ['c0', 'c1'], result['consumption_exact']
    elif spec['family'] == 'service_queue':
        jobs = result['actual']['jobs']
        return {'schema': 'societybox-engine-study-chart-v1',
                'labels': ['job_'+str(job['id']) for job in jobs],
                'values_exact': [str(job['wait']) for job in jobs],
                'unit': 'model ticks', 'quantity': 'waiting time for admitted jobs only',
                'source_id': spec['source']['id']}
    else:
        values = result['solution_exact'] or []
        labels = ['x'+str(i) for i in range(len(values))]
    return {'schema': 'societybox-engine-study-chart-v1', 'labels': labels,
            'values_exact': values, 'unit': 'researcher model units; unverified',
            'source_id': spec['source']['id']}


def chart_svg(chart):
    """Exact values in metadata/text, relative bar widths only for display."""
    from html import escape
    strings = chart.get('values_exact', chart.get('values_decimal', []))
    values = [Fraction(x) for x in strings]
    if not values:
        return ''
    maximum = max(abs(v) for v in values) or Fraction(1)
    signed = any(v < 0 for v in values)
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Model values" '
             'viewBox="0 0 640 ' + str(40*len(values)) + '">',
             '<metadata>'+escape(_bytes(chart).decode())+'</metadata>']
    for i, (label, exact, number) in enumerate(zip(chart['labels'], strings, values)):
        width = int(abs(number)/maximum*(180 if signed else 300))
        start = (280-width if number < 0 else 280) if signed else 70
        direction = 'negative' if number < 0 else 'positive' if number > 0 else 'zero'
        parts.append(f'<rect x="{start}" y="{i*40+5}" width="{width}" height="20" '
                     f'data-sign="{direction}" fill="#234d3c"/>'
                     f'<text x="0" y="{i*40+20}">{escape(label)}</text>'
                     f'<text x="500" y="{i*40+20}">{escape(exact)}</text>')
    return ''.join(parts)+'</svg>'


def _artifacts(spec, checked, sources=None, runtime=None):
    from .research_review_entry import render_engine_study_review
    result = _calculate(spec, sources, runtime)
    checkpoint()
    chart = _chart(spec, result)
    receipt = {'schema': 'societybox-engine-study-receipt-v1', 'review': checked,
               'operator_acknowledgment': {'review_sha256': checked['review_sha256'],
                                           'authentication': 'NOT_ATTESTED'},
               'result': result, 'chart': chart,
               'requested_upstream': spec['request']['engine'],
               'actual_engine': result['engine'], 'native_execution': result['native_execution'],
               'native_parity': result.get('native_parity', 'NOT_RUN'),
               'source_clearance': 'OPERATOR_DECLARATION_NOT_VERIFIED',
               'interpretation': 'Conditional on declared data and model; causal identification, '
                                 'forecast validity and scientific acceptance NOT_ESTABLISHED.'}
    stream = io.StringIO(newline='')
    writer = csv.writer(stream)
    writer.writerow(['variable', 'value_exact' if 'values_exact' in chart else 'value_decimal', 'unit', 'source_id'])
    def csv_text(value):
        text = str(value)
        return "'" + text if text.lstrip().startswith(('=', '+', '-', '@')) or any(c in text for c in '\t\r\n') else text
    writer.writerows(zip(map(csv_text, chart['labels']), chart.get('values_exact', chart.get('values_decimal', [])),
                        [csv_text(chart['unit'])]*len(chart['labels']),
                        [csv_text(chart['source_id'])]*len(chart['labels'])))
    for node in chart.get('unreachable_nodes', []):
        writer.writerow([node, 'UNREACHABLE', chart['unit'], chart['source_id']])
    files = {'study.json': _bytes(spec), 'review.json': _bytes(checked),
             'receipt.json': _bytes(receipt), 'chart.svg': chart_svg(chart).encode(),
             'values.csv': stream.getvalue().encode('utf-8')}
    if spec['family'] in ('input_output', 'survey_replicate', 'vintage_comparison'):
        files['evidence.json'] = _bytes({key: result[key] for key in ('evidence', 'unknowns', 'obligations', 'coverage')})
        stream = io.StringIO(newline=''); writer = csv.writer(stream)
        if spec['family'] == 'vintage_comparison':
            fields = ['series_id', 'period_start', 'period_end', 'status', 'earlier_value_decimal',
                      'later_value_decimal', 'delta_exact', 'unit', 'earlier_revision_id', 'later_revision_id']
            writer.writerow(fields)
            for row in result['changes']:
                writer.writerow([csv_text(str(row[field])) if row[field] is not None else '' for field in fields])
            files['revisions.csv'] = stream.getvalue().encode('utf-8')
        elif spec['family'] == 'input_output':
            writer.writerow(['sector', 'direct_exact', 'indirect_exact', 'gross_exact', 'unit'])
            writer.writerows(zip(map(csv_text, result['sector_names']), result['final_demand_direct_exact'],
                                 result['indirect_output_change_exact'], result['gross_output_change_exact'],
                                 [csv_text(result['unit'])]*len(result['sector_names'])))
            files['effects.csv'] = stream.getvalue().encode('utf-8')
        else:
            writer.writerow(['quantity', 'value_exact', 'unit'])
            writer.writerow(['mean', result['point_exact'], csv_text(result['unit'])])
            writer.writerow(['variance', result['variance_exact'], csv_text(result['unit'] + ' squared')])
            for i, value in enumerate(result['replicate_points_exact']):
                writer.writerow(['replicate_' + str(i+1), value, csv_text(result['unit'])])
            files['replicates.csv'] = stream.getvalue().encode('utf-8')
    if spec['family'] == 'dynamic_inverse':
        from .inverse_native import export_artifacts
        files.update(export_artifacts(spec['request']['parameters'], result, chart))
    if spec['family'] == 'forecast_validation':
        from .forecast_validation import export_artifacts
        files.update(export_artifacts(spec['request']['parameters'], result, chart))
    if spec['family'] == 'regression_sensitivity':
        from .regression_sensitivity import export_artifacts
        files.update(export_artifacts(spec['request']['parameters'], result, chart))
    if spec['family'] == 'fixed_effects_regression':
        from .fixed_effects import export_artifacts
        files.update(export_artifacts(spec['request']['parameters'],result,chart))
    if spec['family'] == 'supply_cost':
        from .network_cost_flow import export_artifacts
        files.update(export_artifacts(spec['request']['parameters'], result, chart))
    if spec['family'] == 'coefficient_identification':
        from .inverse_identification import export_artifacts
        files.update(export_artifacts(spec['request']['parameters'], result, chart))
    if spec['family'] == 'mathematical_program':
        parameters = spec['request']['parameters']; actual = result['actual']
        stream = io.StringIO(newline=''); writer = csv.writer(stream)
        writer.writerow(['row', 'lower', 'upper', 'native_value_decimal'])
        for i, (lower, upper) in enumerate(zip(parameters['row_lower'], parameters['row_upper'])):
            writer.writerow([i, lower, upper, '' if actual['row_values'] is None else actual['row_values'][i]])
        files['constraints.csv'] = stream.getvalue().encode('utf-8')
        if spec['request']['engine'] == 'Pyomo+HiGHS':
            derivatives = result['upstream']['modeling']['derivatives']
            stream = io.StringIO(newline=''); writer = csv.writer(stream)
            writer.writerow(['quantity', 'row', 'column', 'value_decimal', 'interpretation'])
            for quantity in ('gradient_constant', 'gradient_linear', 'hessian', 'constraint_jacobian'):
                for i, row in enumerate(derivatives[quantity]):
                    if quantity == 'gradient_constant':
                        writer.writerow([quantity, i, '', row, 'continuous polynomial extension; no feasible integer marginal implied'])
                    else:
                        for j, value in enumerate(row):
                            writer.writerow([quantity, i, j, value, 'continuous polynomial extension; declared model only'])
            files['derivatives.csv'] = stream.getvalue().encode('utf-8')
    if spec['family'] == 'supply_capacity':
        stream = io.StringIO(newline=''); writer = csv.writer(stream)
        writer.writerow(['supplier', 'customer', 'capacity', 'flow', 'unit'])
        for (a, b, capacity), (_, _, flow) in zip(spec['request']['parameters']['links'],
                                                  result['actual']['edge_flows']):
            writer.writerow([a, b, capacity, flow, chart['unit']])
        files['flows.csv'] = stream.getvalue().encode('utf-8')
    if spec['family'] == 'service_staffing':
        policies = result['policies']
        jobs = [dict(server_count=p['server_count'], **job)
                for p, r in zip(policies, result['runs']) for job in r['actual']['jobs']]
        for name, rows, fields in (
                ('policies.csv', policies, list(policies[0])),
                ('jobs.csv', jobs, ['server_count', 'id', 'arrival', 'start', 'end', 'wait'])):
            stream = io.StringIO(newline='')
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else fields)
            writer.writeheader(); writer.writerows(rows)
            files[name] = stream.getvalue().encode('utf-8')
    for locale in LOCALES:
        files['review-'+locale+'.html'] = render_engine_study_review(spec, receipt, locale).encode('utf-8')
    if any(len(raw) > ARTIFACT_LIMIT for raw in files.values()):
        raise ValueError('ENGINE_STUDY_ARTIFACT_BUDGET')
    return receipt, files


def run(spec, output, acknowledgment, *, sources=None, runtime=None):
    checked = review(spec, runtime=runtime)
    if type(acknowledgment) is not str or acknowledgment != checked['review_sha256']:
        raise ValueError('ENGINE_STUDY_ACKNOWLEDGMENT_REQUIRED')
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError('ENGINE_STUDY_OUTPUT_EXISTS')
    receipt, files = _artifacts(spec, checked, sources, runtime)
    if (_code_hash() != checked['binding']['engine_code_sha256'] or
            _sha(_bytes(spec)) != checked['binding']['study_sha256']):
        raise ValueError('ENGINE_STUDY_CHANGED_DURING_RUN')
    # Stage siblings and rename only after every artifact succeeds. Never merge
    # with existing research output or touch caller inputs.
    staging = Path(tempfile.mkdtemp(prefix='.engine-study-', dir=output.parent))
    try:
        for name, raw in files.items():
            (staging / name).write_bytes(raw)
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return receipt


def verify(bundle, *, sources=None, runtime=None):
    bundle = Path(bundle)
    if bundle.is_symlink() or not bundle.is_dir():
        raise ValueError('ENGINE_STUDY_BUNDLE_INVALID')
    spec_raw = _read(bundle / 'study.json')
    spec = _parse(spec_raw)
    checked = review(spec, runtime=runtime)
    # A modified input/code binding requires fresh operator review before any
    # pinned upstream reexecution. Existing receipts cannot authorize new inputs.
    if spec_raw != _bytes(spec) or _read(bundle / 'review.json') != _bytes(checked):
        raise ValueError('ENGINE_STUDY_REVIEW_CHANGED')
    # The artifact layout depends only on the reviewed study family. Check it
    # before invoking costly native workers; successful verification still reruns
    # the calculation and compares every artifact against fresh bytes below.
    names = {'study.json', 'review.json', 'receipt.json', 'chart.svg', 'values.csv'}
    names.update('review-' + locale + '.html' for locale in LOCALES)
    if spec['family'] in ('input_output', 'survey_replicate', 'vintage_comparison'):
        names.update(('evidence.json', {'input_output': 'effects.csv', 'survey_replicate': 'replicates.csv',
                                      'vintage_comparison': 'revisions.csv'}[spec['family']]))
    if spec['family'] == 'dynamic_inverse':
        names.update(('coefficients.csv', 'candidates.csv', 'holdout.csv', 'investigation.json'))
    if spec['family'] == 'coefficient_identification':
        names.update(('ranges.csv', 'witnesses.csv', 'separating-observations.csv', 'identification.json'))
    if spec['family'] == 'supply_cost':
        names.update(('flows.csv', 'node-balances.csv', 'certificate.csv', 'evidence.json'))
    if spec['family'] == 'forecast_validation':
        names.update(('forecasts.csv', 'forecast-candidates.csv', 'forecast-summary.csv', 'forecast-evidence.json'))
        if 'preprocessing' in spec['request']['parameters']:
            names.add('forecast-preprocessing.csv')
    if spec['family'] == 'regression_sensitivity':
        names.update(('coefficients.csv', 'covariances.csv', 'regression-residuals.csv', 'regression-evidence.json'))
    if spec['family'] == 'fixed_effects_regression':
        names.update(('coefficients.csv','covariances.csv','fixed-effects-residuals.csv','fixed-effects-evidence.json'))
    if spec['family'] == 'mathematical_program':
        names.add('constraints.csv')
        if spec['request']['engine'] == 'Pyomo+HiGHS':
            names.add('derivatives.csv')
    elif spec['family'] == 'supply_capacity':
        names.add('flows.csv')
    elif spec['family'] == 'service_staffing':
        names.update(('policies.csv', 'jobs.csv'))
    if {p.name for p in bundle.iterdir()} != names:
        raise ValueError('ENGINE_STUDY_ARTIFACT_SET_CHANGED')
    receipt, expected = _artifacts(spec, checked, sources, runtime)
    if (_code_hash() != checked['binding']['engine_code_sha256'] or
            _sha(_bytes(spec)) != checked['binding']['study_sha256']):
        raise ValueError('ENGINE_STUDY_CHANGED_DURING_VERIFY')
    if {p.name for p in bundle.iterdir()} != set(expected):
        raise ValueError('ENGINE_STUDY_ARTIFACT_SET_CHANGED')
    for name, raw in expected.items():
        if _read(bundle / name, ARTIFACT_LIMIT) != raw:
            raise ValueError('ENGINE_STUDY_RECOMPUTATION_MISMATCH')
    return {'status': 'RECOMPUTED', 'family': spec['family'],
            'review_sha256': checked['review_sha256'],
            'native_execution': receipt['native_execution'], 'native_parity': receipt['native_parity'],
            'replication_scope': 'SAME_SOURCE_TREE_DETERMINISTIC_RECOMPUTATION',
            'authenticity': 'NOT_ATTESTED', 'scientific_acceptance': 'NOT_ESTABLISHED',
            'artifacts': {name: _sha(raw) for name, raw in expected.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ('review', 'run'):
        p = commands.add_parser(command)
        p.add_argument('--study', required=True)
        p.add_argument('--runtime', help='Reviewed hash-pinned local HiGHS runtime for mathematical programs or selected inverse regression')
        if command == 'run':
            p.add_argument('--output', required=True)
            p.add_argument('--acknowledge-review', required=True)
            p.add_argument('--sources', help='Reviewed pinned archive directory; required for upstream studies')
    p = commands.add_parser('verify'); p.add_argument('--bundle', required=True)
    p.add_argument('--runtime', help='Reviewed hash-pinned local HiGHS runtime')
    p.add_argument('--sources', help='Reviewed pinned archive directory; required for upstream studies')
    args = parser.parse_args()
    try:
        if args.command == 'verify':
            result = verify(args.bundle, sources=args.sources, runtime=args.runtime)
        else:
            spec = _parse(_read(args.study))
            result = review(spec, runtime=args.runtime) if args.command == 'review' else run(
                spec, args.output, args.acknowledge_review, sources=args.sources, runtime=args.runtime)
        sys.stdout.buffer.write(_bytes(result) + b'\n')
        return 0
    except (ValueError, OSError, RuntimeError, RecursionError, UnicodeError) as error:
        # Only fixed, reviewed codes may become public blockers. Never expose
        # arbitrary exception text, private paths, data values or questions.
        if type(error) is ValueError and str(error) in {
                'INVERSE_NATIVE_BOUNDS_EXCLUDE_OLS', 'INVERSE_NATIVE_REGRESSION_PARITY_FAILED',
                'INVERSE_NATIVE_CODE_CHANGED_DURING_RUN', 'HIGHS_NUMBER_LIMIT',
                'IDENTIFICATION_NATIVE_DIMENSION_BLOCKED', 'IDENTIFICATION_REFERENCE_OUTSIDE_BOUNDS',
                'IDENTIFICATION_FULL_DESIGN_NATIVE_PARITY_FAILED', 'IDENTIFICATION_ENDPOINT_OBJECTIVE_MISMATCH',
                'IDENTIFICATION_CODE_CHANGED_DURING_RUN', 'IDENTIFICATION_INPUT_CHANGED_DURING_RUN',
                'FORECAST_CODE_CHANGED_DURING_RUN', 'FORECAST_INPUT_CHANGED_DURING_RUN',
                'REGRESSION_CODE_CHANGED_DURING_RUN', 'REGRESSION_INPUT_CHANGED_DURING_RUN',
                'REGRESSION_NATIVE_DIMENSION_BLOCKED', 'REGRESSION_BANDWIDTH_INVALID',
                'FORECAST_GRID_BUDGET', 'FORECAST_RUN_TIME_BUDGET', 'FORECAST_DATE_RANGE_INVALID'}:
            print(json.dumps({'status': 'BLOCKED', 'reason': str(error),
                              'scope': 'NO_NEW_VERIFIED_RESULT_FOR_THIS_ATTEMPT'}))
            return 2
        # Do not echo private paths, source values or questions to diagnostics.
        print(json.dumps({'status': 'INVALID', 'reason': 'ENGINE_STUDY_VALIDATION_FAILED'}))
        return 2


if __name__ == '__main__':
    sys.exit(main())
