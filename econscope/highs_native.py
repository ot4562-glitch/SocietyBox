"""Bounded original HiGHS LP/MILP/strictly convex QP, independent exact oracles.

Finite signed variable bounds and two-sided rows; no generated model code, paths
or solver options in model data. This scope does not exhaust the HiGHS interface.
"""
from fractions import Fraction
import hashlib
from itertools import combinations, product
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from .optimization_policy import _dot, _solve_square
from .strict_json import loads_strict
from .execution_control import checkpoint, remaining_timeout, safe_local_path, run_worker

COMMIT = '73cac48c5340d775a477087198611862559be250'
ARCHIVE_SHA256 = '828f43c93035ebf8b62e45f5a5033e6ffe2d537f05dc4daf318a506555528289'
VERSION = '1.15.1'
ABS_TOL, REL_TOL = 1e-7, 1e-8
NATIVE_OPTIONS = {'threads': 1, 'mip_max_nodes': 10000, 'simplex_iteration_limit': 100000,
                  'ipm_iteration_limit': 100000, 'time_limit': 8.0,
                  'mip_rel_gap': 0.0, 'mip_abs_gap': 0.0,
                  'primal_feasibility_tolerance': 1e-9, 'dual_feasibility_tolerance': 1e-9,
                  'parallel': 'off'}
BASE_APIS = ['Highs_create', 'Highs_getSizeofHighsInt', 'Highs_setBoolOptionValue',
             'Highs_setIntOptionValue', 'Highs_setDoubleOptionValue', 'Highs_setStringOptionValue',
             'Highs_passModel', 'Highs_run', 'Highs_getModelStatus', 'Highs_destroy', 'Highs_version']
# Set only from an observed local build and separately inspected dependency hashes.
RUNTIME_PIN = {'library': 'runtime/libhighs.dll', 'files': {
    'runtime/libhighs.dll': '2864e7dae9720b4efadee98d3dffa783a5f3570a8949cd3dcd71a17ee3e7f52a',
    'runtime/libgcc_s_seh-1.dll': '4dfa2c0d57e3dcd72b2a90cf6dfbffb5b6c985fc873bcc73ab60eff71cbc6b00',
    'runtime/libstdc++-6.dll': 'a36c2e6871cc90d1d336c0fcb616d84cdaa817d64ac1f883a002850a411bae03',
    'runtime/libwinpthread-1.dll': 'c29cbbe3a9c710df0dfecda4dbc1664b95658278511b961cbd2357c2a40d0494',
    'build-receipt.json': '905c36cf2627d409a46ab6087abfd7b4198187874db9dff426a664c602a1e151',
    'runtime-receipt.json': 'db3569b35ac3ea217f06ae117b0bad141691936778325e65196384dfb1cd2813'}}


def _bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _sha(raw): return hashlib.sha256(raw).hexdigest()


def _num(value):
    if type(value) not in (int, float, str): raise ValueError('HIGHS_FINITE_NUMBER_REQUIRED')
    if type(value) is str and (len(value) > 40 or not re.fullmatch(r'-?\d+(?:\.\d+|/\d+)?', value)):
        raise ValueError('HIGHS_NUMBER_FORMAT_INVALID')
    if type(value) is float and not math.isfinite(value): raise ValueError('HIGHS_FINITE_NUMBER_REQUIRED')
    if type(value) is int and value.bit_length() > 64: raise ValueError('HIGHS_NUMBER_LIMIT')
    try: number = Fraction(str(value))
    except (ValueError, ZeroDivisionError) as error: raise ValueError('HIGHS_NUMBER_INVALID') from error
    if abs(number) > 1000000 or max(number.numerator.bit_length(), number.denominator.bit_length()) > 64:
        raise ValueError('HIGHS_NUMBER_LIMIT')
    return number


def _model(spec):
    keys = {'kind', 'sense', 'objective', 'constraints', 'row_lower', 'row_upper',
            'lower', 'upper', 'integrality', 'hessian', 'offset'}
    if type(spec) is not dict or set(spec) != keys or spec['kind'] not in ('LP', 'MILP', 'QP'):
        raise ValueError('HIGHS_MODEL_SCHEMA_INVALID')
    if spec['sense'] not in ('minimize', 'maximize'): raise ValueError('HIGHS_SENSE_INVALID')
    c, a = spec['objective'], spec['constraints']
    if (type(c) is not list or not 1 <= len(c) <= 4 or type(a) is not list or len(a) > 8 or
            any(type(row) is not list or len(row) != len(c) for row in a)):
        raise ValueError('HIGHS_DIMENSIONS_INVALID')
    n, m = len(c), len(a)
    for key, length in (('lower', n), ('upper', n), ('integrality', n), ('row_lower', m), ('row_upper', m)):
        if type(spec[key]) is not list or len(spec[key]) != length: raise ValueError('HIGHS_DIMENSIONS_INVALID')
    if any(type(v) is not int or v not in (0, 1) for v in spec['integrality']):
        raise ValueError('HIGHS_INTEGRALITY_INVALID')
    if spec['kind'] != 'MILP' and any(spec['integrality']): raise ValueError('HIGHS_INTEGRALITY_KIND_MISMATCH')
    if spec['kind'] == 'MILP' and not any(spec['integrality']): raise ValueError('HIGHS_INTEGER_VARIABLE_REQUIRED')
    model = {key: list(map(_num, spec[key])) for key in ('objective', 'lower', 'upper', 'row_lower', 'row_upper')}
    model.update(kind=spec['kind'], sense=spec['sense'], integrality=spec['integrality'],
                 offset=_num(spec['offset']), constraints=[list(map(_num, row)) for row in a])
    if any(l > u for l, u in zip(model['lower']+model['row_lower'], model['upper']+model['row_upper'])):
        raise ValueError('HIGHS_BOUND_ORDER_INVALID')
    q = spec['hessian']
    if spec['kind'] != 'QP':
        if type(q) is not list or q: raise ValueError('HIGHS_HESSIAN_KIND_MISMATCH')
        model['hessian'] = [[Fraction(0)]*n for _ in range(n)]
    else:
        if type(q) is not list or len(q) != n or any(type(row) is not list or len(row) != n for row in q):
            raise ValueError('HIGHS_HESSIAN_DIMENSIONS_INVALID')
        q = [list(map(_num, row)) for row in q]
        if any(q[i][j] != q[j][i] for i in range(n) for j in range(n)):
            raise ValueError('HIGHS_HESSIAN_NOT_SYMMETRIC')
        # Exact LDL pivots prove strict convexity of the minimization form.
        sign = 1 if spec['sense'] == 'minimize' else -1
        lower = [[Fraction(int(i == j)) for j in range(n)] for i in range(n)]; diagonal = []
        for i in range(n):
            diagonal.append(sign*q[i][i]-sum(lower[i][k]**2*diagonal[k] for k in range(i)))
            if diagonal[-1] <= 0: raise ValueError('HIGHS_QP_STRICT_CONVEXITY_REQUIRED')
            for j in range(i+1, n):
                lower[j][i] = (sign*q[j][i]-sum(lower[j][k]*lower[i][k]*diagonal[k] for k in range(i)))/diagonal[i]
        model['hessian'] = q
    assignments = 1
    for i, integer in enumerate(spec['integrality']):
        if integer: assignments *= max(0, math.floor(model['upper'][i])-math.ceil(model['lower'][i])+1)
    inequalities = 2*(n+m)
    work = (sum(math.comb(inequalities, k) for k in range(n+1)) if spec['kind'] == 'QP'
            else math.comb(inequalities, n)*assignments)
    if assignments > 256 or work > 20000: raise ValueError('HIGHS_ORACLE_WORK_BUDGET')
    return model


def validate(spec):
    _model(spec)
    return spec


def _inequalities(model, assignment):
    n = len(model['objective']); a, b = [], []
    for row, lo, hi in zip(model['constraints'], model['row_lower'], model['row_upper']):
        a += [row, [-v for v in row]]; b += [hi, -lo]
    for i, (lo, hi) in enumerate(zip(model['lower'], model['upper'])):
        if i in assignment: lo = hi = Fraction(assignment[i])
        row = [Fraction(int(i == j)) for j in range(n)]
        a += [row, [-v for v in row]]; b += [hi, -lo]
    return a, b


def oracle(spec):
    model = _model(spec); n = len(model['objective']); sign = 1 if model['sense'] == 'minimize' else -1
    indices = [i for i, v in enumerate(model['integrality']) if v]
    ranges = [range(math.ceil(model['lower'][i]), math.floor(model['upper'][i])+1) for i in indices]
    best, best_x, systems, assignments = None, None, 0, 0
    started = time.monotonic()
    for values in product(*ranges):
        assignments += 1
        a, b = _inequalities(model, dict(zip(indices, values)))
        counts = range(n+1) if model['kind'] == 'QP' else (n,)
        for k in counts:
            for active in combinations(range(len(b)), k):
                checkpoint()
                systems += 1
                if systems % 64 == 0 and time.monotonic()-started > 15:
                    raise RuntimeError('HIGHS_ORACLE_TIME_BUDGET')
                if model['kind'] == 'QP':
                    h = model['hessian']; c = model['objective']
                    matrix = [[sign*h[i][j] for j in range(n)]+[a[z][i] for z in active] for i in range(n)]
                    matrix += [a[z]+[Fraction(0)]*k for z in active]
                    solution = _solve_square(matrix, [-sign*v for v in c]+[b[z] for z in active])
                    if solution is None or any(v < 0 for v in solution[n:]): continue
                    x = solution[:n]
                else:
                    x = _solve_square([a[z] for z in active], [b[z] for z in active])
                if x is None or any(_dot(row, x) > bound for row, bound in zip(a, b)): continue
                objective = model['offset']+_dot(model['objective'], x)+sum(
                    x[i]*model['hessian'][i][j]*x[j]/2 for i in range(n) for j in range(n))
                if best is None or sign*objective < sign*best or (objective == best and x < best_x):
                    best, best_x = objective, x
    return {'status': 'INFEASIBLE' if best is None else 'OPTIMAL',
            'solution_exact': None if best_x is None else list(map(str, best_x)),
            'objective_exact': None if best is None else str(best), 'systems_examined': systems,
            'integer_assignments_examined': assignments,
            'method': ('EXACT_ACTIVE_SET_KKT' if model['kind'] == 'QP' else
                       'EXACT_FINITE_INTEGER_AND_VERTEX_ENUMERATION' if indices else 'EXACT_VERTEX_ENUMERATION'),
            'limits': 'Finite bounds; strictly convex/concave QP only; 20,000 systems; 15 seconds.'}


def compare(spec, actual):
    return _compare(spec, actual, oracle(spec))


def _compare(spec, actual, expected):
    model = _model(spec)
    if (type(actual) is not dict or set(actual) != {'status', 'solution', 'objective', 'row_values'} or
            actual.get('status') != expected['status']):
        raise ValueError('HIGHS_ORACLE_STATUS_MISMATCH')
    if expected['status'] == 'INFEASIBLE':
        if any(actual.get(k) is not None for k in ('solution', 'objective', 'row_values')):
            raise ValueError('HIGHS_INFEASIBLE_RESULT_HAS_VALUES')
        return {'status': 'WITHIN_DECLARED_TOLERANCE', 'oracle': expected,
                'absolute_tolerance': ABS_TOL, 'relative_tolerance': REL_TOL}
    x, values, objective = (actual.get(k) for k in ('solution', 'row_values', 'objective'))
    n, m = len(model['objective']), len(model['constraints'])
    def numeric(value): return type(value) in (int, float) and math.isfinite(value)
    if (type(x) is not list or len(x) != n or type(values) is not list or len(values) != m or
            not all(numeric(v) for v in x+values+[objective])):
        raise ValueError('HIGHS_RESULT_DIMENSION_OR_NUMBER')
    close = lambda a, b: math.isclose(float(a), float(b), abs_tol=ABS_TOL, rel_tol=REL_TOL)
    for i, v in enumerate(x):
        if ((v < model['lower'][i] and not close(v, model['lower'][i])) or
                (v > model['upper'][i] and not close(v, model['upper'][i])) or
                (model['integrality'][i] and abs(v-round(v)) > ABS_TOL)):
            raise ValueError('HIGHS_PRIMAL_BOUND_OR_INTEGRALITY_MISMATCH')
    for row, lower, upper, returned in zip(model['constraints'], model['row_lower'], model['row_upper'], values):
        v = sum(float(c)*z for c, z in zip(row, x))
        if (not close(v, returned) or (v < lower and not close(v, lower)) or (v > upper and not close(v, upper))):
            raise ValueError('HIGHS_ROW_FEASIBILITY_MISMATCH')
    recalculated = float(model['offset'])+sum(float(c)*z for c, z in zip(model['objective'], x))+sum(
        x[i]*float(model['hessian'][i][j])*x[j]/2 for i in range(n) for j in range(n))
    if not close(recalculated, objective) or not close(objective, Fraction(expected['objective_exact'])):
        raise ValueError('HIGHS_OBJECTIVE_OR_ORACLE_MISMATCH')
    return {'status': 'WITHIN_DECLARED_TOLERANCE', 'oracle': expected,
            'absolute_tolerance': ABS_TOL, 'relative_tolerance': REL_TOL,
            'integrality_absolute_tolerance': ABS_TOL,
            'objective_absolute_error': abs(objective-float(Fraction(expected['objective_exact']))),
            'nonunique_solution_policy': 'COMPARE_OBJECTIVE_AND_FEASIBILITY_NOT_VECTOR_EQUALITY'}


def runtime_pin(runtime):
    if RUNTIME_PIN is None: raise ValueError('HIGHS_RUNTIME_NOT_YET_REVIEW_PINNED')
    root = safe_local_path(runtime)
    if root.is_symlink() or not root.is_dir(): raise ValueError('HIGHS_RUNTIME_DIRECTORY_INVALID')
    folder = safe_local_path(root/'runtime')
    names = {Path(n).name for n in RUNTIME_PIN['files'] if n.startswith('runtime/')}
    if (folder.is_symlink() or not folder.is_dir() or
            {p.name for p in folder.iterdir()} != names):
        raise ValueError('HIGHS_RUNTIME_DEPENDENCY_SET_MISMATCH')
    for name, wanted in RUNTIME_PIN['files'].items():
        path = safe_local_path(root/name)
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 100_000_000 or _sha(path.read_bytes()) != wanted:
            raise ValueError('HIGHS_RUNTIME_PIN_MISMATCH')
    return {'commit': COMMIT, 'archive_sha256': ARCHIVE_SHA256, 'version': VERSION,
            'files_sha256': RUNTIME_PIN['files'], 'redistribution': 'NOT_AUTHORIZED'}


def _checked_worker(spec, runtime, mode=None):
    if mode not in (None,'--quantile-dual'):raise ValueError('HIGHS_WORKER_MODE_INVALID')
    if runtime is None: raise ValueError('HIGHS_PINNED_RUNTIME_REQUIRED')
    pin = runtime_pin(runtime)
    worker = Path(__file__).with_name('highs_native_worker.py')
    try:
        child = run_worker([sys.executable, '-I', '-S', '-B', str(worker), str(Path(runtime).resolve())]+([] if mode is None else [mode]),
            input=_bytes(spec), capture_output=True, timeout=remaining_timeout(20),
            env={'SYSTEMROOT': os.environ.get('SYSTEMROOT', '')},
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    except subprocess.TimeoutExpired as error: raise RuntimeError('HIGHS_WORKER_TIMEOUT_REAPED') from error
    if child.returncode or len(child.stdout) > 65536: raise RuntimeError('HIGHS_FIXED_WORKER_FAILED')
    raw = loads_strict(child.stdout)
    upstream_keys = {'version', 'runtime_pin', 'apis', 'model_status', 'options', 'integer_abi_bytes',
                     'elevated', 'network_audit_canary_denied', 'loaded_native_files_sha256',
                     'network_enforcement', 'native_network_isolation', 'python_version'}
    if (type(raw) is not dict or set(raw) != {'actual', 'upstream'} or
            type(raw['upstream']) is not dict or set(raw['upstream']) != upstream_keys):
        raise ValueError('HIGHS_UPSTREAM_PROVENANCE_MISMATCH')
    if type(raw['actual']) is not dict or raw['actual'].get('status') not in ('OPTIMAL','INFEASIBLE'):
        raise ValueError('HIGHS_UPSTREAM_STATUS_INVALID')
    upstream = raw['upstream']
    expected_apis = BASE_APIS+(['Highs_getSolution', 'Highs_getObjectiveValue'] if raw['actual']['status'] == 'OPTIMAL' else [])
    if (upstream['version'] != VERSION or _bytes(upstream['runtime_pin']) != _bytes(pin) or
            upstream['elevated'] is not False or upstream['network_audit_canary_denied'] is not True or
            upstream['network_enforcement'] != 'PYTHON_AUDIT_ONLY_NOT_OS_ISOLATION' or
            upstream['native_network_isolation'] != 'NOT_ATTESTED' or upstream['python_version'] != sys.version or
            type(upstream['integer_abi_bytes']) is not int or upstream['integer_abi_bytes'] != 4 or
            _bytes(upstream['options']) != _bytes(NATIVE_OPTIONS) or
            _bytes(upstream['loaded_native_files_sha256']) != _bytes({k: v for k, v in pin['files_sha256'].items() if k.startswith('runtime/')}) or
            upstream['apis'] != sorted(expected_apis) or
            type(upstream['model_status']) is not int or
            upstream['model_status'] != (7 if raw['actual']['status'] == 'OPTIMAL' else 8)):
        raise ValueError('HIGHS_UPSTREAM_PROVENANCE_MISMATCH')
    runtime_pin(runtime)
    return raw,worker


def run(spec, runtime):
    validate(spec);expected=oracle(spec)
    raw,worker=_checked_worker(spec,runtime)
    comparison=_compare(spec,raw['actual'],expected)
    return {'engine': 'HIGHS_PINNED_MATHEMATICAL_PROGRAM', 'mode': 'NO_LLM',
            'native_execution': 'EXECUTED', 'native_parity': 'WITHIN_DECLARED_ORACLE_TOLERANCE',
            'actual': raw['actual'], 'comparison': comparison, 'upstream': raw['upstream'],
            'worker_sha256': _sha(worker.read_bytes()), 'child_exit': 0,
            'os_isolation': 'NOT_ATTESTED', 'assumptions': [
                'Researcher-supplied coefficients and finite bounds are unverified.',
                'Deterministic optimization, not causal identification or a calibrated forecast.',
                'QP objective is offset+c*x+0.5*x^T*Q*x; strict curvature required.',
                'No full upstream API coverage, redistribution or clean-install acceptance.']}
