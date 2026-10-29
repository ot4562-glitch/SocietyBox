"""Fixed pinned native C API worker. Data-only input; not a general OS sandbox."""
import ctypes as ct
import json
import os
from pathlib import Path
import socket
import sys


def main():
    if os.name != 'nt': raise RuntimeError('HIGHS_REVIEWED_WINDOWS_RUNTIME_REQUIRED')
    if ct.windll.shell32.IsUserAnAdmin(): raise RuntimeError('HIGHS_NONADMIN_REQUIRED')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from econscope.highs_native import validate, runtime_pin, RUNTIME_PIN, _model, NATIVE_OPTIONS, BASE_APIS
    from econscope.strict_json import loads_strict
    raw = sys.stdin.buffer.read(65537)
    if len(raw) > 65536: raise ValueError('HIGHS_INPUT_BUDGET')
    if len(sys.argv)==3 and sys.argv[2]=='--quantile-dual':
        from econscope.quantile_regression import _dual_model
        model=_dual_model(loads_strict(raw))
        quantile=True
    elif len(sys.argv)==2:
        spec = validate(loads_strict(raw)); model = _model(spec);quantile=False
    else:raise ValueError('HIGHS_WORKER_MODE_INVALID')
    root = Path(sys.argv[1]).resolve(strict=True)
    pin = runtime_pin(root)
    dll_directory = os.add_dll_directory(str(root/'runtime'))
    lib = ct.CDLL(str(root/RUNTIME_PIN['library']))
    kernel = ct.windll.kernel32
    kernel.GetModuleHandleW.restype = ct.c_void_p
    kernel.GetModuleHandleW.argtypes = [ct.c_wchar_p]
    kernel.GetModuleFileNameW.restype = ct.c_uint
    kernel.GetModuleFileNameW.argtypes = [ct.c_void_p, ct.c_wchar_p, ct.c_uint]
    loaded = {}
    for name in RUNTIME_PIN['files']:
        if not name.startswith('runtime/'): continue
        handle = kernel.GetModuleHandleW(Path(name).name)
        buffer = ct.create_unicode_buffer(32768)
        if not handle or not kernel.GetModuleFileNameW(handle, buffer, len(buffer)):
            raise RuntimeError('HIGHS_EXPECTED_DEPENDENCY_NOT_LOADED')
        if Path(buffer.value).resolve() != (root/name).resolve():
            raise RuntimeError('HIGHS_DEPENDENCY_LOADED_FROM_WRONG_PATH')
        loaded[name] = RUNTIME_PIN['files'][name]
    # Once the one pinned DLL and its pinned runtime closure are loaded, deny
    # Python-level networking/processes/additional dynamic library loads.
    def guard(event, args):
        if event.startswith(('socket.', 'subprocess.', 'os.system', 'os.spawn', 'os.exec', 'ctypes.dlopen')):
            raise RuntimeError('HIGHS_NETWORK_OR_EXTERNAL_PROCESS_DENIED')
    sys.addaudithook(guard)
    denied = False
    try: socket.socket()
    except RuntimeError: denied = True
    if not denied: raise RuntimeError('HIGHS_NETWORK_CANARY_FAILED')
    I, D, P, C = ct.c_int, ct.c_double, ct.c_void_p, ct.c_char_p
    IP, DP = ct.POINTER(I), ct.POINTER(D)
    def bind(name, result, args):
        function = getattr(lib, name); function.restype = result; function.argtypes = args
        return function
    create = bind('Highs_create', P, [])
    destroy = bind('Highs_destroy', None, [P])
    size = bind('Highs_getSizeofHighsInt', I, [P])
    version = bind('Highs_version', C, [])
    bool_option = bind('Highs_setBoolOptionValue', I, [P, C, I])
    int_option = bind('Highs_setIntOptionValue', I, [P, C, I])
    double_option = bind('Highs_setDoubleOptionValue', I, [P, C, D])
    string_option = bind('Highs_setStringOptionValue', I, [P, C, C])
    pass_model = bind('Highs_passModel', I, [P]+[I]*7+[D]+[DP]*5+[IP, IP, DP, IP, IP, DP, IP])
    run = bind('Highs_run', I, [P])
    status = bind('Highs_getModelStatus', I, [P])
    solution = bind('Highs_getSolution', I, [P, DP, DP, DP, DP])
    objective = bind('Highs_getObjectiveValue', D, [P])
    apis = list(BASE_APIS)
    def checked(value):
        if value != 0: raise RuntimeError('HIGHS_C_API_CALL_FAILED')
    def doubles(values): return (D*max(1, len(values)))(*map(float, values))
    def integers(values): return (I*max(1, len(values)))(*values)
    instance = create()
    if not instance: raise RuntimeError('HIGHS_CREATE_FAILED')
    try:
        if size(instance) != 4: raise RuntimeError('HIGHS_INTEGER_ABI_MISMATCH')
        checked(bool_option(instance, b'output_flag', 0))
        options = NATIVE_OPTIONS
        for name, value in options.items():
            if type(value) is int: checked(int_option(instance, name.encode(), value))
            elif type(value) is str: checked(string_option(instance, name.encode(), value.encode()))
            else: checked(double_option(instance, name.encode(), value))
        n, m = len(model['objective']), len(model['constraints'])
        start, index, values = [0], [], []
        for row in model['constraints']:
            for j, v in enumerate(row):
                if v: index.append(j); values.append(v)
            start.append(len(values))
        q_start, q_index, q_values = [0], [], []
        for j in range(n):
            if model['kind']=='QP':
                for i in range(j, n):
                    v = model['hessian'][i][j]
                    if v: q_index.append(i); q_values.append(v)
            q_start.append(len(q_values))
        checked(pass_model(instance, n, m, len(values), len(q_values), 2, 1,
            1 if model['sense'] == 'minimize' else -1, float(model['offset']),
            doubles(model['objective']), doubles(model['lower']), doubles(model['upper']),
            doubles(model['row_lower']), doubles(model['row_upper']), integers(start), integers(index),
            doubles(values), integers(q_start), integers(q_index), doubles(q_values), integers(model['integrality'])))
        checked(run(instance))
        model_status = status(instance)
        if model_status == 7:
            x, col_dual, rows, row_dual = doubles([0]*n), doubles([0]*n), doubles([0]*m), doubles([0]*m)
            checked(solution(instance, x, col_dual, rows, row_dual))
            actual = {'status': 'OPTIMAL', 'solution': list(x)[:n], 'objective': objective(instance),
                      'row_values': list(rows)[:m]}
            if quantile:actual.update(column_duals=list(col_dual)[:n],row_duals=list(row_dual)[:m])
            apis += ['Highs_getSolution', 'Highs_getObjectiveValue']
        else:
            actual = {'status': 'INFEASIBLE' if model_status == 8 else 'INDETERMINATE',
                      'solution': None, 'objective': None, 'row_values': None}
        raw_version = version().decode('ascii')
    finally:
        destroy(instance)
    runtime_pin(root)
    print(json.dumps({'actual': actual, 'upstream': {'version': raw_version, 'runtime_pin': pin,
        'apis': sorted(apis), 'model_status': model_status, 'options': options,
        'integer_abi_bytes': 4, 'elevated': False, 'network_audit_canary_denied': denied,
        'loaded_native_files_sha256': loaded,
        'network_enforcement': 'PYTHON_AUDIT_ONLY_NOT_OS_ISOLATION',
        'native_network_isolation': 'NOT_ATTESTED', 'python_version': sys.version}}, allow_nan=False))
    dll_directory.close()


if __name__ == '__main__': main()
