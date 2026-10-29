"""Fixed original Pyomo core/repn worker; no untrusted Python model intake."""
import ctypes
# Trusted stdlib initialization loads Windows kernel32 in Python 3.14. Complete
# it before the audit; later DLL loading remains denied, including the canary.
import ctypes.util
import json
import os
from pathlib import Path
import platform
import socket
import sys
from types import ModuleType


def main():
    elevated = bool(ctypes.windll.shell32.IsUserAnAdmin()) if os.name == 'nt' else os.geteuid() == 0
    if elevated: raise RuntimeError('PYOMO_NONADMIN_REQUIRED')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from econscope.pyomo_model import audit_archive, verify_prepared, digest, COMMIT, APIS
    from econscope.highs_native import _model
    from econscope.strict_json import loads_strict
    raw = sys.stdin.buffer.read(65537)
    if len(raw) > 65536: raise ValueError('PYOMO_INPUT_BUDGET')
    spec = loads_strict(raw); data = _model(spec)
    mode = sys.argv[3] if len(sys.argv) > 3 else 'none'
    if mode not in ('none', 'relax_integer_vars') or (mode != 'none' and data['kind'] != 'MILP'):
        raise ValueError('PYOMO_TRANSFORMATION_INVALID')
    # Genuine local stdlib metadata. On Windows platform.system() calls
    # gethostname; initialize its cache before the strict network audit and
    # before loading any upstream code. No socket/connection is opened here.
    platform.system()
    def guard(event, args):
        if event.startswith(('socket.', 'subprocess.', 'os.system', 'os.spawn', 'os.exec', 'ctypes.dlopen')):
            raise RuntimeError('PYOMO_NETWORK_OR_EXTERNAL_PROCESS_DENIED')
    sys.addaudithook(guard)
    denied = False
    try: socket.socket()
    except RuntimeError: denied = True
    if not denied: raise RuntimeError('PYOMO_NETWORK_CANARY_FAILED')
    dynamic_library_denied=False
    try:ctypes.CDLL('kernel32' if os.name=='nt' else None)
    except RuntimeError:dynamic_library_denied=True
    if not dynamic_library_denied:raise RuntimeError('PYOMO_DYNAMIC_LIBRARY_CANARY_FAILED')
    root = Path(sys.argv[1]).resolve(strict=True)
    files = audit_archive(sys.argv[2]); verify_prepared(root, files)
    sys.path.insert(0, str(root))
    import pyomo
    from pyomo.core import ConcreteModel, Var, Reals, Integers, Binary, Objective, ConstraintList, minimize, maximize, value, differentiate
    from pyomo.repn import generate_standard_repn
    n = len(data['objective'])
    model = ConcreteModel()
    def domain(_, i):
        return (Binary if data['lower'][i] == 0 and data['upper'][i] == 1 else Integers) if data['integrality'][i] else Reals
    model.x = Var(range(n), domain=domain,
                  bounds=lambda _, i: (float(data['lower'][i]), float(data['upper'][i])))
    expression = float(data['offset'])+sum(float(c)*model.x[i] for i, c in enumerate(data['objective']))
    if data['kind'] == 'QP':
        expression += sum(float(data['hessian'][i][j])*model.x[i]*model.x[j]/2 for i in range(n) for j in range(n))
    model.cost = Objective(expr=expression, sense=minimize if data['sense'] == 'minimize' else maximize)
    model.rows = ConstraintList()
    for row, lo, hi in zip(data['constraints'], data['row_lower'], data['row_upper']):
        model.rows.add((float(lo), sum(float(c)*model.x[i] for i, c in enumerate(row)), float(hi)))
    def represent(model, kind):
        ids = {id(model.x[i]): i for i in range(n)}
        repn = generate_standard_repn(model.cost.expr, quadratic=True)
        linear, quadratic = repn.is_linear(), repn.is_quadratic()
        if not (linear or quadratic): raise ValueError('PYOMO_NONPOLYNOMIAL_REPRESENTATION')
        cost = [0.0]*n; hessian = [[0.0]*n for _ in range(n)]
        for variable, coefficient in zip(repn.linear_vars, repn.linear_coefs): cost[ids[id(variable)]] += float(coefficient)
        for (first, second), coefficient in zip(repn.quadratic_vars, repn.quadratic_coefs):
            i, j = ids[id(first)], ids[id(second)]
            if i == j: hessian[i][i] += 2*float(coefficient)
            else: hessian[i][j] += float(coefficient); hessian[j][i] += float(coefficient)
        a, lower, upper = [], [], []
        for constraint in model.rows.values():
            row_repn = generate_standard_repn(constraint.body)
            if not row_repn.is_linear(): raise ValueError('PYOMO_NONLINEAR_ROW')
            row = [0.0]*n
            for variable, coefficient in zip(row_repn.linear_vars, row_repn.linear_coefs): row[ids[id(variable)]] += float(coefficient)
            a.append(row)
            lower.append(float(value(constraint.lower))-float(row_repn.constant))
            upper.append(float(value(constraint.upper))-float(row_repn.constant))
        canonical = {'kind': kind, 'sense': 'minimize' if model.cost.is_minimizing() else 'maximize',
                     'objective': cost, 'constraints': a, 'row_lower': lower, 'row_upper': upper,
                     'lower': [float(model.x[i].lb) for i in range(n)], 'upper': [float(model.x[i].ub) for i in range(n)],
                     'integrality': [int(model.x[i].is_integer()) for i in range(n)],
                     'offset': float(value(repn.constant)), 'hessian': hessian if kind == 'QP' else []}
        variables = [model.x[i] for i in range(n)]
        gradients = differentiate(model.cost.expr, wrt_list=variables, mode='reverse_symbolic')
        constants, gradient_linear, derivative_hessian = [], [], []
        for expression in gradients:
            derivative_repn = generate_standard_repn(expression)
            if not derivative_repn.is_linear(): raise ValueError('PYOMO_NONLINEAR_GRADIENT')
            constants.append(float(value(derivative_repn.constant)))
            row = [0.0]*n
            for variable, coefficient in zip(derivative_repn.linear_vars, derivative_repn.linear_coefs): row[ids[id(variable)]] += float(coefficient)
            gradient_linear.append(row)
            derivative_hessian.append([float(value(v)) for v in differentiate(expression, wrt_list=variables, mode='reverse_symbolic')])
        derivatives = {'method': 'REVERSE_SYMBOLIC', 'gradient_constant': constants,
            'gradient_linear': gradient_linear, 'hessian': derivative_hessian,
            'constraint_jacobian': [[float(value(v)) for v in differentiate(constraint.body,
                wrt_list=variables, mode='reverse_symbolic')] for constraint in model.rows.values()]}
        return canonical, derivatives
    canonical, derivatives = represent(model, data['kind'])
    transformation, extra_apis = None, []
    if mode == 'relax_integer_vars':
        from pyomo.core import TransformationFactory
        import pyomo.core.plugins.transform.discrete_vars
        original_before = canonical
        relaxed = TransformationFactory('core.relax_integer_vars').create_using(model)
        original_after, after_derivatives = represent(model, data['kind'])
        if original_before != original_after or derivatives != after_derivatives:
            raise ValueError('PYOMO_ORIGINAL_MODEL_CHANGED_BY_CLONE_TRANSFORM')
        canonical, derivatives = represent(relaxed, 'LP')
        transformation = {'name': 'core.relax_integer_vars', 'operation': 'CREATE_USING_CLONE',
                          'original_before': original_before, 'original_after': original_after,
                          'original_model_unchanged': True}
        extra_apis = ['pyomo.core.TransformationFactory', 'pyomo.core.base.transformation.Transformation.create_using']
    imports, stdlib_aliases, deferred = {}, {}, {}
    for name, module in list(sys.modules.items()):
        if name == 'pyomo' or name.startswith('pyomo.'):
            # Original Pyomo publishes deferred dependency objects/stdlib aliases
            # in sys.modules. Do not force those optional imports by reading
            # __file__; they are explicitly not counted as imported source APIs.
            if type(module) is not ModuleType:
                descriptor = type(module).__module__+'.'+type(module).__name__
                if descriptor not in ('pyomo.common.dependencies.DeferredImportModule',
                                      'pyomo.common.dependencies.ModuleUnavailable'):
                    raise ValueError('PYOMO_UNREVIEWED_MODULE_OBJECT:'+descriptor)
                deferred[name] = descriptor
                continue
            actual_name = module.__name__
            if actual_name.split('.')[0] in sys.stdlib_module_names:
                if sys.modules.get(actual_name) is not module: raise ValueError('PYOMO_STDLIB_ALIAS_IDENTITY')
                origin = getattr(module, '__file__', None)
                if origin is None:
                    stdlib_aliases[name] = {'module': actual_name, 'origin': 'BUILTIN_OR_FROZEN'}
                else:
                    path = Path(origin).resolve(strict=True)
                    relative = path.relative_to(Path(os.__file__).resolve().parent).as_posix()
                    stdlib_aliases[name] = {'module': actual_name, 'origin': relative, 'sha256': digest(path.read_bytes())}
                continue
            path = Path(module.__file__).resolve(strict=True)
            relative = path.relative_to(root).as_posix()
            if relative not in files or path.read_bytes() != files[relative]: raise ValueError('PYOMO_IMPORT_TAMPER')
            imports[relative] = digest(files[relative])
    result = {'canonical': canonical, 'derivatives': derivatives, 'upstream': {'version': pyomo.__version__,
        'commit': COMMIT, 'apis': sorted([a for a in APIS if data['constraints'] or a != 'pyomo.core.ConstraintList.add']+extra_apis), 'elevated': elevated,
        'network_audit_canary_denied': denied, 'import_file_sha256': imports,
        'stdlib_aliases': stdlib_aliases, 'deferred_dependency_objects_not_imported': deferred,
        'runtime_dependencies': [], 'python_version': sys.version,
        'isolation': 'TRUST_PINNED_FIXED_WORKER_ONLY_NOT_OS_SANDBOX',
        'dynamic_library_guard_canary_denied':dynamic_library_denied}}
    if transformation is not None: result['transformation'] = transformation
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__': main()
