"""Pinned original Pyomo polynomial model construction composed with native HiGHS.

Fixed public core/repn APIs only. No researcher Python, setup.py, arbitrary plugins,
solver commands, install or network; this is not complete Pyomo functionality.
"""
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

from . import highs_native
from .strict_json import loads_strict
from .execution_control import remaining_timeout, safe_local_path, run_worker

COMMIT = 'fac771679b35a3b6292b1f48729d5298d58e7a57'
ARCHIVE_NAME = 'Pyomo-fac771679b35.tar.gz'
ARCHIVE_SHA256 = 'ebaa508ad3cdbd58512bc00c2eaa2beca0676c2add7a6e52ce586ad32f00647d'
ROOT = 'pyomo-'+COMMIT
VERSION = '6.10.2.a0'
PREPARED_SUBPACKAGES = {'common', 'core', 'opt', 'dataportal', 'repn', 'tpl',
                        'version', 'gdp', 'util', 'network', 'mpec'}
# A fixed-workflow source boundary: core.relax_integer_vars actually imports
# only the bundled FBBT contribution, not all unrelated contributed plugins.
# The entire archive is still SHA-pinned and every archive member is audited.
PREPARED_CONTRIB_FILES = frozenset({
    'pyomo/contrib/__init__.py',
    'pyomo/contrib/fbbt/__init__.py',
    'pyomo/contrib/fbbt/expression_bounds_walker.py',
    'pyomo/contrib/fbbt/interval.py',
    'pyomo/contrib/fbbt/fbbt.py',
})
APIS = ['pyomo.core.ConcreteModel', 'pyomo.core.Var', 'pyomo.core.base.var.VarData.is_integer',
        'pyomo.core.Objective', 'pyomo.core.base.objective.ObjectiveData.is_minimizing',
        'pyomo.core.ConstraintList', 'pyomo.core.ConstraintList.add',
        'pyomo.repn.generate_standard_repn', 'pyomo.repn.StandardRepn.is_linear',
        'pyomo.repn.StandardRepn.is_quadratic', 'pyomo.core.value', 'pyomo.core.differentiate']


def digest(raw): return hashlib.sha256(raw).hexdigest()


def _read(path, cap):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > cap:
        raise ValueError('PYOMO_SOURCE_FILE_INVALID')
    with path.open('rb') as stream: raw = stream.read(cap+1)
    if len(raw) > cap: raise ValueError('PYOMO_SOURCE_SIZE_INVALID')
    return raw


def audit_archive(path):
    raw = _read(path, 20_000_000)
    if digest(raw) != ARCHIVE_SHA256: raise ValueError('PYOMO_ARCHIVE_DIGEST_MISMATCH')
    files, seen, total = {}, set(), 0
    reserved = {'CON', 'PRN', 'AUX', 'NUL'} | {f'{p}{i}' for p in ('COM', 'LPT') for i in range(1, 10)}
    with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as archive:
        for index, member in enumerate(archive):
            parts = member.name.rstrip('/').split('/')
            if (index >= 4200 or parts[0] != ROOT or len(member.name) > 240 or '\\' in member.name or
                    any(p in ('', '.', '..') or ':' in p or p.endswith((' ', '.')) or
                        p.casefold() in {'private', 'windows-api-credentials', '.git'} or
                        p.split('.')[0].upper() in reserved or
                        any(ord(c) < 32 or c in '<>"|?*' for c in p) for p in parts)):
                raise ValueError('PYOMO_ARCHIVE_PATH_INVALID')
            if member.name.casefold() in seen: raise ValueError('PYOMO_ARCHIVE_DUPLICATE')
            seen.add(member.name.casefold())
            if not (member.isfile() or member.isdir()): raise ValueError('PYOMO_ARCHIVE_LINK_OR_SPECIAL')
            total += member.size
            if member.size < 0 or member.size > 6_000_000 or total > 60_000_000:
                raise ValueError('PYOMO_ARCHIVE_SIZE_INVALID')
            relative = '/'.join(parts[1:])
            selected = relative in ('LICENSE.md', 'pyproject.toml') or (
                relative.startswith('pyomo/') and relative.endswith('.py') and
                (relative == 'pyomo/__init__.py' or
                 relative.split('/')[1] in PREPARED_SUBPACKAGES or
                 relative in PREPARED_CONTRIB_FILES) and
                not any(p in {'tests', 'test', 'examples', '__pycache__'} for p in parts))
            if member.isfile() and selected: files[relative] = archive.extractfile(member).read()
    if not PREPARED_CONTRIB_FILES <= files.keys():
        raise ValueError('PYOMO_REQUIRED_CONTRIB_FILES_MISSING')
    license_text = files.get('LICENSE.md', b'').decode('utf-8')
    if ('Redistributions of source code must retain' not in license_text or
            'Neither the name of the Sandia National Laboratories' not in license_text):
        raise ValueError('PYOMO_LICENSE_TEXT_INVALID')
    return files


def verify_prepared(root, files):
    root = Path(root); observed = {}
    for path in root.rglob('*'):
        if path.is_symlink(): raise ValueError('PYOMO_PREPARED_LINK')
        if path.is_file(): observed[path.relative_to(root).as_posix()] = digest(_read(path, 6_000_000))
    if observed != {name: digest(raw) for name, raw in files.items()}:
        raise ValueError('PYOMO_PREPARED_TAMPER')


def check_representation(original, actual):
    a = highs_native._model(original)
    if type(actual) is not dict or set(actual) != set(original):
        raise ValueError('PYOMO_REPRESENTATION_SCHEMA_INVALID')
    for key in ('kind', 'sense', 'integrality'):
        if a[key] != actual[key]: raise ValueError('PYOMO_REPRESENTATION_DOMAIN_MISMATCH')
    if type(actual['integrality']) is not list or any(type(v) is not int for v in actual['integrality']):
        raise ValueError('PYOMO_REPRESENTATION_DOMAIN_MISMATCH')
    def check(observed, wanted):
        if type(wanted) is list:
            if type(observed) is not list or len(observed) != len(wanted):
                raise ValueError('PYOMO_REPRESENTATION_DIMENSION_MISMATCH')
            for x, y in zip(observed, wanted): check(x, y)
        elif (type(observed) not in (int, float) or not math.isfinite(observed) or
              float(observed).hex() != float(wanted).hex()):
            raise ValueError('PYOMO_REPRESENTATION_COEFFICIENT_MISMATCH')
    for key in ('objective', 'constraints', 'row_lower', 'row_upper', 'lower', 'upper', 'hessian', 'offset'):
        check(actual[key], [] if key == 'hessian' and a['kind'] != 'QP' else a[key])


def _native_encoding(original, canonical):
    """Exact rational transport whose reconversion is the original Pyomo double.

    Native decimal renderings can have longer denominators than declared rational
    inputs. Keep raw Pyomo values as evidence and verify every double is identical
    before using the declared rational as its lossless C-buffer encoding. No
    coefficients are rounded, dropped or substituted within a tolerance.
    """
    check_representation(original, canonical)
    a = highs_native._model(original)
    def encode(value): return [encode(v) for v in value] if type(value) is list else str(value)
    result = {k: encode(a[k]) for k in ('objective', 'constraints', 'row_lower', 'row_upper', 'lower', 'upper', 'offset')}
    result.update(kind=canonical['kind'], sense=canonical['sense'], integrality=canonical['integrality'],
                  hessian=encode(a['hessian']) if a['kind'] == 'QP' else [])
    return result


def check_derivatives(spec, actual):
    data = highs_native._model(spec)
    keys = {'method', 'gradient_constant', 'gradient_linear', 'hessian', 'constraint_jacobian'}
    if type(actual) is not dict or set(actual) != keys or actual['method'] != 'REVERSE_SYMBOLIC':
        raise ValueError('PYOMO_DERIVATIVE_SCHEMA_INVALID')
    expected = {'gradient_constant': data['objective'], 'gradient_linear': data['hessian'],
                'hessian': data['hessian'], 'constraint_jacobian': data['constraints']}
    def check(observed, wanted):
        if type(wanted) is list:
            if type(observed) is not list or len(observed) != len(wanted):
                raise ValueError('PYOMO_DERIVATIVE_DIMENSION_MISMATCH')
            for x, y in zip(observed, wanted): check(x, y)
        else:
            if (type(observed) not in (int, float) or not math.isfinite(observed) or
                    not math.isclose(float(observed), float(wanted), abs_tol=1e-12, rel_tol=1e-12)):
                raise ValueError('PYOMO_DERIVATIVE_ORACLE_MISMATCH')
    for key, wanted in expected.items(): check(actual[key], wanted)


def relaxed_spec(spec):
    highs_native.validate(spec)
    if spec['kind'] != 'MILP': raise ValueError('PYOMO_RELAXATION_REQUIRES_MILP')
    result = loads_strict(highs_native._bytes(spec))
    result.update(kind='LP', integrality=[0]*len(spec['integrality']))
    return result


def canonicalize(spec, sources, *, transformation=None):
    highs_native.validate(spec)
    if transformation not in (None, 'RELAX_INTEGER_VARS'):
        raise ValueError('PYOMO_TRANSFORMATION_UNSUPPORTED')
    expected_spec = spec if transformation is None else relaxed_spec(spec)
    if sources is None: raise ValueError('PYOMO_PINNED_SOURCES_REQUIRED')
    sources = safe_local_path(sources)
    safe_local_path(sources/'manifest.json'); safe_local_path(sources/ARCHIVE_NAME)
    manifest_raw = _read(sources/'manifest.json', 256*1024)
    manifest = loads_strict(manifest_raw)
    rows = manifest.get('archives') if type(manifest) is dict else None
    if type(rows) is not list or len(rows) > 256 or any(type(r) is not dict for r in rows):
        raise ValueError('PYOMO_MANIFEST_INVALID')
    rows = [r for r in rows if r.get('name') == 'Pyomo']
    expected = {'commit': COMMIT, 'file': ARCHIVE_NAME, 'sha256': ARCHIVE_SHA256,
                'origin_url': 'https://github.com/Pyomo/pyomo', 'bytes': 16536371,
                'license_reference_hint': 'BSD-3-Clause', 'license_spdx_from_github': 'NOASSERTION'}
    if len(rows) != 1 or any(rows[0].get(k) != v for k, v in expected.items()):
        raise ValueError('PYOMO_MANIFEST_PIN_MISMATCH')
    archive = (sources/ARCHIVE_NAME).resolve()
    files = audit_archive(sources/ARCHIVE_NAME)
    worker = Path(__file__).with_name('pyomo_model_worker.py').resolve()
    with tempfile.TemporaryDirectory(prefix='societybox-pyomo-') as temp:
        root = Path(temp)
        for name, raw in files.items():
            path = root/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
        verify_prepared(root, files)
        try:
            child = run_worker([sys.executable, '-I', '-S', '-B', str(worker), str(root), str(archive),
                'none' if transformation is None else 'relax_integer_vars'],
                input=highs_native._bytes(spec), capture_output=True, timeout=remaining_timeout(30), cwd=root,
                env={'SYSTEMROOT': os.environ.get('SYSTEMROOT', '')},
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except subprocess.TimeoutExpired as error: raise RuntimeError('PYOMO_WORKER_TIMEOUT_REAPED') from error
        if child.returncode or len(child.stdout) > 256*1024:
            raise RuntimeError('PYOMO_FIXED_WORKER_FAILED')
        result = loads_strict(child.stdout)
        keys = {'canonical', 'upstream', 'derivatives'} | ({'transformation'} if transformation else set())
        if type(result) is not dict or set(result) != keys:
            raise ValueError('PYOMO_OUTPUT_SCHEMA_INVALID')
        check_representation(expected_spec, result['canonical'])
        check_derivatives(expected_spec, result['derivatives'])
        if transformation:
            trace = result['transformation']
            if (type(trace) is not dict or set(trace) != {'name', 'operation', 'original_before', 'original_after', 'original_model_unchanged'} or
                    trace['name'] != 'core.relax_integer_vars' or trace['operation'] != 'CREATE_USING_CLONE' or
                    trace['original_model_unchanged'] is not True or
                    highs_native._bytes(trace['original_before']) != highs_native._bytes(trace['original_after'])):
                raise ValueError('PYOMO_TRANSFORMATION_RECEIPT_INVALID')
            check_representation(spec, trace['original_before'])
            check_representation(spec, trace['original_after'])
        upstream = result['upstream']
        keys = {'version', 'commit', 'apis', 'elevated', 'network_audit_canary_denied', 'dynamic_library_guard_canary_denied',
                'import_file_sha256', 'stdlib_aliases', 'deferred_dependency_objects_not_imported',
                'runtime_dependencies', 'python_version', 'isolation'}
        if (type(upstream) is not dict or set(upstream) != keys or upstream['dynamic_library_guard_canary_denied'] is not True or
                upstream['runtime_dependencies'] != [] or upstream['python_version'] != sys.version or
                upstream['isolation'] != 'TRUST_PINNED_FIXED_WORKER_ONLY_NOT_OS_SANDBOX' or
                type(upstream['stdlib_aliases']) is not dict or len(upstream['stdlib_aliases']) > 64 or
                type(upstream['deferred_dependency_objects_not_imported']) is not dict or
                len(upstream['deferred_dependency_objects_not_imported']) > 256 or
                any(type(k) is not str or not k.startswith('pyomo.') or
                    v not in ('pyomo.common.dependencies.DeferredImportModule', 'pyomo.common.dependencies.ModuleUnavailable')
                    for k, v in upstream['deferred_dependency_objects_not_imported'].items())):
            raise ValueError('PYOMO_UPSTREAM_PROVENANCE_MISMATCH')
        imports = upstream['import_file_sha256']
        if type(imports) is not dict or len(imports) > 512: raise ValueError('PYOMO_IMPORT_SET_INVALID')
        for alias in upstream['stdlib_aliases'].values():
            if type(alias) is not dict or set(alias) not in ({'module', 'origin'}, {'module', 'origin', 'sha256'}):
                raise ValueError('PYOMO_STDLIB_ALIAS_SCHEMA_INVALID')
            if alias['module'].split('.')[0] not in sys.stdlib_module_names:
                raise ValueError('PYOMO_UNREVIEWED_STDLIB_ALIAS')
            if alias['origin'] != 'BUILTIN_OR_FROZEN':
                parts = alias['origin'].split('/')
                if any(p in ('', '.', '..') or ':' in p or '\\' in p or
                       p.casefold() in {'private', 'windows-api-credentials', 'site-packages'} for p in parts):
                    raise ValueError('PYOMO_STDLIB_ALIAS_PATH_INVALID')
                path = Path(os.__file__).resolve().parent.joinpath(*parts)
                if digest(_read(path, 6_000_000)) != alias['sha256']:
                    raise ValueError('PYOMO_STDLIB_ALIAS_HASH_MISMATCH')
        expected_apis = [a for a in APIS if spec['constraints'] or a != 'pyomo.core.ConstraintList.add']
        if transformation:
            expected_apis += ['pyomo.core.TransformationFactory', 'pyomo.core.base.transformation.Transformation.create_using']
        if (upstream['version'] != VERSION or upstream['commit'] != COMMIT or
                upstream['elevated'] is not False or upstream['network_audit_canary_denied'] is not True or
                upstream['apis'] != sorted(expected_apis) or not imports or
                any(name not in files or digest(files[name]) != sha for name, sha in imports.items()) or
                not {'pyomo/core/base/PyomoModel.py', 'pyomo/repn/standard_repn.py'} <= set(imports)):
            raise ValueError('PYOMO_UPSTREAM_PROVENANCE_MISMATCH')
        verify_prepared(root, files)
    return {'canonical': result['canonical'], 'derivatives': result['derivatives'], 'upstream': upstream,
            **({'transformation': result['transformation']} if transformation else {}),
            'native_program': _native_encoding(expected_spec, result['canonical']),
            'native_number_encoding': 'DECLARED_RATIONAL_WITH_IDENTICAL_BINARY64_C_BUFFER_RECONVERSION',
            'archive_sha256': ARCHIVE_SHA256, 'license_sha256': digest(files['LICENSE.md']),
            'license': 'BSD-3-Clause_ROOT_WITH_BUNDLED_NOTICES_NOT_REDISTRIBUTION_CLEARANCE',
            'manifest_sha256': digest(manifest_raw), 'worker_sha256': digest(worker.read_bytes()),
            'prepared_source_sha256': digest(highs_native._bytes({k: digest(v) for k, v in files.items()})),
            'child_exit': child.returncode, 'source_files_unchanged': True,
            'prepared_package_scope': sorted(PREPARED_SUBPACKAGES | {'contrib/fbbt'}),
            'temporary_source_removed_on_return': True, 'os_isolation': 'NOT_ATTESTED'}


def run(spec, sources, runtime):
    modeled = canonicalize(spec, sources)
    solved = highs_native.run(modeled['native_program'], runtime)
    comparison = highs_native.compare(spec, solved['actual'])
    return {'engine': 'PYOMO_PINNED_MODEL_HIGHS_C_API', 'mode': 'NO_LLM',
            'native_execution': 'EXECUTED', 'native_parity': 'WITHIN_DECLARED_ORACLE_TOLERANCE',
            'actual': solved['actual'], 'comparison': comparison,
            'upstream': {'modeling': modeled, 'solver': solved['upstream']},
            'solver_worker_sha256': solved['worker_sha256'], 'os_isolation': 'NOT_ATTESTED',
            'assumptions': solved['assumptions']+[
                'Original Pyomo polynomial construction/representation composed with original native HiGHS.',
                'Symbolic marginals describe the continuous polynomial extension; integer decisions do not imply feasible differential moves.',
                'Other Pyomo models, plugins, nonlinear solvers and full API coverage remain unimplemented.']}
