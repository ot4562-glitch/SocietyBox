"""Bounded pinned SimPy service queue; conditional synthetic model, no code intake.

Run: python -m econscope.simpy_queue --sources references/oss-candidates-20261024
      --input examples/simpy_market_queue.json
No install, download, setup.py execution or redistribution of upstream source.
"""
import argparse
import ast
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

COMMIT = '9ba33ff8e8f23391f27572d03e4fe874bbf5f12c'
ARCHIVE_NAME = 'SimPy-9ba33ff8e8f2.tar.gz'
ARCHIVE_SHA256 = 'a0a8887968f91faa03cbbc76652ac75ccdb002257997efa412f1131fbdaa2bbf'
ROOT = 'simpy-' + COMMIT


def digest(data):
    return hashlib.sha256(data).hexdigest()


def validate_manifest(manifest):
    if type(manifest) is not dict:
        raise ValueError('MANIFEST_SCHEMA_INVALID')
    entries = manifest.get('archives')
    if type(entries) is not list or len(entries) > 256 or any(
            type(item) is not dict for item in entries):
        raise ValueError('MANIFEST_SCHEMA_INVALID')
    rows = [x for x in entries if x.get('name') == 'SimPy']
    if len(rows) != 1:
        raise ValueError('MANIFEST_SIMPY_REQUIRED')
    row = rows[0]
    expected = {'commit': COMMIT, 'sha256': ARCHIVE_SHA256, 'file': ARCHIVE_NAME,
                'origin_url': 'https://github.com/simpx/simpy',
                'license_spdx_from_github': 'MIT', 'bytes': 402037}
    if any(row.get(k) != v for k, v in expected.items()):
        raise ValueError('MANIFEST_PIN_MISMATCH')
    return row


def validate_member(member, seen, total):
    name = member.name
    parts = name.split('/')
    reserved = {'CON', 'PRN', 'AUX', 'NUL'} | {f'{p}{i}' for p in ('COM', 'LPT') for i in range(1, 10)}
    if (parts[0] != ROOT or len(name) > 240 or '\\' in name or
            any(not p or p in ('.', '..') or ':' in p or p.endswith((' ', '.')) or
                p.split('.')[0].upper() in reserved or
                any(ord(c) < 32 or c in '<>"|?*' for c in p) for p in parts) or
            not (member.isfile() or member.isdir())):
        raise ValueError('ARCHIVE_MEMBER_PATH_OR_LINK_INVALID')
    if name.casefold() in seen:
        raise ValueError('ARCHIVE_DUPLICATE')
    seen.add(name.casefold())
    if member.size < 0 or member.size > 1_000_000 or total + member.size > 2_000_000:
        raise ValueError('ARCHIVE_SIZE_INVALID')
    return total + member.size


def audit_archive(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 500_000:
        raise ValueError('ARCHIVE_FILE_INVALID')
    data = path.read_bytes()
    if digest(data) != ARCHIVE_SHA256:
        raise ValueError('ARCHIVE_DIGEST_MISMATCH')
    files, seen, total = {}, set(), 0
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as archive:
        for i, member in enumerate(archive):
            if i >= 150:
                raise ValueError('ARCHIVE_MEMBER_BUDGET')
            total = validate_member(member, seen, total)
            if member.isfile():
                files[member.name[len(ROOT)+1:]] = archive.extractfile(member).read()
    license_text = files['LICENSE.rst'].decode('utf-8')
    if ('The MIT License (MIT)' not in license_text or
            'The above copyright notice and this permission notice shall be included' not in license_text):
        raise ValueError('LICENSE_TEXT_MISMATCH')
    # Inspect, never execute packaging code. The immutable source pins the rest
    # of the license text, packaging declaration and each runtime file as well.
    setup = ast.parse(files['setup.py'])
    requirements = [k.value for n in ast.walk(setup) if isinstance(n, ast.Call)
                    for k in n.keywords if k.arg == 'install_requires']
    if len(requirements) != 1 or ast.literal_eval(requirements[0]) != []:
        raise ValueError('RUNTIME_DEPENDENCIES_NOT_REVIEWED')
    imports = {}
    allowed = {'simpy', 'pkgutil', 'collections', 'sys', 'traceback', 'heapq', 'itertools', 'types', 'time'}
    for name, body in files.items():
        if name.startswith('src/simpy/'):
            if not name.endswith('.py'):
                raise ValueError('UNREVIEWED_RUNTIME_FILE')
            tree = ast.parse(body)
            modules = {n.module or '' for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
            modules |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
            if any(m.split('.')[0] not in allowed for m in modules):
                raise ValueError('UNREVIEWED_RUNTIME_IMPORT')
            imports[name] = sorted(modules)
    return files, {'archive_sha256': digest(data), 'commit': COMMIT,
                   'member_count': len(seen), 'declared_bytes': total,
                   'license_text': license_text, 'license_sha256': digest(files['LICENSE.rst']),
                   'runtime_dependencies': [], 'runtime_import_review': imports,
                   'redistribution': 'NOT_AUTHORIZED', 'setup_executed': False}


def validate_spec(spec):
    keys = {'arrivals', 'service_times', 'system_capacity', 'waiting_cost_per_tick'}
    if type(spec) is not dict or set(spec) not in (keys, keys | {'server_count'}):
        raise ValueError('QUEUE_SCHEMA_INVALID')
    a, s = spec['arrivals'], spec['service_times']
    if (type(a) is not list or type(s) is not list or not 1 <= len(a) <= 100 or len(a) != len(s)):
        raise ValueError('QUEUE_JOBS_INVALID')
    values = a + s + [spec['system_capacity'], spec['waiting_cost_per_tick']]
    if any(type(v) is not int or not 0 <= v <= 1_000_000 for v in values):
        raise ValueError('QUEUE_INTEGER_BOUNDS_INVALID')
    if a != sorted(a) or min(s) < 1 or not 1 <= spec['system_capacity'] <= 100:
        raise ValueError('QUEUE_PARAMETERS_INVALID')
    servers = spec.get('server_count', 1)
    if type(servers) is not int or not 1 <= servers <= min(16, spec['system_capacity']):
        raise ValueError('QUEUE_SERVER_COUNT_INVALID')
    return spec


def verify_prepared_sources(root, files):
    """Verify the complete disposable namespace before any upstream import."""
    root = Path(root)
    expected = {name.removeprefix('src/'): digest(data) for name, data in files.items()
                if name.startswith('src/simpy/') or name == 'LICENSE.rst'}
    actual = {}
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ValueError('PREPARED_SOURCE_LINK_INVALID')
        if path.is_file():
            actual[path.relative_to(root).as_posix()] = digest(path.read_bytes())
    if actual != expected:
        raise ValueError('PREPARED_SOURCE_DIGEST_OR_FILESET_MISMATCH')


def queue_oracle(spec):
    """Independent FIFO parallel-server recurrence; no event engine import."""
    validate_spec(spec)
    jobs, lost = [], []
    available = [0] * spec.get('server_count', 1)
    for i, (arrival, service) in enumerate(zip(spec['arrivals'], spec['service_times'])):
        # All arrivals at t precede completions at t.
        active = sum(j['finish'] >= arrival for j in jobs)
        if active >= spec['system_capacity']:
            lost.append(i)
            continue
        slot = min(range(len(available)), key=available.__getitem__)
        start = max(arrival, available[slot])
        end = start + service
        available[slot] = end
        jobs.append({'id': i, 'arrival': arrival, 'start': start,
                     'finish': end, 'wait': start-arrival})
    times = sorted(set(spec['arrivals']) | {j[k] for j in jobs for k in ('start', 'finish')})
    states = [{'time': t, 'busy': sum(j['start'] <= t < j['finish'] for j in jobs),
               'queued': sum(j['arrival'] <= t < j['start'] for j in jobs),
               'completed': sum(j['finish'] <= t for j in jobs),
               'lost': sum(spec['arrivals'][i] <= t for i in lost)} for t in times]
    return {'jobs': jobs, 'lost_jobs': lost, 'states': states,
            'waiting_cost': sum(j['wait'] for j in jobs)*spec['waiting_cost_per_tick'],
            'busy_ticks': sum(j['finish']-j['start'] for j in jobs)}


def check_result(spec, actual):
    if actual != queue_oracle(spec):
        raise ValueError('ORACLE_MISMATCH')


def run_queue(spec, sources):
    from .execution_control import remaining_timeout, safe_local_path, run_worker
    validate_spec(spec)
    sources = safe_local_path(sources)
    safe_local_path(sources/'manifest.json'); safe_local_path(sources/ARCHIVE_NAME)
    manifest_path = sources / 'manifest.json'
    # Bounded read even if the file grows after stat, before JSON decoding.
    if (manifest_path.is_symlink() or not manifest_path.is_file() or
            manifest_path.stat().st_size > 256 * 1024):
        raise ValueError('MANIFEST_SIZE_INVALID')
    with manifest_path.open('rb') as stream:
        manifest_data = stream.read(256 * 1024 + 1)
    if len(manifest_data) > 256 * 1024:
        raise ValueError('MANIFEST_SIZE_INVALID')
    validate_manifest(json.loads(manifest_data))
    files, audit = audit_archive(sources / ARCHIVE_NAME)
    with tempfile.TemporaryDirectory(prefix='societybox-simpy-') as temp:
        temp = Path(temp)
        hashes = {}
        for name, data in files.items():
            if name.startswith('src/simpy/') or name == 'LICENSE.rst':
                relative = name.removeprefix('src/')
                destination = temp / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
                hashes[relative] = digest(data)
        worker = Path(__file__).with_name('simpy_queue_worker.py').resolve()
        # Only this fixed reviewed worker may execute; researcher JSON is stdin.
        command = [sys.executable, '-I', '-S', '-B', str(worker), str(temp),
                   str((sources / ARCHIVE_NAME).resolve())]
        try:
            result = run_worker(command, input=json.dumps(spec), text=True,
                capture_output=True, cwd=temp, timeout=remaining_timeout(15),
                env={'SYSTEMROOT': __import__('os').environ.get('SYSTEMROOT', '')},
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except subprocess.TimeoutExpired as error:
            # subprocess.run kills and waits for its child before raising.
            # Leaving this context then removes the disposable source. Return
            # a stable refusal instead of leaking a raw command/path exception.
            raise RuntimeError('UPSTREAM_WORKER_TIMEOUT_REAPED') from error
        if result.returncode:
            raise RuntimeError('UPSTREAM_IMPORT_OR_EXECUTION_BLOCKED: ' + result.stderr[-1800:])
        receipt = json.loads(result.stdout)
        if receipt['upstream']['version'] != '3.0.11' or receipt['upstream']['elevated']:
            raise ValueError('UPSTREAM_VERSION_OR_PRIVILEGE_INVALID')
        for relative, record in receipt['upstream']['import_files'].items():
            if (relative not in hashes or record['sha256'] != hashes[relative] or
                    Path(record['path']).resolve() != (temp / relative).resolve()):
                raise ValueError('UPSTREAM_IMPORT_PROVENANCE_MISMATCH')
        for relative, expected in hashes.items():
            if digest((temp / relative).read_bytes()) != expected:
                raise ValueError('UPSTREAM_SOURCE_CHANGED')
        verify_prepared_sources(temp, files)
        check_result(spec, receipt['actual'])
        receipt['upstream'].update(audit)
        receipt.update({'schema': 'societybox-simpy-queue-v1', 'status': 'ENGINEERING_CANDIDATE',
            'comparison': 'EXACT_MATCH', 'spec': spec,
            'prepared_file_sha256': hashes,
            'manifest_sha256': digest(manifest_data), 'worker_sha256': digest(worker.read_bytes()),
            'child_argv': command, 'child_exit': result.returncode,
            'source_files_unchanged': True, 'temporary_source_removed_on_return': True,
            'limits': ['Synthetic conditional FIFO model only; no empirical or causal inference.',
                       'One pinned API slice, not general SimPy parity.',
                       'Python audit guards and timeout are not general OS isolation.',
                       'No arbitrary researcher/LLM code accepted; no install or redistribution.']})
    return receipt


def load_spec(path):
    """Bounded, unambiguous researcher JSON: never load an arbitrary-size file."""
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16000:
        raise ValueError('QUEUE_INPUT_SIZE_INVALID')
    with path.open('rb') as stream:
        raw = stream.read(16001)
    if len(raw) > 16000:
        raise ValueError('QUEUE_INPUT_SIZE_INVALID')

    def unique_object(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError('QUEUE_DUPLICATE_JSON_KEY')
            obj[key] = value
        return obj

    return validate_spec(json.loads(raw, object_pairs_hook=unique_object))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', required=True, type=Path)
    parser.add_argument('--input', required=True, type=Path)
    args = parser.parse_args()
    try:
        result = run_queue(load_spec(args.input), args.sources)
    except (ValueError, RuntimeError, OSError, subprocess.TimeoutExpired) as error:
        print(json.dumps({'status': 'BLOCKED', 'reason': str(error)}))
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
