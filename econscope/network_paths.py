"""Pinned NetworkX delivery connectivity. No code intake, installation or network.

Supplier -> customer edges carry positive integer model-tick delivery delays.
Minimum additive path delay is a connectivity measure, not shock transmission,
input-output multipliers, empirical identification or a calibrated forecast.
"""
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tempfile

COMMIT = '9d866a5f447d9fe1c989b31f90c520dc2813ee26'
ARCHIVE_NAME = 'NetworkX-9d866a5f447d.tar.gz'
ARCHIVE_SHA256 = '7d2c901b1928e1d5f4fc14e78c058f453f3570a2079c1b62f04818a74d351023'
ROOT = 'networkx-' + COMMIT
VERSION = '3.7.1rc0.dev0'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path, limit):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError('NETWORK_FILE_INVALID')
    with path.open('rb') as stream:
        raw = stream.read(limit+1)
    if len(raw) > limit:
        raise ValueError('NETWORK_FILE_BUDGET')
    return raw


def parse(raw):
    def pairs(items):
        result = {}
        for k, v in items:
            if k in result:
                raise ValueError('NETWORK_DUPLICATE_KEY')
            result[k] = v
        return result
    def invalid(_):
        raise ValueError('NETWORK_NONFINITE')
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def validate(parameters):
    if sys.version_info < (3, 12):
        raise RuntimeError('NETWORK_PYTHON_312_REQUIRED')
    basic = {'nodes', 'links', 'origin'}
    scenario = basic | {'disrupted_node', 'demand_units_per_day'}
    if type(parameters) is not dict or set(parameters) not in (basic, scenario):
        raise ValueError('NETWORK_SCHEMA_INVALID')
    nodes, links = parameters['nodes'], parameters['links']
    if (type(nodes) is not list or not 1 <= len(nodes) <= 32 or
            any(type(n) is not str or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,31}', n) for n in nodes) or
            len(set(nodes)) != len(nodes) or type(parameters['origin']) is not str or
            parameters['origin'] not in nodes or type(links) is not list or len(links) > 128):
        raise ValueError('NETWORK_NODES_OR_BUDGET_INVALID')
    seen = set()
    for link in links:
        if (type(link) is not list or len(link) != 3 or
                any(type(n) is not str or n not in nodes for n in link[:2]) or
                link[0] == link[1] or type(link[2]) is not int or not 1 <= link[2] <= 1_000_000):
            raise ValueError('NETWORK_LINK_INVALID')
        pair = tuple(link[:2])
        if pair in seen:
            raise ValueError('NETWORK_DUPLICATE_LINK')
        seen.add(pair)
    if 'disrupted_node' in parameters:
        demand = parameters['demand_units_per_day']
        if (type(parameters['disrupted_node']) is not str or parameters['disrupted_node'] not in nodes or
                type(demand) is not dict or set(demand) != set(nodes) or
                any(type(v) is not int or not 0 <= v <= 1_000_000 for v in demand.values())):
            raise ValueError('NETWORK_DISRUPTION_OR_DEMAND_INVALID')
    return parameters


def oracle(parameters):
    """Independent Bellman-Ford relaxation; never imports NetworkX/Dijkstra."""
    validate(parameters)
    distances = dict.fromkeys(parameters['nodes'])
    distances[parameters['origin']] = 0
    for _ in range(len(distances)-1):
        previous = dict(distances)
        for supplier, customer, delay in parameters['links']:
            if previous[supplier] is not None:
                candidate = previous[supplier] + delay
                if distances[customer] is None or candidate < distances[customer]:
                    distances[customer] = candidate
    return distances


def disruption_summary(parameters, baseline, disrupted):
    """Conditional access/delay accounting; never a claim of realized output loss.

    One model tick is one declared day in this optional scenario. Demand must use
    one common unit/day. Only previously reachable demand can become newly
    unreachable. Delay burden excludes unreachable nodes (reported separately).
    """
    demand = parameters['demand_units_per_day']
    newly = [n for n in parameters['nodes'] if baseline[n] is not None and disrupted[n] is None]
    return {'baseline_days': baseline, 'disrupted_days': disrupted,
            'newly_unreachable_nodes': newly,
            'newly_unreachable_demand_units_per_day': sum(demand[n] for n in newly),
            'previously_unreachable_demand_units_per_day': sum(demand[n] for n in demand if baseline[n] is None),
            'additional_unit_days_per_day': sum(demand[n] * (disrupted[n]-baseline[n])
                for n in demand if baseline[n] is not None and disrupted[n] is not None)}


def expected_result(parameters):
    baseline = oracle(parameters)
    if 'disrupted_node' not in parameters:
        return baseline
    removed = parameters['disrupted_node']
    # The independent recurrence retains the removed node as unreachable, removes
    # all its edges and never seeds it when the supply origin itself closes.
    remaining = {'nodes': parameters['nodes'], 'origin': parameters['origin'],
                 'links': [e for e in parameters['links'] if removed not in e[:2]]}
    disrupted = (dict.fromkeys(parameters['nodes']) if removed == parameters['origin']
                 else oracle(remaining))
    disrupted[removed] = None
    return disruption_summary(parameters, baseline, disrupted)


def check_result(parameters, actual):
    if actual != expected_result(parameters):
        raise ValueError('NETWORK_ORACLE_MISMATCH')


def audit_archive(path):
    import tomllib
    raw = read(path, 3_000_000)
    if digest(raw) != ARCHIVE_SHA256:
        raise ValueError('NETWORK_ARCHIVE_DIGEST_MISMATCH')
    files, seen, total = {}, set(), 0
    with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as archive:
        for index, member in enumerate(archive):
            name = member.name
            parts = name.split('/')
            if (index >= 1200 or parts[0] != ROOT or len(name) > 240 or '\\' in name or
                    any(not p or p in ('.', '..') or ':' in p or p.endswith((' ', '.')) or
                        any(ord(c) < 32 or c in '<>"|?*' for c in p) or
                        p.split('.')[0].upper() in {'CON', 'PRN', 'AUX', 'NUL'} |
                        {f'{q}{i}' for q in ('COM', 'LPT') for i in range(1, 10)} for p in parts)):
                raise ValueError('NETWORK_ARCHIVE_PATH_INVALID')
            if name.casefold() in seen:
                raise ValueError('NETWORK_ARCHIVE_DUPLICATE')
            seen.add(name.casefold())
            # The pinned benchmark-only symlink is inspected but never followed,
            # extracted or executed. Every other link or special file is denied.
            if member.issym() and name == ROOT+'/benchmarks/pyproject.toml' and member.linkname == '../pyproject.toml\n':
                continue
            if not (member.isfile() or member.isdir()):
                raise ValueError('NETWORK_ARCHIVE_LINK_INVALID')
            total += member.size
            if member.size < 0 or member.size > 2_000_000 or total > 12_000_000:
                raise ValueError('NETWORK_ARCHIVE_SIZE_INVALID')
            relative = name[len(ROOT)+1:]
            # This fixed worker executes runtime APIs only. Keep the complete
            # archive/path/license audit above, but do not stage upstream tests
            # that none of the reviewed delivery/flow operations imports.
            selected = relative in ('LICENSE.txt', 'pyproject.toml') or (
                relative.startswith('networkx/') and relative.endswith('.py') and '/tests/' not in relative)
            if member.isfile() and selected:
                files[relative] = archive.extractfile(member).read()
    license_text = files['LICENSE.txt'].decode('utf-8')
    if ('3-clause BSD license' not in license_text or
            'Redistributions of source code must retain' not in license_text or
            'Neither the name of the NetworkX Developers' not in license_text):
        raise ValueError('NETWORK_LICENSE_INVALID')
    project = tomllib.loads(files['pyproject.toml'].decode())['project']
    if project['dependencies'] != [] or project['license'] != 'BSD-3-Clause':
        raise ValueError('NETWORK_RUNTIME_DEPENDENCIES_UNREVIEWED')
    return files


def verify_prepared(root, files):
    root = Path(root)
    actual = {}
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ValueError('NETWORK_PREPARED_LINK')
        if path.is_file():
            actual[path.relative_to(root).as_posix()] = digest(read(path, 2_000_000))
    if actual != {k: digest(v) for k, v in files.items()}:
        raise ValueError('NETWORK_PREPARED_TAMPER')


def run(parameters, sources, *, calculation='delivery'):
    from .execution_control import remaining_timeout, safe_local_path, run_worker
    if calculation not in ('delivery', 'capacity', 'cost_flow'):
        raise ValueError('NETWORK_CALCULATION_INVALID')
    capacity = calculation == 'capacity'
    cost_flow = calculation == 'cost_flow'
    if cost_flow:
        from .network_cost_flow import validate as validate_cost_flow, check_result as check_cost_flow
        validate_cost_flow(parameters)
    elif capacity:
        from .network_capacity import validate as validate_capacity, check_result as check_capacity, cut_oracle
        validate_capacity(parameters)
    else:
        validate(parameters)
    if sources is None:
        raise ValueError('ENGINE_STUDY_PINNED_SOURCES_REQUIRED')
    sources = safe_local_path(sources)
    safe_local_path(sources/'manifest.json'); safe_local_path(sources/ARCHIVE_NAME)
    manifest_raw = read(sources/'manifest.json', 256*1024)
    manifest = parse(manifest_raw)
    rows = manifest.get('archives') if type(manifest) is dict else None
    if type(rows) is not list or len(rows) > 256 or any(type(row) is not dict for row in rows):
        raise ValueError('NETWORK_MANIFEST_INVALID')
    rows = [row for row in rows if row.get('name') == 'NetworkX']
    expected = {'commit': COMMIT, 'file': ARCHIVE_NAME, 'sha256': ARCHIVE_SHA256,
                'origin_url': 'https://github.com/networkx/networkx', 'bytes': 2783502,
                'license_reference_hint': 'BSD-3-Clause', 'license_spdx_from_github': 'NOASSERTION'}
    if len(rows) != 1 or any(rows[0].get(k) != v for k, v in expected.items()):
        raise ValueError('NETWORK_MANIFEST_PIN_MISMATCH')
    archive = (sources/ARCHIVE_NAME).resolve()
    files = audit_archive(sources/ARCHIVE_NAME)
    worker = Path(__file__).with_name('network_paths_worker.py').resolve()
    with tempfile.TemporaryDirectory(prefix='societybox-network-') as temp:
        root = Path(temp)
        for name, raw in files.items():
            path = root/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        verify_prepared(root, files)
        try:
            child = run_worker([sys.executable, '-I', '-S', '-B', str(worker), str(root), str(archive), calculation],
                                   input=json.dumps(parameters).encode(), capture_output=True, timeout=remaining_timeout(20),
                                   cwd=root, env={'SYSTEMROOT': os.environ.get('SYSTEMROOT', '')},
                                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except subprocess.TimeoutExpired as error:
            raise RuntimeError('NETWORK_WORKER_TIMEOUT_REAPED') from error
        if child.returncode or len(child.stdout) > 256*1024:
            raise RuntimeError('NETWORK_FIXED_WORKER_FAILED')
        raw = parse(child.stdout)
        if cost_flow:
            upstream_keys = {'version', 'api', 'flow_algorithm', 'additional_apis', 'import_file_sha256',
                             'elevated', 'network_audit_canary_denied', 'python_version', 'runtime_dependencies', 'isolation'}
            if (type(raw) is not dict or set(raw) != {'actual', 'upstream'} or
                    type(raw['upstream']) is not dict or set(raw['upstream']) != upstream_keys or
                    type(raw['upstream']['import_file_sha256']) is not dict):
                raise ValueError('NETWORK_UPSTREAM_SCHEMA_INVALID')
        verify_prepared(root, files)
        upstream = raw['upstream']
        imports = upstream['import_file_sha256']
        if cost_flow:
            cost_certificate = check_cost_flow(parameters, raw['actual'])
        else:
            (check_capacity if capacity else check_result)(parameters, raw['actual'])
        expected_api = 'networkx.network_simplex' if cost_flow else 'networkx.maximum_flow' if capacity else 'networkx.single_source_dijkstra_path_length'
        expected_module = ('networkx/algorithms/flow/networksimplex.py' if cost_flow else 'networkx/algorithms/flow/edmondskarp.py' if capacity else
                           'networkx/algorithms/shortest_paths/weighted.py')
        if (upstream['version'] != VERSION or
                upstream['elevated'] is not False or upstream['network_audit_canary_denied'] is not True or
                upstream['api'] != expected_api or
                not imports or any(name not in files or digest(files[name]) != sha for name, sha in imports.items()) or
                expected_module not in imports or
                (capacity and upstream.get('flow_algorithm') != 'networkx.algorithms.flow.edmonds_karp')):
            raise ValueError('NETWORK_ORACLE_OR_PROVENANCE_MISMATCH')
        if cost_flow:
            expected_additional = ['networkx.cost_of_flow'] if raw['actual']['status'] == 'OPTIMAL' else []
            if (upstream.get('additional_apis') != expected_additional or
                    upstream.get('flow_algorithm') != 'networkx.network_simplex' or
                    upstream.get('python_version') != sys.version or upstream.get('runtime_dependencies') != [] or
                    upstream.get('isolation') != 'TRUST_PINNED_FIXED_WORKER_ONLY_NOT_OS_SANDBOX' or
                    (expected_additional and 'networkx/algorithms/flow/mincost.py' not in imports)):
                raise ValueError('NETWORK_ORACLE_OR_PROVENANCE_MISMATCH')
    result = {'engine': 'NETWORKX_PINNED_DELIVERY_PATHS', 'mode': 'NO_LLM', 'status': 'REFERENCE_EXECUTED',
            'native_execution': 'EXECUTED', 'native_parity': 'EXACT_MATCH_BELLMAN_FORD_ORACLE',
            'actual': raw['actual'], 'upstream': upstream,
            'archive_sha256': ARCHIVE_SHA256, 'commit': COMMIT, 'license': 'BSD-3-Clause',
            'license_sha256': digest(files['LICENSE.txt']), 'manifest_sha256': digest(manifest_raw),
            'worker_sha256': digest(worker.read_bytes()), 'child_exit': child.returncode,
            'prepared_source_sha256': digest(json.dumps({k: digest(v) for k, v in files.items()}, sort_keys=True).encode()),
            'os_isolation': 'NOT_ATTESTED', 'source_files_unchanged': True,
            'temporary_source_removed_on_return': True,
            'assumptions': ['supplier to customer; positive additive delay in model ticks',
                            'duplicate links, self links and undeclared nodes rejected; isolated nodes retained as null',
                            'shortest connectivity delay only; no shock magnitude or causal identification'],
            'scenario_convention': ('Closure removes all incoming/outgoing delivery edges; positive days; '
                                    'cycles allowed; no negative weights or dynamic stability inference. '
                                    'Common demand units/day; unreachable demand is access at risk, not realized loss. '
                                    'Finite delay burden excludes unreachable customers; no capacity or substitution model.'),
            'limits': {'nodes': 32, 'links': 128, 'edge_delay': 1000000, 'child_seconds': 20},
            'redistribution': 'NOT_AUTHORIZED; source retained only in local reference archive'}
    if capacity:
        result.update(engine='NETWORKX_PINNED_SUPPLY_CAPACITY',
                      native_parity='EXACT_MATCH_EXHAUSTIVE_CUT_ORACLE', oracle=cut_oracle(parameters),
                      assumptions=['one divisible commodity in common goods units/day',
                                   'steady flow; conservation at intermediate nodes; nonnegative integer capacities',
                                   'maximum throughput only; no demand, prices, storage, substitution or causal inference'],
                      scenario_convention='Researcher-supplied capacities; flow vectors may be nonunique.',
                      limits={'nodes': 12, 'links': 64, 'capacity': 1000000,
                              'oracle_cuts': 1024, 'child_seconds': 20})
    if cost_flow:
        result.update(engine='NETWORKX_PINNED_MINIMUM_COST_FLOW',
                      native_parity='EXACT_FEASIBILITY_AND_PRIMAL_DUAL_CERTIFICATE',oracle=cost_certificate,
                      assumptions=['One commodity; explicit common flow/cost units and fixed declared net demands',
                                   'Positive demand receives flow; negative demand supplies flow; all capacities finite',
                                   'Signed unit costs, bounded negative-cost cycles and self-loops are allowed',
                                   'Minimum declared static flow/circulation cost, not welfare, equilibrium, causality or forecast'],
                      scenario_convention='Duplicate directed pairs rejected; flow vectors may be nonunique; compare feasibility and optimum cost.',
                      limits={'nodes': 12, 'links': 64, 'capacity': 1000000, 'absolute_cost': 1000000,
                              'absolute_demand': 1000000, 'oracle_cuts': 4096, 'child_seconds': 20})
    return result
