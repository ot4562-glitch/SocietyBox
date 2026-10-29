"""Offline finite-workload staffing comparison using pinned MIT SimPy.

Fixed FIFO models only. Integer costs share one researcher-chosen currency unit.
Staffing cost is per server for the whole workload, not an hourly wage.
No fitting, causal identification, stochastic inference or generated code intake.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

from .simpy_queue import run_queue, validate_spec
from .strict_json import loads_strict


def validate_study(spec):
    keys = {'queue', 'server_counts', 'staffing_cost_per_server', 'rejected_job_cost'}
    if type(spec) is not dict or set(spec) != keys:
        raise ValueError('STAFFING_SCHEMA_INVALID')
    base = spec['queue']
    if type(base) is not dict or 'server_count' in base:
        raise ValueError('STAFFING_QUEUE_SCHEMA_INVALID')
    validate_spec(base)
    counts = spec['server_counts']
    if (type(counts) is not list or not 1 <= len(counts) <= 8 or
            any(type(n) is not int for n in counts) or counts != sorted(set(counts))):
        raise ValueError('STAFFING_CANDIDATES_INVALID')
    for n in counts:
        validate_spec(dict(base, server_count=n))
    for key in ('staffing_cost_per_server', 'rejected_job_cost'):
        if type(spec[key]) is not int or not 0 <= spec[key] <= 1_000_000:
            raise ValueError('STAFFING_COST_INVALID')
    return spec


def _load(path, limit):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError('STAFFING_FILE_SIZE_OR_TYPE_INVALID')
    with path.open('rb') as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError('STAFFING_FILE_SIZE_OR_TYPE_INVALID')
    return loads_strict(raw)


def load_study(path):
    return validate_study(_load(path, 16000))


def run_study(spec, sources):
    validate_study(spec)
    runs, policies = [], []
    for servers in spec['server_counts']:
        run = run_queue(dict(spec['queue'], server_count=servers), sources)
        actual = run['actual']
        staffing_cost = servers * spec['staffing_cost_per_server']
        rejection_cost = len(actual['lost_jobs']) * spec['rejected_job_cost']
        policies.append({'server_count': servers, 'completed_jobs': len(actual['jobs']),
                         'rejected_jobs': len(actual['lost_jobs']),
                         'total_wait_ticks': sum(j['wait'] for j in actual['jobs']),
                         'waiting_cost': actual['waiting_cost'], 'staffing_cost': staffing_cost,
                         'rejection_cost': rejection_cost,
                         'total_cost': actual['waiting_cost'] + staffing_cost + rejection_cost})
        runs.append(run)
    minimum = min(p['total_cost'] for p in policies)
    winners = [p['server_count'] for p in policies if p['total_cost'] == minimum]
    return {'schema': 'societybox-simpy-staffing-v1', 'status': 'ENGINEERING_CANDIDATE',
            'proof': 'REFERENCE_EXECUTED', 'spec': spec, 'policies': policies, 'runs': runs,
            'minimizing_server_counts': winners, 'selected_server_count': winners[0],
            'selection_rule': 'Minimum total cost; ties choose fewest servers.',
            'limits': ['Conditional finite deterministic workload; no empirical validation.',
                       'System capacity includes service and waiting positions.',
                       'Arrivals at a time precede completions at that time.',
                       'Staffing cost per server covers the entire workload.',
                       'Only supplied staffing alternatives evaluated; no uncertainty estimate.',
                       'Pinned fixed worker and audit guards are not general OS isolation.',
                       'No upstream source redistribution or arbitrary code execution.']}


def _json(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + '\n'


def _csv(rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _artifacts(result):
    jobs = [dict(server_count=r['spec']['server_count'], **job)
            for r in result['runs'] for job in r['actual']['jobs']]
    policies = result['policies']
    scale = max(1, max(p['total_cost'] for p in policies))
    bars = ''.join(f'<text x="5" y="{35+i*35}">{p["server_count"]} servers</text>'
                   f'<rect x="110" y="{20+i*35}" width="{400*p["total_cost"]/scale:.3f}" '
                   f'height="20" fill="#285b80"/>'
                   f'<text x="520" y="{35+i*35}">{p["total_cost"]}</text>'
                   for i, p in enumerate(policies))
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="650" height="{50+35*len(policies)}" '
           'role="img"><title>Total conditional cost by staffing alternative</title>' + bars + '</svg>')
    selected = result['selected_server_count']
    html = ('<!doctype html><html lang="en"><meta charset="utf-8">'
            '<title>Conditional staffing comparison</title><h1>Conditional staffing comparison</h1>'
            '<p>Evidence: supplied finite workload and assumed integer costs. '
            'Pinned MIT SimPy execution matched an independent interval oracle exactly.</p>'
            '<p>Total cost = waiting ticks * waiting cost per tick + rejected jobs * rejection cost '
            '+ servers * staffing cost per server for the whole workload.</p>'
            f'<p>Selected servers among supplied alternatives: {selected}. Ties choose fewest servers.</p>'
            + svg + '<p>See input.json for assumptions, policies.csv and jobs.csv for numeric results, '
            'and result.json for source, license, worker and manifest provenance.</p><ul>'
            + ''.join('<li>' + text + '</li>' for text in result['limits']) + '</ul></html>')
    return {'input.json': _json(result['spec']), 'policies.csv': _csv(policies),
            'jobs.csv': _csv(jobs), 'costs.svg': svg, 'report.html': html}


def export_bundle(result, destination):
    # Exclusive directory creation preserves all existing research work.
    destination = Path(destination)
    artifacts = _artifacts(result)
    record = dict(result, artifact_sha256={name: hashlib.sha256(body.encode('utf-8')).hexdigest()
                                         for name, body in artifacts.items()})
    destination.mkdir(parents=True, exist_ok=False)
    for name, body in dict(artifacts, **{'result.json': _json(record)}).items():
        with (destination / name).open('x', encoding='utf-8', newline='') as stream:
            stream.write(body)


def _stable(result):
    """Compare results and durable provenance, excluding disposable runtime paths."""
    return {key: result[key] for key in ('schema', 'status', 'proof', 'spec', 'policies',
            'minimizing_server_counts', 'selected_server_count', 'selection_rule', 'limits')} | {
        'runs': [{'spec': r['spec'], 'actual': r['actual'], 'comparison': r['comparison'],
                  'upstream': {key: r['upstream'][key] for key in (
                      'version', 'archive_sha256', 'commit', 'license_sha256',
                      'elevated', 'network_audit_canary_denied', 'setup_executed', 'redistribution')},
                  'import_sha256': {name: item['sha256'] for name, item in
                                    r['upstream']['import_files'].items()},
                  'child_exit': r['child_exit'],
                  'manifest_sha256': r['manifest_sha256'], 'worker_sha256': r['worker_sha256'],
                  'prepared_file_sha256': r['prepared_file_sha256']} for r in result['runs']]}


def verify_bundle(destination, sources):
    destination = Path(destination)
    # Refuse incomplete or augmented bundles before starting expensive upstream workers.
    names = {'input.json', 'result.json', 'policies.csv', 'jobs.csv', 'costs.svg', 'report.html'}
    if (destination.is_symlink() or not destination.is_dir() or
            {path.name for path in destination.iterdir()} != names):
        raise ValueError('BUNDLE_ARTIFACT_MISMATCH')
    saved = _load(destination / 'result.json', 2_000_000)
    declared = saved.get('artifact_sha256') if type(saved) is dict else None
    if type(declared) is not dict or set(declared) != names - {'result.json'}:
        raise ValueError('BUNDLE_ARTIFACT_MISMATCH')
    # This is only a fast-fail check against the saved manifest. A forged manifest
    # cannot pass: the full native rerun and independent rendering still follow.
    for name, expected_hash in declared.items():
        path = destination / name
        if path.is_symlink() or not path.is_file():
            raise ValueError('BUNDLE_ARTIFACT_MISMATCH')
        with path.open('rb') as stream:
            raw = stream.read(2_000_001)
        if len(raw) > 2_000_000 or hashlib.sha256(raw).hexdigest() != expected_hash:
            raise ValueError('BUNDLE_ARTIFACT_MISMATCH')
    fresh = run_study(load_study(destination / 'input.json'), sources)
    try:
        matches = _stable(saved) == _stable(fresh)
    except (KeyError, TypeError):
        matches = False
    if not matches:
        raise ValueError('BUNDLE_RECOMPUTATION_MISMATCH')
    # Compare against freshly rendered artifacts, never trust self-supplied hashes.
    expected = _artifacts(fresh)
    hashes = {name: hashlib.sha256(body.encode('utf-8')).hexdigest() for name, body in expected.items()}
    if saved.get('artifact_sha256') != hashes:
        raise ValueError('BUNDLE_ARTIFACT_MISMATCH')
    for name, body in expected.items():
        path = destination / name
        if (path.is_symlink() or not path.is_file() or path.stat().st_size != len(body.encode('utf-8'))
                or path.read_bytes() != body.encode('utf-8')):
            raise ValueError('BUNDLE_ARTIFACT_MISMATCH')
    # The bundle must still have the same closed file set after native execution.
    if {path.name for path in destination.iterdir()} != names:
        raise ValueError('BUNDLE_ARTIFACT_MISMATCH')
    return {'comparison': 'RECOMPUTED_EXACT_MATCH',
            'selected_server_count': fresh['selected_server_count'],
            'scope': 'Same local source tree; not independent clean-install replication.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--input', type=Path)
    mode.add_argument('--verify', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        if args.verify:
            result = verify_bundle(args.verify, args.sources)
        else:
            if args.output is None:
                raise ValueError('STAFFING_OUTPUT_REQUIRED')
            result = run_study(load_study(args.input), args.sources)
            export_bundle(result, args.output)
            result = {key: result[key] for key in ('status', 'proof', 'policies', 'selected_server_count')}
        print(_json(result), end='')
        return 0
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as error:
        print(_json({'status': 'BLOCKED', 'reason': str(error)}), end='')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
