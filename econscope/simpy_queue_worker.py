"""Fixed subprocess worker. Never accepts code, commands or import names from input."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import socket
import sys


def main():
    elevated = bool(ctypes.windll.shell32.IsUserAnAdmin()) if os.name == 'nt' else os.geteuid() == 0
    if elevated:
        raise RuntimeError('NONADMIN_PROCESS_REQUIRED')
    root = Path(sys.argv[1]).resolve(strict=True)
    # Defense in depth for this immutable trusted module slice. Not an OS sandbox.
    denied = ('socket.', 'subprocess.', 'os.system', 'os.spawn', 'os.exec', 'ctypes.dlopen')

    def guard(event, args):
        if event.startswith(denied):
            raise RuntimeError('NETWORK_OR_EXTERNAL_PROCESS_DENIED')

    sys.addaudithook(guard)
    network_guard = False
    try:
        socket.socket()
    except RuntimeError:
        network_guard = True
    if not network_guard:
        raise RuntimeError('NETWORK_GUARD_CANARY_FAILED')
    spec = json.loads(sys.stdin.read(16001))
    # Validate before importing upstream even when invoked directly.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from simpy_queue import validate_spec, audit_archive, verify_prepared_sources
    validate_spec(spec)
    files, _ = audit_archive(Path(sys.argv[2]))
    verify_prepared_sources(root, files)
    sys.path.insert(0, str(root))
    import simpy

    imports = {}
    for name, module in list(sys.modules.items()):
        if name == 'simpy' or name.startswith('simpy.'):
            path = Path(module.__file__).resolve(strict=True)
            relative = path.relative_to(root).as_posix()
            imports[relative] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    env = simpy.Environment()
    server = simpy.Resource(env, capacity=spec.get('server_count', 1))
    jobs, lost, states = [], [], []
    completed = 0

    def customer(i, arrival, service):
        nonlocal completed
        yield env.timeout(arrival)
        if server.count + len(server.queue) >= spec['system_capacity']:
            lost.append(i)
            return
        with server.request() as request:
            yield request
            start = env.now
            job = {'id': i, 'arrival': arrival, 'start': start, 'wait': start-arrival}
            jobs.append(job)
            yield env.timeout(service)
            job['finish'] = env.now
            completed += 1

    for i, (arrival, service) in enumerate(zip(spec['arrivals'], spec['service_times'])):
        env.process(customer(i, arrival, service))
    while env.peek() != float('inf'):
        time = env.peek()
        while env.peek() == time:
            env.step()
        # Actual Resource occupancy, not reconstructed from the oracle intervals.
        states.append({'time': time, 'busy': server.count, 'queued': len(server.queue),
                       'completed': completed, 'lost': len(lost)})
    # Initialization at t=0 is not an economic event if first arrival is later.
    states = [s for s in states if s['time'] >= spec['arrivals'][0]]
    actual = {'jobs': jobs, 'lost_jobs': lost, 'states': states,
              'waiting_cost': sum(j['wait'] for j in jobs)*spec['waiting_cost_per_tick'],
              'busy_ticks': sum(j['finish']-j['start'] for j in jobs)}
    print(json.dumps({'actual': actual, 'upstream': {'version': simpy.__version__,
        'import_files': imports, 'interpreter': sys.executable, 'python_version': sys.version,
        'elevated': elevated, 'network_audit_canary_denied': network_guard,
        'network_canary_operation': 'socket.socket',
        'network_policy': 'Python audit denies socket operations before upstream import',
        'isolation': 'TRUST_PINNED_FIXED_WORKER_ONLY_NOT_OS_SANDBOX'}}))


if __name__ == '__main__':
    main()
