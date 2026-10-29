"""Fixed reviewed NetworkX worker; JSON data only. Not an OS sandbox."""
import ctypes
import json
import os
from pathlib import Path
import socket
import sys


def main():
    elevated = bool(ctypes.windll.shell32.IsUserAnAdmin()) if os.name == 'nt' else os.geteuid() == 0
    if elevated:
        raise RuntimeError('NONADMIN_PROCESS_REQUIRED')
    def guard(event, args):
        if event.startswith(('socket.', 'subprocess.', 'os.system', 'os.spawn', 'os.exec', 'ctypes.dlopen')):
            raise RuntimeError('NETWORK_OR_EXTERNAL_PROCESS_DENIED')
    sys.addaudithook(guard)
    denied = False
    try:
        socket.socket()
    except RuntimeError:
        denied = True
    if not denied:
        raise RuntimeError('NETWORK_CANARY_FAILED')
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from network_paths import parse, validate, audit_archive, verify_prepared, digest, disruption_summary
    raw = sys.stdin.buffer.read(65537)
    if len(raw) > 65536:
        raise ValueError('NETWORK_INPUT_BUDGET')
    mode = sys.argv[3]
    if mode not in ('delivery', 'capacity', 'cost_flow'):
        raise ValueError('NETWORK_CALCULATION_INVALID')
    capacity = mode == 'capacity'
    cost_flow = mode == 'cost_flow'
    if cost_flow:
        from network_cost_flow import validate as validate_cost_flow
        parameters = validate_cost_flow(parse(raw))
    elif capacity:
        from network_capacity import validate as validate_capacity
        parameters = validate_capacity(parse(raw))
    else:
        parameters = validate(parse(raw))
    root = Path(sys.argv[1]).resolve(strict=True)
    files = audit_archive(sys.argv[2])
    verify_prepared(root, files)
    sys.path.insert(0, str(root))
    import networkx as nx
    graph = nx.DiGraph()
    graph.add_nodes_from(parameters['nodes'])
    additional_apis = []
    if cost_flow:
        for node in parameters['nodes']:
            graph.nodes[node]['demand'] = parameters['demands'][node]
        graph.add_edges_from((a, b, {'capacity': cap, 'weight': cost}) for a, b, cap, cost in parameters['links'])
        try:
            total, flows = nx.network_simplex(graph, demand='demand', capacity='capacity', weight='weight', backend='networkx')
            recomputed = nx.cost_of_flow(graph, flows, weight='weight', backend='networkx')
            additional_apis = ['networkx.cost_of_flow']
            actual = {'status': 'OPTIMAL', 'cost': total, 'recomputed_cost': recomputed,
                      'edge_flows': [[a, b, flows[a][b]] for a, b, _, _ in parameters['links']]}
        except nx.NetworkXUnfeasible:
            actual = {'status': 'INFEASIBLE', 'cost': None, 'recomputed_cost': None, 'edge_flows': None}
    elif capacity:
        graph.add_weighted_edges_from(parameters['links'], weight='capacity')
        value, flows = nx.maximum_flow(graph, parameters['origin'], parameters['sink'],
                                      capacity='capacity', flow_func=nx.algorithms.flow.edmonds_karp,
                                      backend='networkx')
        actual = {'flow_value': value,
                  'edge_flows': [[a, b, flows[a][b]] for a, b, _ in parameters['links']]}
    else:
        graph.add_weighted_edges_from(parameters['links'], weight='delay')
        lengths = nx.single_source_dijkstra_path_length(graph, parameters['origin'], weight='delay', backend='networkx')
        actual = {node: lengths.get(node) for node in parameters['nodes']}
    if 'disrupted_node' in parameters:
        removed = parameters['disrupted_node']
        graph.remove_node(removed)
        after = ({} if removed == parameters['origin'] else
                 nx.single_source_dijkstra_path_length(graph, parameters['origin'], weight='delay', backend='networkx'))
        disrupted = {node: after.get(node) for node in parameters['nodes']}
        actual = disruption_summary(parameters, actual, disrupted)
    imports = {}
    for name, module in list(sys.modules.items()):
        if name == 'networkx' or name.startswith('networkx.'):
            path = Path(module.__file__).resolve(strict=True)
            relative = path.relative_to(root).as_posix()
            if relative not in files or path.read_bytes() != files[relative]:
                raise ValueError('NETWORK_IMPORT_TAMPER')
            imports[relative] = digest(files[relative])
    print(json.dumps({'actual': actual, 'upstream': {'version': nx.__version__,
          'api': 'networkx.network_simplex' if cost_flow else 'networkx.maximum_flow' if capacity else 'networkx.single_source_dijkstra_path_length',
          'flow_algorithm': 'networkx.network_simplex' if cost_flow else 'networkx.algorithms.flow.edmonds_karp' if capacity else None,
          'additional_apis': additional_apis,
          'import_file_sha256': imports,
          'elevated': elevated, 'network_audit_canary_denied': denied,
          'python_version': sys.version, 'runtime_dependencies': [],
          'isolation': 'TRUST_PINNED_FIXED_WORKER_ONLY_NOT_OS_SANDBOX'}}))


if __name__ == '__main__':
    main()
