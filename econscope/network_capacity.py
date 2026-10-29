"""Fixed one-commodity directed supply capacity, not calibrated economic losses.

Nonnegative integer goods/day, steady flow with conservation, no storage, demand
curve, cost or substitution. Native flow vectors need not be unique. The oracle
enumerates every source/sink cut independently of the upstream augmenting paths.
"""
import re


def validate(parameters):
    if type(parameters) is not dict or set(parameters) != {'nodes', 'links', 'origin', 'sink'}:
        raise ValueError('CAPACITY_SCHEMA_INVALID')
    nodes, links = parameters['nodes'], parameters['links']
    if (type(nodes) is not list or not 2 <= len(nodes) <= 12 or
            any(type(n) is not str or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,31}', n) for n in nodes) or
            len(set(nodes)) != len(nodes) or
            any(type(parameters[k]) is not str or parameters[k] not in nodes for k in ('origin', 'sink')) or
            parameters['origin'] == parameters['sink'] or type(links) is not list or len(links) > 64):
        raise ValueError('CAPACITY_NODES_OR_BUDGET_INVALID')
    seen = set()
    for link in links:
        if (type(link) is not list or len(link) != 3 or
                any(type(n) is not str or n not in nodes for n in link[:2]) or
                link[0] == link[1] or type(link[2]) is not int or not 0 <= link[2] <= 1000000):
            raise ValueError('CAPACITY_LINK_INVALID')
        pair = tuple(link[:2])
        if pair in seen:
            raise ValueError('CAPACITY_DUPLICATE_LINK')
        seen.add(pair)
    return parameters


def cut_oracle(parameters):
    validate(parameters)
    middle = [n for n in parameters['nodes'] if n not in (parameters['origin'], parameters['sink'])]
    best, partition = None, None
    for mask in range(1 << len(middle)):
        left = {parameters['origin']} | {n for i, n in enumerate(middle) if mask & (1 << i)}
        capacity = sum(c for a, b, c in parameters['links'] if a in left and b not in left)
        if best is None or capacity < best:
            best, partition = capacity, sorted(left)
    return {'min_cut_capacity': best, 'source_side': partition,
            'cuts_enumerated': 1 << len(middle), 'method': 'EXHAUSTIVE_DIRECTED_CUT_SUM'}


def check_result(parameters, actual):
    validate(parameters)
    if (type(actual) is not dict or set(actual) != {'flow_value', 'edge_flows'} or
            type(actual['flow_value']) is not int or actual['flow_value'] < 0 or
            type(actual['edge_flows']) is not list or len(actual['edge_flows']) != len(parameters['links'])):
        raise ValueError('CAPACITY_RESULT_INVALID')
    balance = dict.fromkeys(parameters['nodes'], 0)
    for declared, row in zip(parameters['links'], actual['edge_flows']):
        if (type(row) is not list or len(row) != 3 or row[:2] != declared[:2] or
                type(row[2]) is not int or not 0 <= row[2] <= declared[2]):
            raise ValueError('CAPACITY_EDGE_FLOW_INVALID')
        a, b, flow = row; balance[a] -= flow; balance[b] += flow
    expected = dict.fromkeys(parameters['nodes'], 0)
    expected[parameters['origin']] = -actual['flow_value']
    expected[parameters['sink']] = actual['flow_value']
    if balance != expected or actual['flow_value'] != cut_oracle(parameters)['min_cut_capacity']:
        raise ValueError('CAPACITY_CONSERVATION_OR_CUT_MISMATCH')


def run(parameters, sources):
    from .network_paths import run as run_pinned
    validate(parameters)
    return run_pinned(parameters, sources, calculation='capacity')
