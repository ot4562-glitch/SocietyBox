"""Bounded local JSON transport for synthetic ledger/proposal/math tools; no writes."""
import json
import math
from pathlib import Path

from .mechanism_vintage import review_ledger, as_of_snapshot
from .workspace import ScopeError

MAX_INPUT_BYTES = 2 * 1024 * 1024  # Per file, including UTF-8 encoding.
MAX_OUTPUT_BYTES = 4 * 1024 * 1024  # Complete ASCII JSON envelope, before newline.
MAX_DEPTH = 32
MAX_NODES = 100000


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate key')
        result[key] = value
    return result


def _constant(value):
    raise ValueError('nonfinite constant')


def _float(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('nonfinite number')
    return result


def load_input(path):
    try:
        with Path(path).open('rb') as stream:
            raw = stream.read(MAX_INPUT_BYTES + 1)
    except (OSError, ValueError):
        raise ScopeError('LEDGER_INPUT_UNREADABLE') from None
    if len(raw) > MAX_INPUT_BYTES:
        raise ScopeError('LEDGER_INPUT_TOO_LARGE')
    try:
        value = json.loads(raw.decode('utf-8'), object_pairs_hook=_pairs,
                           parse_constant=_constant, parse_float=_float)
    except (UnicodeError, ValueError, RecursionError):
        raise ScopeError('LEDGER_JSON_INVALID') from None
    stack = [(value, 0)]
    count = 0
    while stack:
        node, depth = stack.pop()
        count += 1
        if depth > MAX_DEPTH or count > MAX_NODES:
            raise ScopeError('LEDGER_JSON_COMPLEXITY_LIMIT')
        if isinstance(node, dict):
            stack.extend((v, depth + 1) for v in node.values())
        elif isinstance(node, list):
            stack.extend((v, depth + 1) for v in node)
    return value


def run_ledger_cli(ns):
    ledger, policy = load_input(ns.ledger), load_input(ns.policy)
    if ns.command == 'ledger-review':
        return review_ledger(ledger, policy, as_of=ns.as_of)
    return as_of_snapshot(ledger, policy, as_of=ns.as_of,
                          vintage_target=ns.vintage_target)


def encode_result(result):
    blocked = result['status'] == 'BLOCKED'
    encoded = json.dumps(dict(status='BLOCKED' if blocked else 'OK', data=result),
                         ensure_ascii=True, sort_keys=True, allow_nan=False)
    if len(encoded) > MAX_OUTPUT_BYTES:  # ensure_ascii makes chars equal bytes.
        raise ScopeError('LEDGER_OUTPUT_TOO_LARGE')
    return encoded, 2 if blocked else 0


def run_mechanism_study_cli(ns):
    from .mechanism_study import review_mechanism_study
    study = load_input(ns.study)
    ledger, policy = load_input(ns.ledger), load_input(ns.policy)
    return review_mechanism_study(study, ledger, policy)


def run_ols_reconstruction_cli(ns):
    from .mechanism_regression import reconstruct_ols
    return reconstruct_ols(load_input(ns.spec))


def run_synthetic_gaussian_cli(ns):
    from .mechanism_gaussian import synthetic_gaussian_states
    return synthetic_gaussian_states(load_input(ns.spec))
