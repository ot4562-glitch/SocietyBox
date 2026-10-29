"""Additive, read-only mechanism proposal review; no scientific execution.

This deliberately small v1 contract supports synthetic retrospective proposals
only. Candidate rules are inert text. A READY review means structurally complete
proposals bound to declared inputs, never permission to run or a fitted result.
Hashes detect changed JSON, not authorship, approval or source authenticity.
"""
from copy import deepcopy
import hashlib
import json
import math

from .mechanism_vintage import _digest, _instant, _keys, _text, review_ledger


def mechanism_input_digest(value):
    """SHA256 of sorted compact ASCII JSON, with bounded JSON-only inputs.

    Object ordering is ignored; list order and JSON types remain significant.
    At most 32 levels, 100,000 nodes, 512 characters per string/key and 2 MiB
    encoded bytes. This helper reads no files and does not authenticate inputs.
    """
    stack, nodes = [(value, 0)], 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if depth > 32 or nodes > 100_000:
            raise ValueError('JSON bounds')
        if type(item) is dict:
            if len(item) > 100_000:
                raise ValueError('JSON bounds')
            for key in item:
                if type(key) is not str or len(key) > 512:
                    raise ValueError('JSON key')
            stack.extend((v, depth + 1) for v in item.values())
        elif type(item) is list:
            if len(item) > 100_000:
                raise ValueError('JSON bounds')
            stack.extend((v, depth + 1) for v in item)
        elif type(item) is str:
            if len(item) > 512:
                raise ValueError('JSON string')
        elif type(item) is float:
            if not math.isfinite(item):
                raise ValueError('JSON number')
        elif item is not None and type(item) not in (bool, int):
            raise ValueError('JSON type')
    digest, size = hashlib.sha256(), 0
    try:
        encoder = json.JSONEncoder(sort_keys=True, separators=(',', ':'),
                                   ensure_ascii=True, allow_nan=False)
        for part in encoder.iterencode(value):
            encoded = part.encode('ascii')
            size += len(encoded)
            if size > 2 * 1024 * 1024:
                raise ValueError('JSON byte limit')
            digest.update(encoded)
    except (OverflowError, RecursionError) as exc:
        raise ValueError('JSON encoding') from exc
    return digest.hexdigest()


def review_mechanism_study(study, ledger, policy):
    """Validate proposals and re-review their exact ledger/policy at the cutoff.

    Requires 2..32 explicitly supplied candidates and 1..128 unverified
    assumptions. All candidates remain PROPOSED; no ranking, equivalence test,
    identification, estimator, forecast, execution or authorization is granted.
    A blocked review emits no candidate/assumption declarations or cell values.
    No input mutation, IO, arbitrary code evaluation or tool registration.
    """
    result = dict(schema='societybox-mechanism-study-review-v1', status='BLOCKED',
        review_scope='PROPOSAL_CONTRACT_ONLY', assurance='SYNTHETIC_LEDGER_CONTRACT_ONLY',
        claim_level='SIMULATED_RETROSPECTIVE_ANALYSIS', diagnostics=[], binding=None,
        ledger_review=None, candidates=[], assumptions=[], execution='NOT_RUN',
        execution_authorization='NOT_GRANTED', source_authentication='NOT_ESTABLISHED',
        researcher_information_set='NOT_ESTABLISHED', mechanism_identification='NOT_ASSESSED',
        estimation='NOT_IMPLEMENTED', forecast='NOT_IMPLEMENTED',
        identify='NOT_IMPLEMENTED', causal_claim='NOT_IMPLEMENTED')

    def block(code):
        result['diagnostics'].append(dict(code=code, blocking=True))
        return result

    try:
        _keys(study, {'schema', 'question', 'claim_goal', 'info_policy',
                      'assumptions', 'mechanism_candidates'})
        if study['schema'] != 'societybox-mechanism-study-v1':
            raise ValueError('schema')
        _text(study['question'])
        assumptions, candidates = study['assumptions'], study['mechanism_candidates']
        if (not isinstance(assumptions, list) or not 1 <= len(assumptions) <= 128
                or not isinstance(candidates, list) or not 2 <= len(candidates) <= 32):
            raise ValueError('proposal bounds')
    except (ValueError, TypeError):
        return block('MECHANISM_STUDY_INVALID')
    if study['claim_goal'] != 'SIMULATED_RETROSPECTIVE_ANALYSIS':
        return block('MECHANISM_CLAIM_UNSUPPORTED')
    try:
        info = study['info_policy']
        _keys(info, {'cutoff', 'vintage_target', 'ledger_digest', 'policy_digest'})
        _instant(info['cutoff'])
        if info['vintage_target'] != 'AS_OF':
            raise ValueError('vintage target')
        _digest(info['ledger_digest'])
        _digest(info['policy_digest'])
    except (ValueError, TypeError, OverflowError):
        return block('MECHANISM_INFORMATION_POLICY_INVALID')
    ids = set()
    try:
        for assumption in assumptions:
            _keys(assumption, {'assumption_id', 'statement', 'evidence_status'})
            aid = _text(assumption['assumption_id'])
            _text(assumption['statement'])
            if aid in ids or assumption['evidence_status'] != 'DECLARED_UNVERIFIED':
                raise ValueError('unverified assumptions required')
            ids.add(aid)
    except (ValueError, TypeError):
        return block('MECHANISM_ASSUMPTION_INVALID')
    candidate_ids = set()
    try:
        for candidate in candidates:
            _keys(candidate, {'candidate_id', 'proposed_rule', 'assumption_ids',
                              'identification_strategy'})
            cid = _text(candidate['candidate_id'])
            _text(candidate['proposed_rule'])
            refs = candidate['assumption_ids']
            if (cid in candidate_ids or not isinstance(refs, list)
                    or not 1 <= len(refs) <= 128
                    or any(not isinstance(ref, str) for ref in refs)
                    or len(set(refs)) != len(refs) or not set(refs) <= ids
                    or candidate['identification_strategy'] != 'NOT_ESTABLISHED'):
                raise ValueError('candidate')
            candidate_ids.add(cid)
    except (ValueError, TypeError):
        return block('MECHANISM_CANDIDATE_INVALID')
    try:
        study_digest = mechanism_input_digest(study)
        ledger_digest = mechanism_input_digest(ledger)
        policy_digest = mechanism_input_digest(policy)
    except ValueError:
        return block('MECHANISM_INPUT_INVALID')
    if (ledger_digest != info['ledger_digest'] or policy_digest != info['policy_digest']):
        return block('MECHANISM_INPUT_DIGEST_MISMATCH')
    result['binding'] = dict(study_digest=study_digest, ledger_digest=ledger_digest,
                             policy_digest=policy_digest, cutoff=info['cutoff'],
                             vintage_target=info['vintage_target'])
    result['ledger_review'] = review_ledger(ledger, policy, as_of=info['cutoff'])
    if result['ledger_review']['status'] != 'READY':
        return block('MECHANISM_LEDGER_BLOCKED')
    if result['ledger_review']['eligible_cell_count'] == 0:
        return block('MECHANISM_INFORMATION_EMPTY')
    result['status'] = 'READY'
    result['assumptions'] = deepcopy(assumptions)
    result['candidates'] = [dict(deepcopy(c), status='PROPOSED') for c in candidates]
    return result
