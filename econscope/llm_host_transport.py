"""Opt-in, bounded Society Box LLM host transports.

These are transports, not permission to run tools, export studies, execute code,
assert source accuracy or infer that a model semantically obeyed AGENT.md.
Only explicit loopback endpoints are supported. Cloud Codex is a separate
synthetic-only integration probe, never the air-gapped production backend.
"""
import http.client
import errno
import ipaddress
import json
import math
import re
import select
import socket
import time
from urllib.parse import urlsplit

from .agent_host import consume_host_request
from .agent_response import parse_host_response
from .workspace import ScopeError, canonical
from .execution_control import checkpoint, remaining_timeout

MAX_WIRE_BYTES = 64000
_MODEL = re.compile(r'[A-Za-z0-9_][A-Za-z0-9_.:/-]{0,127}', re.ASCII)


def _model_id(value):
    if not isinstance(value, str) or _MODEL.fullmatch(value) is None:
        raise ScopeError('LLM_MODEL_ID_INVALID')
    return value


def _local_endpoint(base_url):
    """Reject DNS names, credentials, non-loopback IPs, fragments and proxy paths."""
    if not isinstance(base_url, str) or len(base_url) > 150:
        raise ScopeError('LLM_LOCAL_ENDPOINT_INVALID')
    try:
        parsed = urlsplit(base_url)
        if (parsed.scheme != 'http' or parsed.username or parsed.password or
                parsed.query or parsed.fragment or parsed.path not in ('', '/') or
                parsed.hostname is None or parsed.port is None):
            raise ValueError
        address = ipaddress.ip_address(parsed.hostname)
        if not address.is_loopback:
            raise ValueError
    except (TypeError, ValueError):
        raise ScopeError('LLM_LOCAL_ENDPOINT_INVALID') from None
    return parsed.hostname, parsed.port


def _validate_request(request):
    snapshot = consume_host_request(request, lambda body: body)
    decoded = json.loads(snapshot)
    first, supplemental, user = decoded['messages']
    if [m['role'] for m in (first, supplemental, user)] != [
            'host_instruction', 'host_instruction', 'user']:
        raise ScopeError('AGENT_HOST_REQUEST_ROLE_INVALID')
    return decoded, first['content'], supplemental['content'], user['content']


class _BudgetSocket(socket.socket):
    """HTTP's buffered file calls recv_into; keep that wait cancellable too.

    Windows socket shutdown need not wake a timeout-mode buffered header read.
    Nonblocking I/O with bounded select waits avoids an uninterruptible reader
    or a second thread that might outlive the approved operation.
    """
    def __init__(self, *args, guard, **kwargs):
        super().__init__(*args, **kwargs)
        self._guard = guard

    def recv_into(self, buffer, nbytes=0, flags=0):
        while True:
            self._guard()
            readable, _, _ = select.select([self], [], [], .025)
            if readable:
                self._guard()
                try:
                    return super().recv_into(buffer, nbytes, flags)
                except BlockingIOError:
                    pass

    def sendall(self, data, flags=0):
        remaining = memoryview(data)
        while remaining:
            self._guard()
            _, writable, _ = select.select([], [self], [], .025)
            if writable:
                self._guard()
                try:
                    sent = super().send(remaining, flags)
                except BlockingIOError:
                    continue
                if not sent:
                    raise OSError('loopback connection closed')
                remaining = remaining[sent:]


def _connect_loopback(conn, guard):
    """Own a nonblocking literal-IP socket so cancellation also covers connect."""
    address = ipaddress.ip_address(conn.host)
    sock = _BudgetSocket(socket.AF_INET6 if address.version == 6 else socket.AF_INET,
                         socket.SOCK_STREAM, guard=guard)
    try:
        sock.setblocking(False)
        guard()
        pending = sock.connect_ex((conn.host, conn.port))
        if pending not in (0, errno.EINPROGRESS, errno.EWOULDBLOCK,
                           errno.EALREADY, errno.EINTR, 10035, 10036, 10037):
            raise OSError(pending, 'loopback connection refused')
        while pending:
            guard()
            _, writable, exceptional = select.select([], [sock], [sock], .025)
            if writable or exceptional:
                error = sock.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
                if error:
                    raise OSError(error, 'loopback connection failed')
                break
        guard()
        conn.sock = sock
    except BaseException:
        sock.close()
        raise


def _post_json(host, port, resource, payload, timeout):
    data = canonical(payload)
    if len(data) > MAX_WIRE_BYTES:
        raise ScopeError('LLM_LOCAL_REQUEST_TOO_LARGE')
    if not isinstance(timeout, (int, float)) or not 0 < timeout <= 30:
        raise ScopeError('LLM_HOST_TIMEOUT_INVALID')
    checkpoint()
    end = time.monotonic() + timeout
    conn = http.client.HTTPConnection(host, port, timeout=remaining_timeout(timeout))
    response = None

    def guard():
        checkpoint()
        if time.monotonic() >= end:
            raise ScopeError('LLM_LOCAL_DEADLINE_EXCEEDED')

    try:
        try:
            _connect_loopback(conn, guard)
            guard()  # A cancellation during connect must prevent sending text.
            conn.request('POST', resource, body=data, headers={
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'Connection': 'close',
            })
            response = conn.getresponse()
            raw = response.read(MAX_WIRE_BYTES + 1)
            guard()
        except (OSError, TimeoutError, http.client.HTTPException) as exc:
            guard()
            raise ScopeError('LLM_LOCAL_PROVIDER_UNAVAILABLE') from exc
        if len(raw) > MAX_WIRE_BYTES:
            raise ScopeError('LLM_LOCAL_RESPONSE_TOO_LARGE')
        if response.status == 429:
            # Do not retry, surface provider text, or fall back to another host.
            raise ScopeError('LLM_LOCAL_USAGE_LIMIT')
        if response.status != 200:
            raise ScopeError('LLM_LOCAL_HTTP_REJECTED')
        def unique_object(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError('duplicate provider key')
                result[key] = value
            return result

        def invalid_constant(value):
            raise ValueError('nonfinite provider number')

        try:
            decoded = json.loads(raw.decode('utf-8'), object_pairs_hook=unique_object,
                                 parse_constant=invalid_constant)
            if not isinstance(decoded, dict):
                raise ValueError('provider envelope must be an object')
            pending = [(decoded, 0)]
            while pending:
                value, depth = pending.pop()
                if depth > 32:
                    raise ValueError('provider nesting limit')
                # JSON exponent literals bypass parse_constant and can overflow
                # Python's float decoder, including inside ignored metadata.
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError('nonfinite provider number')
                children = value.values() if isinstance(value, dict) else value if isinstance(value, list) else ()
                pending.extend((child, depth + 1) for child in children)
            return decoded
        except (ValueError, UnicodeError, RecursionError):
            raise ScopeError('LLM_LOCAL_RESPONSE_INVALID') from None
    finally:
        if response is not None:
            response.close()
        conn.close()


def call_loopback_host(request, *, provider, model, base_url=None,
                       timeout=10, allow_local_host=False,
                       include_untrusted_arguments=False):
    """Use an explicitly selected loopback model without granting any tools.

    The first system message contains EXACT product AGENT.md bytes, followed by
    the supplementary instruction and user message. The server can accept the
    payload; that alone cannot prove the model consumed/obeyed its system role.
    """
    if allow_local_host is not True:
        raise ScopeError('LLM_LOCAL_HOST_EXPLICIT_APPROVAL_REQUIRED')
    _model_id(model)
    if provider == 'ollama':
        host, port = _local_endpoint(base_url or 'http://127.0.0.1:11434')
        endpoint = '/api/chat'
    elif provider == 'openai-compatible-local':
        host, port = _local_endpoint(base_url or 'http://127.0.0.1:1234')
        endpoint = '/v1/chat/completions'
    else:
        raise ScopeError('LLM_LOCAL_PROVIDER_UNSUPPORTED')
    decoded, first, supplemental, user = _validate_request(request)
    messages = [
        {'role': 'system', 'content': first},
        {'role': 'system', 'content': supplemental},
        {'role': 'user', 'content': user},
    ]
    if provider == 'ollama':
        payload = {'model': model, 'stream': False, 'messages': messages,
                   'format': 'json', 'options': {'temperature': 0}}
    else:
        payload = {'model': model, 'stream': False, 'messages': messages,
                   'temperature': 0}
    response = _post_json(host, port, endpoint, payload, timeout)
    try:
        if 'error' in response:
            raise ValueError
        if provider == 'ollama':
            if 'done' in response and response['done'] is not True:
                raise ValueError
            message = response['message']
        else:
            choices = response['choices']
            if type(choices) is not list or len(choices) != 1:
                raise ValueError
            if ('finish_reason' in choices[0] and
                    choices[0]['finish_reason'] != 'stop'):
                raise ValueError
            message = choices[0]['message']
        if (not isinstance(message, dict) or
                message.get('role', 'assistant') != 'assistant' or
                message.get('tool_calls') or message.get('function_call')):
            raise ValueError
        content = message['content']
        if not isinstance(content, str):
            raise ValueError
        checked = parse_host_response(content.encode('utf-8'))
    except (KeyError, IndexError, TypeError, ValueError, UnicodeError, ScopeError) as exc:
        raise ScopeError('LLM_LOCAL_MODEL_RESPONSE_NOT_VALIDATED') from exc
    return {
        'schema': 'societybox-local-host-result-v1',
        'provider': provider,
        'endpoint_policy': 'EXPLICIT_LITERAL_LOOPBACK_IP_ONLY',
        'model': model,
        'first_message_sha256': decoded['first_message_sha256'],
        'first_role_on_wire': 'system',
        'server_response': 'RECEIVED_AND_SCHEMA_VALIDATED',
        'model_instruction_consumption': 'NOT_INDEPENDENTLY_ATTESTED',
        'offline_os_network_enforcement': 'NOT_ESTABLISHED',
        'claim_verification': checked.claim_verification,
        'assistant_text': checked.assistant_text,
        'tool_calls': [{'id': x.id, 'name': x.name, 'authorization': x.authorization,
                        'status': x.status,
                        **({'untrusted_arguments': json.loads(x.arguments_json)}
                           if include_untrusted_arguments is True else {})}
                       for x in checked.tool_calls],
    }
