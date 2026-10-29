"""Loopback-only inline response review service, independent of desktop dispatch.

No MCP compatibility claim: this is a small versioned JSON/HTTP service. Request
data stays in memory. It exposes no filesystem, model, network client or execution
operation. This application boundary is not an OS sandbox or tenant identity.
"""
import argparse
import hmac
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
import secrets
import socket
import threading
import time

from .agent_response import MAX_RESPONSE_BYTES, parse_host_response
from .workspace import ScopeError, canonical


PROTOCOL = 'societybox-inline-review-service-v1'
INPUT_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'required': ['schema', 'assistant_text', 'tool_calls'],
    'properties': {
        'schema': {'const': 'societybox-host-response-v1'},
        'assistant_text': {'type': 'string'},
        'tool_calls': {'type': 'array', 'maxItems': 64, 'items': {
            'type': 'object', 'additionalProperties': False,
            'required': ['id', 'name', 'arguments'], 'properties': {
                'id': {'type': 'string', 'pattern': r'^[A-Za-z0-9_][A-Za-z0-9_.:-]{0,127}$'},
                'name': {'type': 'string', 'pattern': r'^[A-Za-z0-9_][A-Za-z0-9_.:-]{0,127}$'},
                'arguments': {'type': 'object'},
            }}}
    },
}
OUTPUT_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'required': ['schema', 'assistant_text', 'tool_calls', 'trust', 'claim_verification'],
    'properties': {
        'schema': {'const': 'societybox-inline-review-v1'},
        'assistant_text': {'type': 'string'},
        'trust': {'const': 'UNTRUSTED_MODEL_OUTPUT'},
        'claim_verification': {'const': 'NOT_RUN'},
        'tool_calls': {'type': 'array', 'maxItems': 64, 'items': {
            'type': 'object', 'additionalProperties': False,
            'required': ['id', 'name', 'authorization', 'status'], 'properties': {
                'id': {'type': 'string'}, 'name': {'type': 'string'},
                'authorization': {'const': 'NOT_GRANTED'},
                'status': {'const': 'BLOCKED_NO_AUTHORIZED_TOOLS'},
            }}}
    },
}


class ReviewServer(HTTPServer):
    allow_reuse_address = False

    def __init__(self, port, ttl, max_requests, max_bytes):
        # No caller-supplied host or desktop workspace exists in this API.
        for value, low, high in ((port, 0, 65535), (ttl, 1, 3600),
                                 (max_requests, 1, 10000), (max_bytes, 1, 64000000)):
            if type(value) is not int or not low <= value <= high:
                raise ValueError('service limit out of bounds')
        super().__init__(('127.0.0.1', port), ReviewHandler)
        self.token = secrets.token_urlsafe(32)
        self.deadline = time.monotonic() + ttl
        self.remaining_requests = max_requests
        self.remaining_bytes = max_bytes
        self.stopping = False
        self.timeout = 0.1


class ReviewHandler(BaseHTTPRequestHandler):
    server_version = 'SocietyBoxInline/1'
    sys_version = ''

    def log_message(self, *args):
        pass  # No request, untrusted payload or capability logging.

    def handle(self):
        self.connection.settimeout(1)
        # An absolute deadline also bounds slow clients that drip header bytes.
        timer = threading.Timer(max(0.001, min(2, self.server.deadline - time.monotonic())),
                                self._disconnect)
        timer.daemon = True
        timer.start()
        try:
            super().handle()
        except (OSError, ValueError):
            pass
        finally:
            timer.cancel()

    def _disconnect(self):
        try:
            self.connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

    def _reply(self, code, value):
        blob = canonical(value)
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(blob)))
        self.send_header('Connection', 'close')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.close_connection = True
        self.wfile.write(blob)

    def send_error(self, code, message=None, explain=None):
        self._reply(code, {'error': 'HTTP_REQUEST_REJECTED'})

    def _error(self, code, reason):
        self._reply(code, {'error': reason})

    def _dispatch(self):
        server = self.server
        if time.monotonic() >= server.deadline:
            return self._error(410, 'SERVICE_EXPIRED')
        lengths = self.headers.get_all('Content-Length') or []
        if len(lengths) > 1 or (lengths and (not lengths[0].isascii() or not lengths[0].isdigit())):
            return self._error(400, 'CONTENT_LENGTH_INVALID')
        # Avoid parsing an unbounded integer from a header.
        if lengths and len(lengths[0]) > 9:
            return self._error(413, 'BODY_TOO_LARGE')
        size = int(lengths[0]) if lengths else 0
        if size > MAX_RESPONSE_BYTES:
            return self._error(413, 'BODY_TOO_LARGE')
        # Consume safely framed bounded bodies even for negative calls. Closing
        # with an unread POST body can reset TCP before the JSON refusal arrives
        # (observed on Windows). The absolute connection timer bounds this read.
        raw = self.rfile.read(size)
        if len(raw) != size:
            return self._error(400, 'BODY_INCOMPLETE')
        if (self.headers.get_all('Origin') or
                self.headers.get_all('Host') != [f'127.0.0.1:{server.server_port}']):
            return self._error(403, 'LOCAL_NON_BROWSER_CLIENT_REQUIRED')
        auth = self.headers.get_all('Authorization') or []
        if (len(auth) != 1 or not hmac.compare_digest(
                auth[0].encode('utf-8'), ('Bearer ' + server.token).encode('ascii'))):
            return self._error(401, 'LOCAL_CAPABILITY_REQUIRED')
        if self.headers.get_all('Transfer-Encoding'):
            return self._error(400, 'TRANSFER_ENCODING_FORBIDDEN')
        # Authenticated shutdown is available after quota exhaustion.
        shutdown = self.path == '/shutdown' and self.command == 'POST'
        if not shutdown:
            if server.remaining_requests <= 0 or size > server.remaining_bytes:
                return self._error(429, 'SERVICE_QUOTA_EXHAUSTED')
            server.remaining_requests -= 1
            server.remaining_bytes -= size
        routes = {'/schemas': 'GET', '/review': 'POST', '/shutdown': 'POST'}
        if self.path not in routes:
            return self._error(404, 'OPERATION_NOT_EXPOSED')
        if self.command != routes[self.path]:
            return self._error(405, 'METHOD_NOT_ALLOWED')
        if self.path == '/schemas':
            if size:
                return self._error(400, 'BODY_FORBIDDEN')
            return self._reply(200, {
                'protocol': PROTOCOL, 'desktop_file_access': 'NOT_EXPOSED',
                'persistence': 'NONE', 'provider': 'NONE',
                'authentication': 'PROCESS_LOCAL_BEARER_CAPABILITY',
                'max_request_bytes': MAX_RESPONSE_BYTES,
                'remaining_requests': server.remaining_requests,
                'remaining_bytes': server.remaining_bytes,
                'remaining_seconds': max(0, server.deadline - time.monotonic()),
                'operations': {
                    'POST /review': {'input_schema': INPUT_SCHEMA, 'output_schema': OUTPUT_SCHEMA},
                    'POST /shutdown': {'input_schema': {'type': 'object', 'maxProperties': 0}},
                },
                'limitations': ['NO_MCP_COMPATIBILITY_CLAIM', 'NO_OS_SANDBOX',
                                'NO_MODEL_CONSUMPTION', 'NO_INSTITUTIONAL_AUTHENTICATION'],
            })
        if not lengths:
            return self._error(411, 'CONTENT_LENGTH_REQUIRED')
        if self.headers.get_all('Content-Type') != ['application/json']:
            return self._error(415, 'JSON_CONTENT_TYPE_REQUIRED')
        if time.monotonic() >= server.deadline:
            return self._error(410, 'SERVICE_EXPIRED')
        if shutdown:
            if raw.strip() != b'{}':
                return self._error(400, 'EMPTY_OBJECT_REQUIRED')
            server.stopping = True
            return self._reply(200, {'status': 'STOPPING'})
        try:
            response = parse_host_response(raw)
        except ScopeError as exc:
            return self._error(400, exc.code)
        self._reply(200, {
            'schema': 'societybox-inline-review-v1',
            'assistant_text': response.assistant_text,
            'trust': response.trust, 'claim_verification': response.claim_verification,
            'tool_calls': [{'id': c.id, 'name': c.name, 'authorization': c.authorization,
                            'status': c.status} for c in response.tool_calls],
        })

    do_GET = _dispatch
    do_POST = _dispatch
    do_PUT = _dispatch
    do_DELETE = _dispatch
    do_OPTIONS = _dispatch
    do_HEAD = _dispatch


def _bounded(low, high):
    def parse(value):
        try:
            number = int(value)
            if not low <= number <= high:
                raise ValueError
            return number
        except ValueError:
            raise argparse.ArgumentTypeError(f'expected integer in [{low}, {high}]') from None
    return parse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=_bounded(0, 65535), default=0)
    parser.add_argument('--ttl-seconds', type=_bounded(1, 3600), default=300)
    parser.add_argument('--max-requests', type=_bounded(1, 10000), default=128)
    parser.add_argument('--max-bytes', type=_bounded(1, 64000000), default=1024000)
    args = parser.parse_args()
    with ReviewServer(args.port, args.ttl_seconds, args.max_requests, args.max_bytes) as server:
        # Private supervising process/terminal only; never put this line in shared logs.
        print(json.dumps({'protocol': PROTOCOL, 'host': '127.0.0.1',
                          'port': server.server_port, 'pid': os.getpid(),
                          'token': server.token, 'ttl_seconds': args.ttl_seconds}), flush=True)
        try:
            while not server.stopping and time.monotonic() < server.deadline:
                server.handle_request()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
