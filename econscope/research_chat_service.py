"""Explicitly launched, finite-lived loopback research chat over trusted studies.

No general file routes, uploads of executable code, remote LLM or upstream
installation. Optional local model turns require explicit selected-text approval. The private startup URL carries a process-local capability in a
fragment; requests send it in Authorization, never a query or a server log.
"""
import argparse
import hmac
from http.server import BaseHTTPRequestHandler
import json
from pathlib import Path
from socketserver import ThreadingMixIn
import time
import threading
from urllib.parse import urlsplit, parse_qs

from . import engine_study as engine
from .research_chat import ResearchChat
from .standalone_service import ReviewServer, ReviewHandler, _bounded
from .execution_control import deadline_budget, checkpoint
from .research_jobs import ResearchJobs
from .research_conversation import ResearchConversation


class ChatServer(ThreadingMixIn,ReviewServer):
    daemon_threads=False
    block_on_close=True

    def __init__(self, workspace, port=0, ttl=3600, max_requests=10000, max_bytes=64000000, *, runtime=None, sources=None, llm_provider=None, llm_model=None, llm_base_url=None):
        self.stop_event=threading.Event();self._quota_lock=threading.Lock();self._connections=threading.BoundedSemaphore(16);self.jobs=None
        self.remaining_control_requests=20000
        self.chat = ResearchChat(workspace, runtime=runtime, sources=sources)
        try:
            self.conversation = ResearchConversation(self.chat, provider=llm_provider,
                                                     model=llm_model, base_url=llm_base_url)
            super().__init__(port, ttl, max_requests, max_bytes)
            self.RequestHandlerClass = ChatHandler
            self.jobs=ResearchJobs(self.chat,deadline=self.deadline)
        except BaseException:
            if hasattr(self,'socket'):self.server_close()
            self.chat.close(); raise

    def server_close(self):
        self.stop_event.set()
        if self.jobs is not None:self.jobs.close()
        super().server_close()
        self.chat.close()

    def process_request(self,request,client_address):
        if not self._connections.acquire(blocking=False):
            self.shutdown_request(request);return
        try:super().process_request(request,client_address)
        except BaseException:
            self._connections.release();raise

    def process_request_thread(self,request,client_address):
        try:super().process_request_thread(request,client_address)
        finally:self._connections.release()


class ChatHandler(ReviewHandler):
    server_version = 'SocietyBoxResearch/1'

    def handle(self):
        self.connection.settimeout(1)
        self._timer=threading.Timer(max(.001,min(2,self.server.deadline-time.monotonic())),self._disconnect)
        self._timer.daemon=True;self._timer.start()
        try:
            BaseHTTPRequestHandler.handle(self)
        except (OSError,ValueError): pass
        finally:self._timer.cancel()

    def _authenticated_operation(self):
        # Header/body drip protection ends only after full framing and capability
        # validation. Built-in studies retain their own finite worker/oracle limits.
        self._timer.cancel()
        self.operation_deadline=min(self.server.deadline,time.monotonic()+600)
        self._timer=threading.Timer(max(.001,self.operation_deadline-time.monotonic()),self._disconnect)
        self._timer.daemon=True;self._timer.start()
        self.connection.settimeout(10)

    def _send(self, status, raw, content_type='application/json; charset=utf-8', filename=None):
        if 200<=status<300:checkpoint()
        self.send_response(status)
        for key, value in (('Content-Type',content_type), ('Content-Length',str(len(raw))),
                           ('Cache-Control','no-store'), ('X-Content-Type-Options','nosniff'),
                           ('Referrer-Policy','no-referrer'), ('X-Frame-Options','DENY'),
                           ('Connection','close')):
            self.send_header(key,value)
        if filename: self.send_header('Content-Disposition','attachment; filename="'+filename+'"')
        self.end_headers(); self.close_connection=True
        self.wfile.write(raw)

    def _reply(self, status, value):
        self._send(status, engine._bytes(value))

    def _dispatch(self):
        try:
            self._route()
        except (ValueError, OSError, RuntimeError, KeyError, TypeError, RecursionError, UnicodeError):
            # Fixed errors only. Never echo data, capability, private path or traceback.
            self._error(400,'RESEARCH_REQUEST_REJECTED')

    def _route(self):
        server=self.server
        if time.monotonic() >= server.deadline: return self._error(410,'SERVICE_EXPIRED')
        origin=f'http://127.0.0.1:{server.server_port}'
        if (self.headers.get_all('Host') != [f'127.0.0.1:{server.server_port}']
                or self.headers.get_all('Origin') not in (None,[origin])
                or self.headers.get('Sec-Fetch-Site') in ('cross-site','same-site')):
            return self._error(403,'SAME_ORIGIN_LOOPBACK_REQUIRED')
        if self.headers.get_all('Transfer-Encoding'): return self._error(400,'TRANSFER_ENCODING_FORBIDDEN')
        lengths=self.headers.get_all('Content-Length') or []
        if len(lengths)>1 or (lengths and (not lengths[0].isascii() or not lengths[0].isdigit() or len(lengths[0])>9)):
            return self._error(400,'CONTENT_LENGTH_INVALID')
        size=int(lengths[0]) if lengths else 0
        if size>131072: return self._error(413,'BODY_TOO_LARGE')
        raw=self.rfile.read(size)
        if len(raw)!=size: return self._error(400,'BODY_INCOMPLETE')
        parts=urlsplit(self.path)
        if parts.scheme or parts.netloc or parts.fragment: return self._error(400,'PATH_INVALID')
        query=parse_qs(parts.query,keep_blank_values=True,strict_parsing=True,max_num_fields=8)
        if any(len(v)!=1 for v in query.values()): return self._error(400,'QUERY_INVALID')
        path=parts.path
        is_cancel=(self.command=='POST' and path.startswith('/api/jobs/') and path.endswith('/cancel') and size<=64 and raw.strip()==b'{}')
        is_status=(self.command=='GET' and not size and (path=='/api/jobs' or path.startswith('/api/jobs/')))
        with server._quota_lock:
            if not is_cancel and not is_status:
                if server.remaining_requests<=0 or size>server.remaining_bytes:return self._error(429,'SERVICE_QUOTA_EXHAUSTED')
                server.remaining_requests-=1;server.remaining_bytes-=size
        if path=='/' and self.command=='GET' and not size and set(query)<= {'locale'}:
            from .research_review_entry import render_research_chat
            return self._send(200,render_research_chat(query.get('locale',['ko'])[0]).encode('utf-8'),'text/html; charset=utf-8')
        auth=self.headers.get_all('Authorization') or []
        if len(auth)!=1 or not hmac.compare_digest(auth[0].encode('utf-8'),('Bearer '+server.token).encode('ascii')):
            return self._error(401,'LOCAL_CAPABILITY_REQUIRED')
        if is_status:
            with server._quota_lock:
                if server.remaining_control_requests<=0:return self._error(429,'CONTROL_QUOTA_EXHAUSTED')
                server.remaining_control_requests-=1
        self._authenticated_operation()
        with deadline_budget(self.operation_deadline,cancel=server.stop_event):
            return self._authorized(path,query,size,lengths,raw)

    def _authorized(self,path,query,size,lengths,raw):
        server=self.server
        if self.command=='GET':
            if size: return self._error(400,'BODY_FORBIDDEN')
            if path=='/api/state':
                if not set(query)<= {'messages','versions','runs','size'}: raise ValueError('query')
                options={key[:-1]+'_offset':int(query[key][0]) for key in ('messages','versions','runs') if key in query}
                options['page_size']=int(query.get('size',['40'])[0])
                state=server.chat.snapshot(**options)
                state['llm']=server.conversation.status()['mode']
                state['external_transfer']=server.conversation.status()['external_transfer']
                return self._reply(200,state)
            if path=='/api/templates' and not query:
                from .research_chat_ui import default_studies
                return self._reply(200,default_studies())
            if path=='/api/capabilities' and set(query)<= {'locale'}:
                capabilities=server.chat.capabilities(query.get('locale',['ko'])[0])
                capabilities['llm']=server.conversation.status()['mode']
                capabilities['external_transfer']=server.conversation.status()['external_transfer']
                return self._reply(200,capabilities)
            segments=path.split('/')
            if path=='/api/conversation/status' and not query:
                return self._reply(200,server.conversation.status())
            if path=='/api/conversation/turns' and set(query)<= {'offset','size'}:
                return self._reply(200,server.conversation.list(
                    offset=int(query.get('offset',['0'])[0]),size=int(query.get('size',['20'])[0])))
            if len(segments)==5 and segments[1:3]==['api','conversation'] and segments[3]=='turn' and not query:
                return self._reply(200,server.conversation.read(segments[4]))
            if path=='/api/jobs' and set(query)<= {'offset','size'}:
                return self._reply(200,server.jobs.list(offset=int(query.get('offset',['0'])[0]),size=int(query.get('size',['40'])[0])))
            if len(segments)==4 and segments[1:3]==['api','jobs'] and set(query)<= {'wait'}:
                return self._reply(200,server.jobs.get(segments[3],wait=int(query.get('wait',['0'])[0])))
            if len(segments)==4 and segments[1:3]==['api','run']:
                if not set(query)<= {'locale','offset'}: raise ValueError('query')
                return self._reply(200,server.chat.read_run(segments[3],query.get('locale',['ko'])[0],
                                                           evidence_offset=int(query.get('offset',['0'])[0])))
            if len(segments)==5 and segments[1:3]==['api','download'] and not query:
                content_type,blob=server.chat.download(segments[3],segments[4])
                return self._send(200,blob,content_type,segments[4])
            return self._error(404,'OPERATION_NOT_EXPOSED')
        if self.command!='POST': return self._error(405,'METHOD_NOT_ALLOWED')
        if query: raise ValueError('query')
        if not lengths: return self._error(411,'CONTENT_LENGTH_REQUIRED')
        if self.headers.get_all('Content-Type') != ['application/json']: return self._error(415,'JSON_CONTENT_TYPE_REQUIRED')
        payload=engine._parse(raw)
        segments=path.split('/')
        if len(segments)==5 and segments[1:3]==['api','jobs'] and segments[4]=='cancel':
            if payload!={}:raise ValueError('CHAT_CANCEL_EMPTY_OBJECT_REQUIRED')
            return self._reply(200,server.jobs.cancel(segments[3]))
        if len(segments)==5 and segments[1:3]==['api','jobs'] and segments[4]=='recover':
            if type(payload) is not dict or set(payload)!={'acknowledgment'}:raise ValueError('CHAT_RECOVERY_ACK_REQUIRED')
            return self._reply(200,server.jobs.recover(segments[3],payload['acknowledgment']))
        if path=='/api/conversation/review':
            if type(payload) is not dict or set(payload)!={'text','locale','expected_revision','classification'}:
                raise ValueError('CONVERSATION_REVIEW_SCHEMA')
            return self._reply(200,server.conversation.review(**payload))
        if path=='/api/conversation/begin':
            if type(payload) is not dict or set(payload)!={'text','locale','expected_revision','classification','acknowledgment','confirmed'}:
                raise ValueError('CONVERSATION_BEGIN_SCHEMA')
            return self._reply(200,server.conversation.begin(**payload))
        if path=='/api/conversation/send':
            if type(payload) is not dict or set(payload)!={'turn_id','acknowledgment','confirmed'}:
                raise ValueError('CONVERSATION_SEND_SCHEMA')
            return self._reply(200,server.conversation.send(**payload))
        if path=='/api/conversation/draft':
            if type(payload) is not dict or set(payload)!={'turn_id','call_id'}:
                raise ValueError('CONVERSATION_DRAFT_SCHEMA')
            return self._reply(200,server.conversation.draft(**payload))
        routes={'/api/message':{'text','expected_revision'}, '/api/proposal':{'spec','expected_revision'},
                '/api/run':{'version_id','acknowledgment','expected_revision','confirmed'},
                '/api/jobs':{'version_id','acknowledgment','expected_revision','confirmed'}}
        if path not in routes: return self._error(404,'OPERATION_NOT_EXPOSED')
        if type(payload) is not dict or set(payload)!=routes[path]: raise ValueError('payload')
        try:
            if path=='/api/message': result=server.chat.message(**payload)
            elif path=='/api/proposal': result=server.chat.propose(**payload)
            elif path=='/api/jobs':return self._reply(202,server.jobs.submit(**payload))
            else: result=server.chat.execute(**payload,operation_deadline=self.operation_deadline)
        except ValueError as exc:
            if str(exc) in ('CHAT_REVISION_CONFLICT','CHAT_STALE_VERSION','CHAT_STALE_CODE_REVIEW','CHAT_EXECUTION_BUSY'):
                return self._error(409,'REVIEW_AGAIN_REVISION_OR_CODE_CHANGED')
            raise
        self._reply(200,result)

    do_GET=_dispatch
    do_POST=_dispatch
    do_PUT=_dispatch
    do_DELETE=_dispatch
    do_OPTIONS=_dispatch
    do_HEAD=_dispatch


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',required=True,help='New empty directory, or an existing research-chat workspace')
    parser.add_argument('--port',type=_bounded(0,65535),default=0)
    parser.add_argument('--ttl-seconds',type=_bounded(1,3600),default=3600)
    parser.add_argument('--runtime',help='Explicit reviewed local HiGHS runtime directory; pin verified before serving')
    parser.add_argument('--sources',help='Explicit pinned source archive directory; source/license rechecked by each adapter')
    parser.add_argument('--llm-provider',choices=['ollama','openai-compatible-local'],help='Optional explicitly selected loopback-only local LLM')
    parser.add_argument('--llm-model',help='Explicit local model identifier')
    parser.add_argument('--llm-base-url',help='Literal http://127.0.0.1:PORT or [::1] endpoint (no DNS/remote)')
    args=parser.parse_args(argv)
    try:
        with ChatServer(Path(args.workspace),port=args.port,ttl=args.ttl_seconds,runtime=args.runtime,sources=args.sources,
                        llm_provider=args.llm_provider,llm_model=args.llm_model,llm_base_url=args.llm_base_url) as server:
            print(json.dumps({'status':'LOCAL_RESEARCH_CHAT','url':f'http://127.0.0.1:{server.server_port}/#'+server.token,
                              'llm':server.conversation.status()['mode'],
                               'external_transfer':server.conversation.status()['external_transfer'],
                               'ttl_seconds':args.ttl_seconds}),flush=True)
            try:
                while time.monotonic()<server.deadline: server.handle_request()
            except KeyboardInterrupt: pass
        return 0
    except (ValueError,OSError,RuntimeError,KeyError,TypeError):
        print(json.dumps({'status':'BLOCKED','reason':'LOCAL_RESEARCH_CHAT_START_FAILED'}))
        return 2


if __name__=='__main__': raise SystemExit(main())
