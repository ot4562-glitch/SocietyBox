"""Single-writer local chat state over the existing reviewed economic studies.

Only declared data and trusted built-in calculations are accepted. No LLM,
arbitrary researcher code, source downloads or remote clients are invoked.
Integrity digests are not researcher identity or institutional authentication.
"""
from copy import deepcopy
from functools import wraps
import io
import os
from pathlib import Path
import re
import tempfile
import time
import threading
import uuid
import zipfile

from . import engine_study as engine
from .execution_control import safe_local_path as _safe, deadline_budget, checkpoint

STATE_LIMIT = 8 * 1024 * 1024
FAMILIES = tuple(engine.study_families())
_RUN = re.compile(r'[a-f0-9]{32}\Z')


def _synchronized(method):
    @wraps(method)
    def locked(self,*args,**kwargs):
        with self._state_lock:return method(self,*args,**kwargs)
    return locked


def _page(offset, size):
    if type(offset) is not int or offset < 0 or type(size) is not int or not 1 <= size <= 100:
        raise ValueError('CHAT_PAGE_INVALID')


class ResearchChat:
    def __init__(self, root, *, runtime=None, sources=None):
        self._state_lock=threading.RLock()
        self._execution_lock=threading.Lock()
        self._recovery_jobs=set()
        self.runtime = _safe(runtime) if runtime is not None else None
        self.sources = _safe(sources) if sources is not None else None
        if self.runtime is not None:
            from .highs_native import runtime_pin
            runtime_pin(self.runtime)  # authenticate the declared pin; do not execute
        if self.sources is not None and not self.sources.is_dir():
            raise ValueError('CHAT_PINNED_SOURCES_DIRECTORY_REQUIRED')
        self.root = _safe(root)
        self.root.mkdir(exist_ok=True)  # caller chooses an existing parent
        self._lock = None
        if not _safe(self.root/'state.json').exists() and any(
                p.name != '.research-chat.lock' for p in self.root.iterdir()):
            raise ValueError('CHAT_NEW_WORKSPACE_MUST_BE_EMPTY')
        lock_path = _safe(self.root / '.research-chat.lock')
        lock = lock_path.open('a+b')
        try:
            if lock_path.stat().st_size == 0:
                lock.write(b'1'); lock.flush()
            lock.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._lock = lock
            state_path = _safe(self.root / 'state.json')
            if state_path.exists():
                sealed = engine._parse(engine._read(state_path, STATE_LIMIT))
                state = sealed['state']
                if sealed.get('sha256') != engine._sha(engine._bytes(state)):
                    raise ValueError('CHAT_STATE_INTEGRITY')
                self._validate(state)
                self.state = state
            else:
                if {p.name for p in self.root.iterdir()} != {'.research-chat.lock'}:
                    raise ValueError('CHAT_NEW_WORKSPACE_MUST_BE_EMPTY')
                self.state = {'schema': 'societybox-research-chat-v1', 'revision': 0,
                              'messages': [], 'versions': [], 'runs': []}
                self._persist(self.state)
            _safe(self.root / 'runs').mkdir(exist_ok=True)
            from .research_jobs import ResearchJobs
            self._recovery_jobs={key for key,row in ResearchJobs.read_records(self.root/'jobs').items()
                                 if row['status'] not in ('SUCCEEDED','FAILED','CANCELLED','EXPIRED')}
            attempts=_safe(self.root/'attempts')
            if attempts.exists():
                published={row['id'] for row in self.state['runs']}
                for attempt in attempts.iterdir():
                    if not _RUN.fullmatch(attempt.name) or attempt.name in published:continue
                    started=_safe(attempt/'started.json')
                    failed=_safe(attempt/'failed.json');closed=False
                    if failed.exists():
                        try:
                            row=engine._parse(engine._read(failed,16384))
                            closed=(type(row) is dict and row.get('schema')=='societybox-chat-attempt-v1'
                                    and row.get('attempt_id')==attempt.name and row.get('publication')=='NOT_RECORDED'
                                    and row.get('status') in ('FAILED','EXPIRED','CANCELLED'))
                        except (ValueError,KeyError,TypeError,UnicodeError):pass
                    if started.exists() and not closed:
                        self._recovery_jobs.add(attempt.name)
        except BaseException as exc:
            unowned = self._lock is None
            if unowned: lock.close()
            self.close()
            if unowned and isinstance(exc, OSError):
                raise ValueError('CHAT_WORKSPACE_ALREADY_OPEN_OR_LOCK_UNAVAILABLE') from None
            raise

    def close(self):
        if self._execution_lock.locked():raise RuntimeError('CHAT_WORKSPACE_BUSY_CLOSE_JOBS_FIRST')
        if self._lock is not None:
            lock, self._lock = self._lock, None
            lock.close()  # OS releases the lock, including on process death

    @_synchronized
    def require_recovery(self,key):self._recovery_jobs.add(key)

    @_synchronized
    def clear_recovery(self,key):self._recovery_jobs.discard(key)

    def capabilities(self, locale='en'):
        families=[]
        for family, row in engine.study_families(locale).items():
            blockers=[]
            if 'runtime' in row['requires'] and self.runtime is None: blockers.append('PINNED_RUNTIME_NOT_CONFIGURED')
            if 'sources' in row['requires'] and self.sources is None: blockers.append('PINNED_ARCHIVES_NOT_CONFIGURED')
            families.append(dict(row, blockers=blockers, execution_authority='NOT_GRANTED'))
        return {'schema':'societybox-chat-capabilities-v1','families':families,'recovery_required':bool(self._recovery_jobs),
                'runtime':'PIN_VERIFIED' if self.runtime is not None else 'NOT_CONFIGURED',
                'sources':'CONFIGURED_REVERIFY_PER_OPERATION' if self.sources is not None else 'NOT_CONFIGURED',
                'native_execution':'NOT_RUN','llm':'NO_LLM','external_transfer':'DISABLED'}

    def _review(self, spec):
        if self.runtime is not None:_safe(self.runtime)
        if self.sources is not None:_safe(self.sources)
        if type(spec.get('request')) is not dict: raise ValueError('CHAT_STUDY_REQUEST_INVALID')
        requirements=engine.study_families()[spec['family']]['requires']
        if 'runtime' in requirements and self.runtime is None: raise ValueError('CHAT_PINNED_RUNTIME_REQUIRED')
        if ('sources' in requirements or spec['request'].get('engine')=='Pyomo+HiGHS') and self.sources is None:
            raise ValueError('CHAT_PINNED_SOURCES_REQUIRED')
        return engine.review(spec, runtime=self.runtime)

    @staticmethod
    def _validate(state):
        if (type(state) is not dict or set(state) != {'schema', 'revision', 'messages', 'versions', 'runs'}
                or state['schema'] != 'societybox-research-chat-v1'
                or type(state['revision']) is not int or state['revision'] < 0
                or any(type(state[key]) is not list for key in ('messages', 'versions', 'runs'))):
            raise ValueError('CHAT_STATE_INVALID')
        for i, version in enumerate(state['versions'], 1):
            if (set(version) != {'id', 'spec', 'review_sha256', 'authorization'} or version['id'] != 'v'+str(i)
                    or version['authorization'] != 'NOT_GRANTED'
                    or version['spec'].get('family') not in FAMILIES
                    or not re.fullmatch('[a-f0-9]{64}', version['review_sha256'])):
                raise ValueError('CHAT_STATE_INVALID')
        known = {v['id'] for v in state['versions']}
        if any(type(m) is not dict or set(m) != {'id', 'text', 'kind', 'version_id', 'run_id'}
               or m['id'] != i or m['kind'] not in ('researcher', 'proposal', 'calculation')
               or type(m['text']) is not str or len(m['text']) > 4000
               for i, m in enumerate(state['messages'], 1)):
            raise ValueError('CHAT_STATE_INVALID')
        if any(set(r) != {'id', 'version_id', 'receipt_sha256'} or not _RUN.fullmatch(r['id'])
               or r['version_id'] not in known or not re.fullmatch('[a-f0-9]{64}', r['receipt_sha256'])
               for r in state['runs']):
            raise ValueError('CHAT_STATE_INVALID')
        # A nonsecret state digest is not an authority for double-publication:
        # every run identity must bind to exactly one version and receipt.
        if len({row['id'] for row in state['runs']}) != len(state['runs']):
            raise ValueError('CHAT_STATE_INVALID')

    def _persist(self, state, expected=None):
        self._validate(state)
        raw = engine._bytes({'state': state, 'sha256': engine._sha(engine._bytes(state))})
        if len(raw) > STATE_LIMIT: raise ValueError('CHAT_WORKSPACE_BYTE_BUDGET')
        path = _safe(self.root / 'state.json')
        fd, name = tempfile.mkstemp(prefix='.chat-state-', dir=self.root)
        try:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(raw); stream.flush(); os.fsync(stream.fileno())
            if expected is not None:
                if engine._read(_safe(path), STATE_LIMIT) != expected:
                    raise ValueError('CHAT_STATE_CHANGED_DURING_COMMIT')
            elif path.exists():
                raise ValueError('CHAT_INITIAL_STATE_ALREADY_EXISTS')
            checkpoint()
            os.replace(name, path)
        finally:
            if os.path.exists(name): os.unlink(name)

    def _edit(self, expected_revision):
        if self._lock is None: raise ValueError('CHAT_WORKSPACE_CLOSED')
        expected = engine._bytes({'state': self.state, 'sha256': engine._sha(engine._bytes(self.state))})
        if engine._read(_safe(self.root/'state.json'), STATE_LIMIT) != expected:
            raise ValueError('CHAT_STATE_CHANGED_OUTSIDE_SESSION')
        if type(expected_revision) is not int or expected_revision != self.state['revision']:
            raise ValueError('CHAT_REVISION_CONFLICT')
        # Rows are immutable once appended: message/propose/execute only append
        # new records. Copy the three lists, not every nested historical study
        # and chat row. A failed commit must not change the original state.
        state = dict(self.state)
        for key in ('messages', 'versions', 'runs'):
            state[key] = list(self.state[key])
        state['revision'] += 1
        return state

    def _commit(self, state):
        expected = engine._bytes({'state': self.state, 'sha256': engine._sha(engine._bytes(self.state))})
        self._persist(state, expected); self.state = state
        return self.snapshot()

    @staticmethod
    def _append(state, text, kind='researcher', version_id=None, run_id=None):
        state['messages'].append({'id': len(state['messages'])+1, 'text': text,
                                  'kind': kind, 'version_id': version_id, 'run_id': run_id})

    @_synchronized
    def snapshot(self, *, message_offset=None, version_offset=None, run_offset=None, page_size=40):
        for offset in (message_offset, version_offset, run_offset):
            _page(0 if offset is None else offset, page_size)
        result = {'schema': self.state['schema'], 'revision': self.state['revision'],
                  'llm': 'NO_LLM', 'external_transfer': 'DISABLED',
                  'latest_question': next((m['text'][:2000] for m in reversed(self.state['messages'])
                                           if m['kind'] == 'researcher'), ''),
                  'current_version_id': self.state['versions'][-1]['id'] if self.state['versions'] else None}
        for key, offset in (('messages', message_offset), ('versions', version_offset), ('runs', run_offset)):
            count = len(self.state[key]); start = max(0, count-page_size) if offset is None else offset
            result[key] = deepcopy(self.state[key][start:start+page_size])
            result[key[:-1]+'_count'] = count
            result[key[:-1]+'_offset'] = start
        return result

    @_synchronized
    def message(self, text, *, expected_revision):
        state = self._edit(expected_revision)
        if type(text) is not str or not text.strip() or len(text) > 4000:
            raise ValueError('CHAT_MESSAGE_INVALID')
        self._append(state, text)
        return self._commit(state)

    @_synchronized
    def propose(self, spec, *, expected_revision):
        state = self._edit(expected_revision)
        if type(spec) is not dict or type(spec.get('family')) is not str or spec['family'] not in FAMILIES:
            raise ValueError('CHAT_FAMILY_REQUIRES_OTHER_REVIEWED_ENTRY')
        checked = self._review(spec)  # validates but does not calculate
        version_id = 'v'+str(len(state['versions'])+1)
        state['versions'].append({'id': version_id, 'spec': deepcopy(spec),
                                  'review_sha256': checked['review_sha256'], 'authorization': 'NOT_GRANTED'})
        self._append(state, spec['question'], 'proposal', version_id)
        return self._commit(state)

    @_synchronized
    def prepare_execution(self, version_id, acknowledgment, *, expected_revision, confirmed):
        if self._recovery_jobs:raise ValueError('CHAT_RECOVERY_REQUIRED')
        self._edit(expected_revision)
        if confirmed is not True: raise ValueError('CHAT_EXPLICIT_CONFIRMATION_REQUIRED')
        if not self.state['versions'] or version_id != self.state['versions'][-1]['id']:
            raise ValueError('CHAT_STALE_VERSION')
        version = self.state['versions'][-1]; spec = version['spec']
        checked = self._review(spec)
        if checked['review_sha256'] != version['review_sha256']:
            raise ValueError('CHAT_STALE_CODE_REVIEW')
        if acknowledgment != checked['review_sha256']:
            raise ValueError('ENGINE_STUDY_ACKNOWLEDGMENT_REQUIRED')
        return deepcopy(version)

    def execute(self, version_id, acknowledgment, *, expected_revision, confirmed, operation_deadline=None):
        version=self.prepare_execution(version_id,acknowledgment,expected_revision=expected_revision,confirmed=confirmed)
        return self.execute_prepared(version,operation_deadline=operation_deadline)

    def execute_prepared(self, version, *, operation_deadline=None, cancel_event=None, attempt_id=None):
        if not self._execution_lock.acquire(blocking=False):raise ValueError('CHAT_EXECUTION_BUSY')
        try:
            with self._state_lock:
                if self._recovery_jobs:raise ValueError('CHAT_RECOVERY_REQUIRED')
                if not self.state['versions'] or version!=self.state['versions'][-1]:raise ValueError('CHAT_STALE_VERSION')
                self._edit(self.state['revision'])
                spec=deepcopy(version['spec']);checked=self._review(spec)
                if checked['review_sha256']!=version['review_sha256']:raise ValueError('CHAT_STALE_CODE_REVIEW')
            return self._execute_prepared(version,spec,checked,operation_deadline,cancel_event,attempt_id)
        finally:self._execution_lock.release()

    def _execute_prepared(self, version, spec, checked, operation_deadline, cancel_event, attempt_id):
        version_id=version['id'];acknowledgment=version['review_sha256']
        run_id = uuid.uuid4().hex if attempt_id is None else attempt_id
        if type(run_id) is not str or not _RUN.fullmatch(run_id):raise ValueError('CHAT_ATTEMPT_ID_INVALID')
        def deadline():
            if operation_deadline is not None and time.monotonic() >= operation_deadline:
                raise ValueError('CHAT_OPERATION_EXPIRED')
        deadline()
        attempts=_safe(self.root/'attempts');attempts.mkdir(exist_ok=True)
        attempt=_safe(attempts/run_id);attempt.mkdir()
        bundle=_safe(attempt/'bundle');published=_safe(self.root/'runs'/run_id)
        def record(name,status):
            value={'schema':'societybox-chat-attempt-v1','attempt_id':run_id,'version_id':version_id,
                   'review_sha256':checked['review_sha256'],'status':status,
                   'publication':'RECORDED' if status=='COMPLETED' else 'NOT_RECORDED'}
            with _safe(attempt/name).open('xb') as stream:stream.write(engine._bytes(value))
        record('started.json','RUNNING')
        try:
            with deadline_budget(operation_deadline,cancel=cancel_event):
                receipt=engine.run(spec,bundle,acknowledgment,runtime=self.runtime,sources=self.sources)
                engine.verify(bundle,runtime=self.runtime,sources=self.sources)
                deadline();checkpoint()
                # Both checked absolute paths stay inside this owned workspace.
                root=self.root.resolve()
                if not all(_safe(p).resolve().is_relative_to(root) for p in (bundle,published)):
                    raise ValueError('CHAT_PUBLICATION_PATH_INVALID')
                with self._state_lock:
                    checkpoint()
                    state=self._edit(self.state['revision'])
                    bundle.rename(published)
                    state['runs'].append({'id':run_id,'version_id':version_id,
                                          'receipt_sha256':engine._sha(engine._bytes(receipt))})
                    self._append(state,'RECOMPUTED · NO_LLM · CAUSAL_AND_EXTERNAL_VALIDITY_NOT_ESTABLISHED',
                                 'calculation',version_id,run_id)
                    try:result=self._commit(state)
                    except BaseException:
                        _safe(published).rename(_safe(bundle))
                        raise
        except BaseException as error:
            expired=(operation_deadline is not None and time.monotonic()>=operation_deadline)
            record('failed.json','CANCELLED' if str(error)=='OPERATION_CANCELLED' else
                   'EXPIRED' if expired or str(error)=='OPERATION_DEADLINE_EXCEEDED' else 'FAILED')
            raise
        # The state/run binding is authoritative even if a diagnostic receipt
        # cannot be written after successful publication (for example disk full).
        try:record('completed.json','COMPLETED')
        except OSError:pass
        return result

    def _verified(self, run_id):
        with self._state_lock:
            if self._recovery_jobs:raise ValueError('CHAT_RECOVERY_REQUIRED')
        if not self._execution_lock.acquire(blocking=False):raise ValueError('CHAT_EXECUTION_BUSY')
        try:return self._verified_locked(run_id)
        finally:self._execution_lock.release()

    def _verified_locked(self, run_id):
        checkpoint()
        if self.runtime is not None:_safe(self.runtime)
        if self.sources is not None:_safe(self.sources)
        if type(run_id) is not str or not _RUN.fullmatch(run_id): raise ValueError('CHAT_RUN_INVALID')
        entry = next((r for r in self.state['runs'] if r['id'] == run_id), None)
        if entry is None: raise ValueError('CHAT_RUN_UNKNOWN')
        bundle = _safe(self.root/'runs'/run_id)
        for path in bundle.iterdir(): _safe(path)
        verification = engine.verify(bundle, runtime=self.runtime, sources=self.sources)
        blobs = {name: engine._read(_safe(bundle/name), engine.ARTIFACT_LIMIT)
                 for name in verification['artifacts']}
        if any(engine._sha(raw) != verification['artifacts'][name] for name, raw in blobs.items()):
            raise ValueError('CHAT_RUN_CHANGED_AFTER_VERIFY')
        if engine._sha(blobs['receipt.json']) != entry['receipt_sha256']:
            raise ValueError('CHAT_RUN_RECEIPT_BINDING')
        version = next(v for v in self.state['versions'] if v['id'] == entry['version_id'])
        if blobs['study.json'] != engine._bytes(version['spec']): raise ValueError('CHAT_RUN_VERSION_BINDING')
        checkpoint()
        return verification, blobs

    def read_run(self, run_id, locale, *, evidence_offset=0, page_size=30):
        if locale not in engine.LOCALES: raise ValueError('CHAT_LOCALE_INVALID')
        _page(evidence_offset, page_size)
        verification, blobs = self._verified(run_id)
        receipt = engine._parse(blobs['receipt.json']); result = receipt['result']
        spec = engine._parse(blobs['study.json'])
        run_record = next(r for r in self.state['runs'] if r['id'] == run_id)
        parameters=spec['request'].get('parameters', {})
        data=(parameters.get('data') or parameters.get('screen', {}).get('study') or parameters)
        evidence = result.get('evidence', [])
        evidence = [evidence] if type(evidence) is dict else evidence
        rows = [('Evidence', x) for x in evidence]
        rows += [('Unknown', x) for x in result.get('unknowns', [])]
        rows += [('Obligation', x) for x in result.get('coverage', result.get('native_coverage', {})).get('obligations', [])]
        vintage_context = {}
        if spec['family'] == 'forecast_validation' and 'revision_history' in result:
            vintage_context['revision_history'] = result['revision_history']
        if spec['family'] == 'forecast_validation' and 'preprocessing' in parameters:
            vintage_context['preprocessing'] = parameters['preprocessing']
        if spec['family'] == 'vintage_comparison':
            vintage_context['revision_snapshot'] = {
                'availability_mode': parameters['availability_mode'],
                'earlier_as_of': parameters['earlier_as_of'], 'later_as_of': parameters['later_as_of'],
                'ledger_sha256': result['ledger_sha256'], 'series_metadata': parameters['ledger']['series'],
                'timestamp_authentication': 'NOT_ESTABLISHED', 'historical_researcher_access': 'NOT_ESTABLISHED'}
        return {'verification': verification, 'chart': receipt['chart'],
                'result': {k:v for k,v in result.items() if k not in ('evidence','unknowns','obligations','coverage')},
                'authorization': 'EXPLICIT_DIGEST_ACKNOWLEDGED_IDENTITY_NOT_ATTESTED',
                'source_context': {'version_id': run_record['version_id'], 'run_id': run_id,
                    'source': spec['source'], 'rights': spec['rights'], 'question': spec['question'],
                    'declared_source_sha256': data.get('source_declared_sha256', 'NOT_ESTABLISHED'),
                    'study_sha256': receipt['review']['binding']['study_sha256'], 'release_vintage': 'NOT_ESTABLISHED',
                    **vintage_context},
                'html': blobs['review-'+locale+'.html'].decode('utf-8'),
                'evidence': {'items': [{'kind': kind, 'record': rec} for kind, rec in rows[evidence_offset:evidence_offset+page_size]],
                             'total': len(rows), 'offset': evidence_offset},
                'artifacts': sorted(blobs), 'source_authentication': 'NOT_ESTABLISHED',
                'release_vintage': 'NOT_ESTABLISHED'}

    def download(self, run_id, artifact):
        _, blobs = self._verified(run_id)
        if artifact == 'bundle.zip':
            stream = io.BytesIO()
            with zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED) as archive:
                for name, raw in blobs.items(): archive.writestr(name, raw)
            return 'application/zip', stream.getvalue()
        if artifact not in blobs: raise ValueError('CHAT_ARTIFACT_NOT_EXPOSED')
        return 'application/octet-stream', blobs[artifact]
