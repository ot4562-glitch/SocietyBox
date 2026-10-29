"""One active, explicitly approved local study job; no retries or auto-resume."""
from bisect import bisect_right
from copy import deepcopy
import os
import re
import tempfile
import threading
import time
import uuid

from . import engine_study as engine
from .execution_control import safe_local_path

_ID=re.compile(r'[a-f0-9]{32}\Z')


class ResearchJobs:
    TERMINAL=frozenset(('SUCCEEDED','FAILED','CANCELLED','EXPIRED','RECOVERY_REQUIRED'))
    STATES=TERMINAL|{'PENDING','RUNNING','CANCEL_REQUESTED'}

    def __init__(self,chat,*,deadline):
        self.chat=chat;self.deadline=deadline;self._lock=threading.RLock();self._closed=False
        self._changed=threading.Condition(self._lock)
        self._rows={};self._threads={};self._events={}
        self.root=safe_local_path(chat.root/'jobs');self.root.mkdir(exist_ok=True)
        self._rows=self.read_records(self.root)
        # Job creation times do not change when statuses update. Keep the
        # sorted order once; polling a small page should not sort all history.
        self._ordered_ids=sorted(self._rows,key=lambda key:self._rows[key]['created_ns'],reverse=True)
        self._ordered_times=[-self._rows[key]['created_ns'] for key in self._ordered_ids]
        published={r['id']:r for r in chat.state['runs']}
        reviewed={v['id']:v['review_sha256'] for v in chat.state['versions']}
        for row in list(self._rows.values()):
            if row['status']=='SUCCEEDED':
                # A sealed metadata record alone cannot prove that a real run
                # was published. Do not claim SUCCEEDED after state rollback,
                # missing bundle, or mismatched version/review binding.
                recorded=published.get(row['id'])
                bundle=safe_local_path(chat.root/'runs'/row['id'])
                if (row['run_id'] != row['id'] or recorded is None
                        or recorded['version_id'] != row['version_id']
                        or reviewed.get(row['version_id']) != row['review_sha256']
                        or not bundle.is_dir()):
                    self._save(dict(row,status='RECOVERY_REQUIRED',
                                    reason='PUBLISHED_RUN_BINDING_NOT_OBSERVED',
                                    updated_ns=time.time_ns()))
            elif row['status'] not in self.TERMINAL:
                self._save(dict(row,status='RECOVERY_REQUIRED',
                                reason='PREVIOUS_SESSION_NOT_OBSERVED',
                                updated_ns=time.time_ns()))
            if self._rows[row['id']]['status']=='RECOVERY_REQUIRED':
                self.chat.require_recovery(row['id'])
        for key in self.chat._recovery_jobs-set(self._rows):
            started=engine._parse(engine._read(safe_local_path(chat.root/'attempts'/key/'started.json'),16384))
            if started.get('attempt_id')!=key or type(started.get('version_id')) is not str or type(started.get('review_sha256')) is not str:
                raise ValueError('CHAT_RECOVERY_ATTEMPT_INVALID')
            now=time.time_ns()
            self._save({'id':key,'version_id':started['version_id'],'review_sha256':started['review_sha256'],
                        'status':'RECOVERY_REQUIRED','run_id':None,'reason':'UNFINISHED_ATTEMPT_NOT_OBSERVED','created_ns':now,'updated_ns':now})

    @staticmethod
    def read_records(root):
        root=safe_local_path(root)
        if not root.exists():return {}
        rows={}
        total=0
        for path in root.iterdir():
            if path.suffix!='.json' or not _ID.fullmatch(path.stem):continue
            raw=engine._read(safe_local_path(path),16384);total+=len(raw)
            if total>16*1024*1024:raise ValueError('CHAT_JOB_METADATA_BUDGET')
            sealed=engine._parse(raw);row=sealed['job']
            if (type(row) is not dict or set(row)!={'id','version_id','review_sha256','status','run_id','reason','created_ns','updated_ns'}
                    or row['id']!=path.stem or row['status'] not in ResearchJobs.STATES
                    or type(row['version_id']) is not str or not re.fullmatch(r'v[1-9][0-9]{0,11}',row['version_id'])
                    or type(row['review_sha256']) is not str or not re.fullmatch('[a-f0-9]{64}',row['review_sha256'])
                    or any(type(row[k]) is not int or row[k]<0 for k in ('created_ns','updated_ns'))
                    or (row['run_id'] is not None and (type(row['run_id']) is not str or not _ID.fullmatch(row['run_id'])))
                    or (row['reason'] is not None and (type(row['reason']) is not str or len(row['reason'])>128))
                    or sealed.get('sha256')!=engine._sha(engine._bytes(row))):
                raise ValueError('CHAT_JOB_METADATA_INVALID')
            rows[row['id']]=row
        return rows

    @staticmethod
    def _bytes(row):return engine._bytes({'job':row,'sha256':engine._sha(engine._bytes(row))})

    def _remember(self,row):
        """Update one row and the immutable creation-order index atomically."""
        previous=self._rows.get(row['id'])
        if previous is None or previous['created_ns'] != row['created_ns']:
            if previous is not None:
                index=self._ordered_ids.index(row['id'])
                self._ordered_ids.pop(index)
                self._ordered_times.pop(index)
            stamp=-row['created_ns']
            index=bisect_right(self._ordered_times,stamp)
            self._ordered_times.insert(index,stamp)
            self._ordered_ids.insert(index,row['id'])
        self._rows[row['id']]=deepcopy(row)

    def _put(self,row):
        path=safe_local_path(self.root/(row['id']+'.json'));old=self._rows.get(row['id'])
        if (old is not None and engine._read(path,16384)!=self._bytes(old)) or (old is None and path.exists()):
            self._remember(dict(row,status='RECOVERY_REQUIRED',reason='JOB_METADATA_CHANGED'))
            self.chat.require_recovery(row['id'])
            event=self._events.get(row['id'])
            if event is not None:event.set()
            return False
        fd,name=tempfile.mkstemp(prefix='.job-',dir=self.root)
        try:
            with os.fdopen(fd,'wb') as stream:
                stream.write(self._bytes(row));stream.flush();os.fsync(stream.fileno())
            os.replace(name,path)
        finally:
            if os.path.exists(name):os.unlink(name)
        self._remember(row)
        with self._changed:self._changed.notify_all()
        return True

    def _save(self,row):
        try:return self._put(row)
        except (OSError,ValueError,KeyError,TypeError):
            self._remember(dict(row,status='RECOVERY_REQUIRED',reason='JOB_STATE_NOT_DURABLE'))
            self.chat.require_recovery(row['id'])
            event=self._events.get(row['id'])
            if event is not None:event.set()
            return False

    def get(self,key,*,wait=0):
        with self._lock:
            if type(key) is not str or not _ID.fullmatch(key) or key not in self._rows:raise ValueError('CHAT_JOB_UNKNOWN')
            if type(wait) is not int or not 0<=wait<=2:raise ValueError('CHAT_JOB_WAIT_INVALID')
            previous=self._rows[key]['status']
            if wait and previous not in self.TERMINAL:
                self._changed.wait_for(lambda:self._rows[key]['status']!=previous or self._closed,timeout=wait)
            worker=self._threads.get(key)
            if worker is not None and not worker.is_alive() and self._rows[key]['status'] not in self.TERMINAL:
                self._save(dict(self._rows[key],status='RECOVERY_REQUIRED',reason='CURRENT_THREAD_TERMINAL_WITHOUT_STATUS',updated_ns=time.time_ns()))
            return deepcopy(self._rows[key])

    def list(self,*,offset=0,size=40):
        if type(offset) is not int or offset<0 or type(size) is not int or not 1<=size<=100:raise ValueError('CHAT_JOB_PAGE_INVALID')
        with self._lock:
            return {'items':deepcopy([self._rows[key] for key in self._ordered_ids[offset:offset+size]]),
                    'total':len(self._ordered_ids),'offset':offset}

    def submit(self,version_id,acknowledgment,*,expected_revision,confirmed):
        with self._lock:
            if self._closed:raise ValueError('CHAT_JOBS_CLOSED')
            if any(t.is_alive() for t in self._threads.values()):raise ValueError('CHAT_EXECUTION_BUSY')
            version=self.chat.prepare_execution(version_id,acknowledgment,expected_revision=expected_revision,confirmed=confirmed)
            key=uuid.uuid4().hex;now=time.time_ns();event=threading.Event()
            row={'id':key,'version_id':version_id,'review_sha256':acknowledgment,'status':'PENDING',
                 'run_id':None,'reason':None,'created_ns':now,'updated_ns':now}
            self._events[key]=event
            if not self._save(row):return self.get(key)
            worker=threading.Thread(target=self._work,args=(key,version,event),name='societybox-study-job')
            self._threads[key]=worker
            try:worker.start()
            except RuntimeError:
                self._threads.pop(key,None)
                self._save(dict(row,status='FAILED',reason='JOB_THREAD_START_FAILED',updated_ns=time.time_ns()))
                return self.get(key)
            return deepcopy(row)

    def _work(self,key,version,event):
        with self._lock:
            if not self._save(dict(self._rows[key],status='RUNNING',updated_ns=time.time_ns())):return
        try:
            operation_deadline=min(self.deadline,time.monotonic()+600)
            self.chat.execute_prepared(version,operation_deadline=operation_deadline,
                                       cancel_event=event,attempt_id=key)
            status,reason,run_id='SUCCEEDED',None,key
        except Exception as error:
            status=('CANCELLED' if str(error)=='OPERATION_CANCELLED' else 'EXPIRED'
                    if str(error) in ('CHAT_OPERATION_EXPIRED','OPERATION_DEADLINE_EXCEEDED') or time.monotonic()>=operation_deadline else 'FAILED')
            reason='OPERATION_CANCELLED' if status=='CANCELLED' else 'OPERATION_EXPIRED' if status=='EXPIRED' else 'REVIEW_OR_EXECUTION_FAILED'
            run_id=None
        with self._lock:
            if self._rows[key]['status']!='RECOVERY_REQUIRED':
                self._save(dict(self._rows[key],status=status,reason=reason,run_id=run_id,updated_ns=time.time_ns()))

    def cancel(self,key):
        with self._lock:
            row=self.get(key)
            if row['status'] in self.TERMINAL:return row
            if row['status']=='CANCEL_REQUESTED':return row
            event=self._events.get(key)
            if event is None:raise ValueError('CHAT_JOB_NOT_OWNED_BY_THIS_SESSION')
            event.set()
            self._save(dict(row,status='CANCEL_REQUESTED',updated_ns=time.time_ns()))
            return self.get(key)

    def recover(self,key,acknowledgment):
        with self._lock:
            row=self.get(key)
            if acknowledgment!=key or row['status']!='RECOVERY_REQUIRED':raise ValueError('CHAT_RECOVERY_ACK_REQUIRED')
            worker=self._threads.get(key)
            if worker is None or worker.is_alive():raise ValueError('CHAT_RECOVERY_EXTERNAL_OBSERVATION_REQUIRED')
            if not self.chat._execution_lock.acquire(blocking=False):raise ValueError('CHAT_EXECUTION_BUSY')
            try:
                with self.chat._state_lock:published=any(r['id']==key for r in self.chat.state['runs'])
                if published:self.chat._verified_locked(key)
                disk=self.read_records(self.root).get(key)
                if disk is None or any(disk[k]!=row[k] for k in ('id','version_id','review_sha256')):
                    raise ValueError('CHAT_RECOVERY_METADATA_CHANGED')
                self._remember(disk)
                resolved=dict(row,status='SUCCEEDED' if published else 'FAILED',run_id=key if published else None,
                              reason='CURRENT_SESSION_STOP_OBSERVED',updated_ns=time.time_ns())
                if self._save(resolved):self.chat.clear_recovery(key)
                return self.get(key)
            finally:self.chat._execution_lock.release()

    def close(self):
        with self._lock:
            self._closed=True
            self._changed.notify_all()
            workers=list(self._threads.items())
            for key,worker in workers:
                if worker.is_alive():self._events[key].set()
        for _,worker in workers:worker.join(timeout=30)
        if any(worker.is_alive() for _,worker in workers):raise RuntimeError('CHAT_JOB_SHUTDOWN_INCOMPLETE')
