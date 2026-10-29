"""Per-operation deadlines and local path checks; not an OS security sandbox."""
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import subprocess
import stat
import threading
import time

_DEADLINE=ContextVar('societybox_operation_deadline',default=None)
_CANCEL=ContextVar('societybox_operation_cancellation',default=())


def checkpoint():
    if any(event.is_set() for event in _CANCEL.get()):raise ValueError('OPERATION_CANCELLED')
    deadline=_DEADLINE.get()
    if deadline is not None and time.monotonic()>=deadline:
        raise ValueError('OPERATION_DEADLINE_EXCEEDED')


def remaining_timeout(maximum):
    checkpoint()
    deadline=_DEADLINE.get()
    return maximum if deadline is None else min(maximum,max(.000001,deadline-time.monotonic()))


@contextmanager
def deadline_budget(deadline, *, cancel=None):
    previous=_DEADLINE.get()
    effective=previous if deadline is None else deadline if previous is None else min(previous,deadline)
    token=_DEADLINE.set(effective)
    events=_CANCEL.get()
    cancellation=_CANCEL.set(events+((cancel,) if cancel is not None and cancel not in events else ()))
    try:
        checkpoint()
        yield
    finally:
        _CANCEL.reset(cancellation)
        _DEADLINE.reset(token)


def run_worker(args, *, input=None, timeout, **kwargs):
    """Kill and reap a fixed trusted worker on cancellation or timeout."""
    if _DEADLINE.get() is None and not _CANCEL.get():
        return subprocess.run(args,input=input,timeout=timeout,**kwargs)
    def guard():
        try:checkpoint()
        except ValueError as error:
            if str(error)=='OPERATION_DEADLINE_EXCEEDED':raise subprocess.TimeoutExpired(args,timeout) from None
            raise
    guard()
    if not kwargs.pop('capture_output',False):raise ValueError('WORKER_CAPTURE_REQUIRED')
    process=subprocess.Popen(args,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,**kwargs)
    done=threading.Event();observed={}
    def communicate():
        try:observed['streams']=process.communicate(input)
        except BaseException as error:observed['error']=error
        finally:done.set()
    reader=threading.Thread(target=communicate,name='societybox-worker-io')
    reader.start();end=time.monotonic()+timeout
    try:
        while not done.wait(.025):
            guard()
            if time.monotonic()>=end:raise subprocess.TimeoutExpired(args,timeout)
        guard()
        if 'error' in observed:raise observed['error']
        stdout,stderr=observed['streams']
        return subprocess.CompletedProcess(args,process.returncode,stdout,stderr)
    finally:
        if process.poll() is None:process.kill()
        reader.join()
        process.wait()


def safe_local_path(path):
    path=Path(path).absolute()
    components=(path,*path.parents)
    if any(part.name.lower() in ('private','windows-api-credentials') for part in components):
        raise ValueError('CHAT_PROTECTED_PATH')
    for part in reversed(components):
        try:info=part.lstat()
        except FileNotFoundError:continue
        if stat.S_ISLNK(info.st_mode) or getattr(info,'st_file_attributes',0)&1024:
            raise ValueError('CHAT_REPARSE_PATH')
    return path
