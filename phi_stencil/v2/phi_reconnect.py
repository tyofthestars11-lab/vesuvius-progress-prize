#!/usr/bin/env python3
"""
phi_reconnect.py -- Tyree's refinement: apply phi to the reconnecting itself.

The connection is a CONTINUOUS FLOW, not a binary connected/disconnected gate.
There are no hard stuck states. Five behaviors, always on:

  1. RESUME -- every transfer resumes from its last byte offset on any stall.
     Partial files are never discarded; HTTP Range requests pick up exactly
     where the bytes stopped.
  2. BACKOFF -- exponential backoff between retries (base * 2**attempt, capped,
     with jitter). A flapping link never spins hot.
  3. HEARTBEAT -- every 60 s the pool logs bytes_done / bytes_total per active
     transfer and in aggregate, plus current rate. A stall is VISIBLE, never
     silent.
  4. FALLBACK -- if one extraction path stalls 3 times, new items route to the
     next (alternate) path: e.g. tar-member extraction -> per-chunk HTTP fetch
     -> byte-range streaming of the archive. In-flight transfers keep flowing;
     the pipeline never waits on the dead path.
  5. FLOW -- transfers run in a worker pool; the consumer takes completions as
     they arrive (as_completed). One stalled transfer NEVER blocks the rest.

Kept beside census.py / build_stencil.py so the reconnect behavior ships with
the phi-stencil v2 pipeline. Wire `log=` to the build log to keep one record.

Example:
    from phi_reconnect import FlowPool, FlowCtx, HttpFileSource, ByteRangeSource

    ctx = FlowCtx(workers=24, log=my_log)          # heartbeat every 60 s
    pool = FlowPool([HttpFileSource(url_for_chunk),   # primary path
                     ByteRangeSource(archive_url, ranges_for)], ctx=ctx)
    futs = [pool.submit(key, f'chunks/{key}.bin', total=2*1024*1024)
            for key in chunk_keys]
    for item, path, ok in pool.iter_completed(futs):
        ...  # completions arrive as they finish; stalled ones don't block
    pool.shutdown()
"""

import os
import re
import time
import math
import random
import tarfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None


# --------------------------------------------------------------------------
# 1+2. flow_download: resume-from-last-byte-offset + exponential backoff
# --------------------------------------------------------------------------

class StallDetected(Exception):
    """Raised when a transfer makes no progress (cut, timeout, short stream)."""


def _backoff(attempt, base, cap, log, state, err):
    wait = min(cap, base * (2 ** (attempt - 1))) * (0.75 + random.random() * 0.5)
    if log:
        log('[reconnect] %s: stall #%d (%s); backoff %.1fs, resume at byte %d'
            % (state.name, state.stalls, err, wait, state.done))
    time.sleep(wait)


class TransferState:
    """Per-transfer counters shared between the worker and the heartbeat."""

    def __init__(self, name, path, total=None, on_stall=None):
        self.name = name
        self.path = path
        self.total = total          # bytes; None if unknown
        self.done = 0               # bytes written (resumed offsets count)
        self.stalls = 0
        self.attempts = 0
        self.started_at = time.time()
        self.last_progress_at = time.time()
        self.finished = False
        self.ok = False
        self._lock = threading.Lock()
        self._on_stall = on_stall   # called (outside the lock) per stall

    def add(self, n):
        with self._lock:
            self.done += n
            self.last_progress_at = time.time()

    def note_stall(self, n=1):
        """Record stall(s) and notify the pool *live* (drives 3-stall fallback)."""
        cb = None
        with self._lock:
            self.stalls += n
            cb = self._on_stall
        if cb:
            for _ in range(n):
                cb()

    def snapshot(self):
        with self._lock:
            return dict(name=self.name, total=self.total, done=self.done,
                        stalls=self.stalls, attempts=self.attempts,
                        finished=self.finished, ok=self.ok,
                        idle=time.time() - self.last_progress_at)


def flow_download(url, path, state, session=None, stall_timeout=30,
                  max_attempts=14, backoff_base=2.0, backoff_cap=120.0,
                  chunk_size=1 << 20, log=None):
    """
    Download `url` -> `path` as a continuous flow.

    On any stall (no bytes for `stall_timeout` seconds, a cut connection, a
    short stream) the transfer backs off exponentially and resumes from the
    last byte offset via HTTP Range. Returns True on success.
    """
    if requests is None:
        raise RuntimeError('requests is required for flow_download')
    sess = session or requests.Session()
    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        state.attempts = attempt
        offset = os.path.getsize(path) if os.path.exists(path) else 0
        headers = {'Range': 'bytes=%d-' % offset} if offset > 0 else {}
        try:
            with sess.get(url, headers=headers, stream=True,
                          timeout=(10, stall_timeout)) as r:
                if r.status_code == 416:
                    # Range unsatisfiable: file already complete.
                    state.finished = True
                    state.ok = True
                    return True
                if r.status_code == 206 and offset > 0:
                    mode = 'ab'
                elif r.status_code == 200:
                    if offset > 0:
                        offset = 0  # server ignored Range: restart from zero
                    mode = 'wb'
                else:
                    raise StallDetected('http %d' % r.status_code)
                if state.total is None:
                    cr = r.headers.get('Content-Range', '')
                    m = re.search(r'/(\d+)\s*$', cr)
                    if m:
                        state.total = int(m.group(1))
                    elif r.headers.get('Content-Length'):
                        state.total = int(r.headers['Content-Length']) + offset
                # Account for bytes already on disk before this attempt.
                with state._lock:
                    if state.done < offset:
                        state.done = offset
                        state.last_progress_at = time.time()
                with open(path, mode) as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if not chunk:
                            continue
                        f.write(chunk)
                        state.add(len(chunk))
                have = os.path.getsize(path)
                if state.total and have >= state.total:
                    state.finished = True
                    state.ok = True
                    return True
                if state.total is None:
                    state.finished = True  # stream ended; total unknown
                    state.ok = True
                    return True
                raise StallDetected('short stream (%d/%d bytes)'
                                    % (have, state.total))
        except StallDetected as e:
            state.note_stall()
            _backoff(attempt, backoff_base, backoff_cap, log, state, e)
        except Exception as e:  # cut, DNS, refused, read timeout...
            state.note_stall()
            _backoff(attempt, backoff_base, backoff_cap, log, state,
                     '%s: %s' % (type(e).__name__, e))
    state.finished = True
    state.ok = False
    if log:
        log('[reconnect] %s: FAILED after %d attempts (%d stalls)'
            % (state.name, attempt, state.stalls))
    return False


# --------------------------------------------------------------------------
# 3. Heartbeat: bytes done / total every 60 s -- stalls visible, never silent
# --------------------------------------------------------------------------

class Heartbeat(threading.Thread):
    def __init__(self, pool, interval=60, log=None):
        super().__init__(daemon=True)
        self.pool = pool
        self.interval = interval
        self.log = log or (lambda m: print(m, flush=True))
        self._stop = threading.Event()

    def run(self):
        while not self._stop.wait(self.interval):
            try:
                self.pool.beat(self.log)
            except Exception:
                pass

    def stop(self):
        self._stop.set()


def _fmt(n):
    if n is None:
        return '?'
    for unit in ('B', 'KB', 'MB', 'GB'):
        if n < 1024 or unit == 'GB':
            return '%.1f%s' % (n, unit)
        n /= 1024.0


# --------------------------------------------------------------------------
# 4+5. Sources (extraction paths) + FlowPool (fallback + non-blocking flow)
# --------------------------------------------------------------------------

class Source:
    """One extraction path. Subclass and implement fetch()."""
    name = 'source'

    def fetch(self, item, path, state, ctx):
        """Fetch `item` -> `path`, updating `state`. Return True on success."""
        raise NotImplementedError


class HttpFileSource(Source):
    """Primary path for the v2 build: one URL per chunk/file, resumed via Range."""

    def __init__(self, url_for, name='http'):
        self.url_for = url_for
        self.name = name

    def fetch(self, item, path, state, ctx):
        return flow_download(self.url_for(item), path, state,
                             session=ctx.session, stall_timeout=ctx.stall_timeout,
                             max_attempts=ctx.max_attempts,
                             backoff_base=ctx.backoff_base,
                             backoff_cap=ctx.backoff_cap, log=ctx.log)


class ByteRangeSource(Source):
    """Alternate path: stream explicit byte ranges of one big remote file.

    ranges_for(item) -> list of (start, end) inclusive byte ranges, fetched
    with Range requests and concatenated into `path`. Used when per-file
    fetches stall but the archive itself is reachable.
    """

    def __init__(self, url, ranges_for, name='byterange'):
        self.url = url
        self.ranges_for = ranges_for
        self.name = name

    def fetch(self, item, path, state, ctx):
        ranges = self.ranges_for(item)
        if state.total is None:
            state.total = sum(e - s + 1 for s, e in ranges)
        tmp = path + '.part'
        try:
            with open(tmp, 'wb') as out:
                for s, e in ranges:
                    sub = TransferState('%s:%s[%d-%d]' % (self.name, item, s, e),
                                        tmp, total=e - s + 1)
                    ok = flow_download(self.url, tmp + '.seg', sub,
                                       session=ctx.session,
                                       stall_timeout=ctx.stall_timeout,
                                       max_attempts=ctx.max_attempts,
                                       log=None)  # pool-level logging only
                    state.note_stall(sub.stalls)
                    if not ok:
                        state.finished = True
                        state.ok = False
                        return False
                    with open(tmp + '.seg', 'rb') as f:
                        while True:
                            b = f.read(1 << 20)
                            if not b:
                                break
                            out.write(b)
                            state.add(len(b))
                    os.remove(tmp + '.seg')
            os.replace(tmp, path)
            state.finished = True
            state.ok = True
            return True
        except Exception as e:
            state.note_stall()
            if ctx.log:
                ctx.log('[reconnect] %s: %s' % (state.name, e))
            state.finished = True
            state.ok = False
            return False


class TarMemberSource(Source):
    """Primary path when a local archive holds the files: stream the .tar.gz
    ONCE in a background thread and extract only wanted members as the stream
    reaches them. Per-item fetch() waits on its member with stall-aware
    polling; if the stream makes no progress for `stall_timeout`, the item
    records a stall (driving the pool's 3-stall fallback) while the flow
    continues for everyone else."""

    def __init__(self, archive, name='tar'):
        self.archive = archive
        self.name = name
        self._lock = threading.Lock()
        self._wanted = {}          # member name -> [outpath, event, state]
        self._streamer_started = False
        self._streamer_done = False
        self._streamer_progress = time.time()

    def _ensure_streamer(self, ctx):
        with self._lock:
            if self._streamer_started:
                return
            self._streamer_started = True
            self._streamer_progress = time.time()
        t = threading.Thread(target=self._stream, args=(ctx,), daemon=True)
        t.start()

    def _stream(self, ctx):
        try:
            with tarfile.open(self.archive, 'r|gz') as tf:
                for member in tf:
                    with self._lock:
                        entry = self._wanted.get(member.name)
                    if entry is None:
                        continue
                    outpath, ev, st = entry
                    try:
                        f = tf.extractfile(member)
                        if f is None:
                            raise StallDetected('not a regular file')
                        if st.total is None and member.size:
                            st.total = member.size
                        with open(outpath, 'wb') as out:
                            while True:
                                b = f.read(1 << 20)
                                if not b:
                                    break
                                out.write(b)
                                st.add(len(b))
                        st.ok = True
                    except Exception as e:
                        st.ok = False
                        if ctx.log:
                            ctx.log('[reconnect] tar member %s: %s'
                                    % (member.name, e))
                    finally:
                        st.finished = True
                        ev.set()
                    with self._lock:
                        self._streamer_progress = time.time()
        except Exception as e:
            if ctx.log:
                ctx.log('[reconnect] tar stream %s: %s: %s'
                        % (self.archive, type(e).__name__, e))
            with self._lock:
                for outpath, ev, st in self._wanted.values():
                    if not ev.is_set():
                        st.finished = True
                        st.ok = False
                        ev.set()
        finally:
            with self._lock:
                self._streamer_done = True

    def fetch(self, item, path, state, ctx):
        self._ensure_streamer(ctx)
        ev = threading.Event()
        with self._lock:
            self._wanted[item] = [path, ev, state]
        idle_since = time.time()
        while True:
            if ev.wait(timeout=5):
                return state.ok
            with self._lock:
                prog = self._streamer_progress
                sdone = self._streamer_done
            if sdone:
                # Stream exhausted and member never appeared: fail fast.
                # (Absent member is not a stall -- nothing to resume.)
                state.finished = True
                state.ok = False
                return False
            if time.time() - prog > ctx.stall_timeout:
                # Stream alive but not advancing: count a stall for fallback,
                # keep waiting -- the flow continues for everyone else.
                if time.time() - idle_since > ctx.stall_timeout:
                    state.note_stall()
                    idle_since = time.time()
                    if ctx.log:
                        ctx.log('[reconnect] %s: tar stream idle %ds '
                                '(stall #%d)' % (state.name, ctx.stall_timeout,
                                                 state.stalls))


class FlowCtx:
    """Shared knobs for the pool."""

    def __init__(self, workers=24, stall_timeout=30, heartbeat_interval=60,
                 max_stalls_before_fallback=3, max_attempts=14,
                 backoff_base=2.0, backoff_cap=120.0, log=None):
        self.workers = workers
        self.stall_timeout = stall_timeout
        self.heartbeat_interval = heartbeat_interval
        self.max_stalls_before_fallback = max_stalls_before_fallback
        self.max_attempts = max_attempts
        self.backoff_base = backoff_base
        self.backoff_cap = backoff_cap
        self.log = log or (lambda m: print(m, flush=True))
        self.session = requests.Session() if requests else None


class FlowPool:
    """Worker pool over an ordered list of extraction paths.

    New items always route to the currently active path. Stall accounting is
    LIVE: the moment a path's stall count reaches `max_stalls_before_fallback`
    (3), the pool logs the fallback and routes new items to the next path;
    in-flight transfers keep their own retries on the old path -- nothing
    blocks. Consumers take completions via iter_completed() as they arrive.
    """

    def __init__(self, sources, ctx=None):
        if not sources:
            raise ValueError('FlowPool needs at least one source')
        self.sources = list(sources)
        self.ctx = ctx or FlowCtx()
        self.active = 0
        self.source_stalls = {s.name: 0 for s in self.sources}
        self.states = []  # list: items may be unhashable (e.g. [cz,cy,cx])
        self._lock = threading.Lock()
        self.exec = ThreadPoolExecutor(max_workers=self.ctx.workers)
        self.hb = Heartbeat(self, interval=self.ctx.heartbeat_interval,
                            log=self.ctx.log)
        self.hb.start()

    def submit(self, item, path, total=None):
        """Submit one item; returns a Future of (item, path, ok).

        The item routes to the currently active path. If that path is
        mid-fallback (3 stalls seen live), new items already go to the
        alternate path -- a stalled transfer never blocks the pipeline.
        """
        with self._lock:
            src = self.sources[self.active]
            src_name = src.name
        st = TransferState('%s:%s' % (src_name, item), path, total,
                           on_stall=lambda: self._note_stall(src_name))
        with self._lock:
            self.states.append(st)
        return self.exec.submit(self._run, src, item, path, st)

    def _note_stall(self, src_name):
        """Live stall accounting: after 3 stalls on a path, fall back.

        Called from worker threads the moment a stall is recorded -- not at
        transfer completion -- so new items route to the alternate path while
        the stalled transfer keeps its own retries flowing.
        """
        with self._lock:
            self.source_stalls[src_name] = self.source_stalls.get(src_name, 0) + 1
            n = self.source_stalls[src_name]
            if (n >= self.ctx.max_stalls_before_fallback
                    and self.active < len(self.sources) - 1):
                old = self.sources[self.active].name
                self.active += 1
                new = self.sources[self.active]
                self.ctx.log(
                    '[reconnect] path "%s" stalled %dx -> FALLBACK to "%s" '
                    'for new items; in-flight transfers keep flowing'
                    % (old, n, new.name))

    def _run(self, src, item, path, st):
        try:
            ok = src.fetch(item, path, st, self.ctx)
        except Exception as e:
            if self.ctx.log:
                self.ctx.log('[reconnect] %s: %s: %s'
                             % (st.name, type(e).__name__, e))
            ok = False
            st.finished = True
        return (item, path, bool(ok))

    def beat(self, log):
        """60 s heartbeat: bytes done / total, per transfer + aggregate."""
        with self._lock:
            states = list(self.states)
        snaps = [st.snapshot() for st in states]
        live = [s for s in snaps if not s['finished']]
        done = sum(s['done'] for s in snaps)
        total = sum(s['total'] for s in snaps if s['total'])
        total_known = all(s['total'] for s in snaps) and snaps
        n_ok = sum(1 for s in snaps if s['finished'] and s['ok'])
        n_fail = sum(1 for s in snaps if s['finished'] and not s['ok'])
        agg = '%s / %s' % (_fmt(done), _fmt(total) if total_known else '?')
        log('[heartbeat] flow %s done | active %d ok %d failed %d | path %s' %
            (agg, len(live), n_ok, n_fail, self.sources[self.active].name))
        for s in live:
            pct = ('%.1f%%' % (100.0 * s['done'] / s['total'])
                   if s['total'] else '')
            log('[heartbeat]   %-40s %s / %s %s stalls=%d idle=%ds' %
                (s['name'][:40], _fmt(s['done']), _fmt(s['total']), pct,
                 s['stalls'], int(s['idle'])))

    def iter_completed(self, futures):
        """Yield (item, path, ok) as transfers finish -- never blocks on one."""
        for fut in as_completed(futures):
            yield fut.result()

    def shutdown(self):
        self.hb.stop()
        self.exec.shutdown(wait=True)
