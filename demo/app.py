"""Run with uvicorn demo.app:app --host 0.0.0.0 --port 7860 --workers 1."""
import asyncio
from contextlib import asynccontextmanager
import json
import logging
import os
from pathlib import Path
import tempfile
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.datastructures import UploadFile

from demo.video import InvalidVideo, MAX_BYTES, MAX_SECONDS, inspect_video

LOG = logging.getLogger(__name__)


def create_app(processor=None, max_bytes=MAX_BYTES, ttl=3600, max_results=20):
    jobs = {}
    busy = False
    pending = set()
    storage = None
    engine = processor

    def purge(reserve=0):
        now = time.time()
        for jid, job in list(jobs.items()):
            if job['state'] in ('done', 'error') and now - job['finished'] > ttl:
                (storage / f'{jid}.json').unlink(missing_ok=True)
                del jobs[jid]
        finished = sorted((j['finished'], jid) for jid, j in jobs.items() if 'finished' in j)
        for _, jid in finished[:max(0, len(finished) - max_results + reserve)]:
            (storage / f'{jid}.json').unlink(missing_ok=True)
            del jobs[jid]

    @asynccontextmanager
    async def lifespan(app):
        nonlocal storage, engine
        with tempfile.TemporaryDirectory(prefix='zlt-demo-') as root:
            storage = Path(root)
            if engine is None:
                from demo.engine import Engine
                engine = await asyncio.to_thread(Engine)
            app.state.ready = True
            async def reap():
                while True:
                    await asyncio.sleep(60)
                    purge()
            reaper = asyncio.create_task(reap())
            try:
                yield
            finally:
                app.state.ready = False
                reaper.cancel()
                await asyncio.gather(reaper, return_exceptions=True)
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)

    app = FastAPI(title='Zeroth Law Traffic Demo', lifespan=lifespan)
    origins = [o.strip() for o in os.getenv('DEMO_CORS_ORIGINS', 'http://localhost:5173,http://localhost:4173').split(',') if o.strip()]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=['GET', 'POST'],
                       allow_headers=['Content-Type'], allow_credentials=False)

    @app.get('/health')
    async def health():
        return dict(ready=getattr(app.state, 'ready', False), busy=busy,
                    max_seconds=MAX_SECONDS, max_bytes=max_bytes, result_ttl_seconds=ttl)

    async def run_job(jid, directory, path, filename):
        nonlocal busy
        job = jobs[jid]
        loop = asyncio.get_running_loop()
        def update(stage, fraction):
            if job['state'] != 'running':
                return
            ranges = {'decoding': (0, .02), 'detecting': (.02, .88), 'events': (.90, .08), 'risk': (.98, .02)}
            base, share = ranges[stage]
            job.update(stage=stage, progress=round(min(.999, base + share * fraction), 4))
        def progress(stage, fraction):
            loop.call_soon_threadsafe(update, stage, fraction)
        try:
            result = await asyncio.to_thread(engine.process, path, progress)
            result['replay']['video'] = filename
            target = storage / f'{jid}.json'
            await asyncio.to_thread(target.write_text, json.dumps(result, allow_nan=False), encoding='utf-8')
            job.update(state='done', stage='done', progress=1.0,
                       result_url=f'/jobs/{jid}/result', metadata=result['metadata'])
        except Exception as exc:
            LOG.exception('Demo job %s failed', jid)
            code = exc.code if isinstance(exc, InvalidVideo) else 'processing_failed'
            message = str(exc) if isinstance(exc, InvalidVideo) else 'Processing failed. Try a shorter, valid MP4 video.'
            job.update(state='error', stage='error', error=dict(code=code, message=message))
            (storage / f'{jid}.json').unlink(missing_ok=True)
        finally:
            directory.cleanup()
            job['finished'] = time.time()
            busy = False

    @app.post('/jobs', status_code=202)
    async def submit(request: Request):
        nonlocal busy
        if busy:
            raise HTTPException(409, detail=dict(code='busy', message='One job is already running. Try again after it finishes.'), headers={'Retry-After': '10'})
        purge(reserve=1)
        busy = True  # Reserve before parsing multipart, including concurrent uploads.
        directory = tempfile.TemporaryDirectory(prefix='upload-', dir=storage)
        handed_off = False
        original_receive = request._receive
        seen = 0
        async def bounded_receive():
            nonlocal seen
            try:
                chunk = await asyncio.wait_for(original_receive(), timeout=30)
            except TimeoutError:
                raise HTTPException(408, detail=dict(code='upload_timeout', message='Upload stalled for 30 seconds.'))
            seen += len(chunk.get('body', b''))
            if seen > max_bytes + 1024**2:
                raise HTTPException(413, detail=dict(code='too_large', message='Upload exceeds the byte limit.'))
            return chunk
        request._receive = bounded_receive
        try:
            length = request.headers.get('content-length')
            if length:
                try:
                    size = int(length)
                except ValueError:
                    raise HTTPException(400, detail=dict(code='invalid_length', message='Invalid Content-Length.'))
                if size > max_bytes + 1024**2:
                    raise HTTPException(413, detail=dict(code='too_large', message='Upload exceeds the byte limit.'))
            async with request.form(max_files=1, max_fields=0) as form:
                video = form.get('video')
                if not isinstance(video, UploadFile) or len(form) != 1:
                    raise HTTPException(422, detail=dict(code='missing_video', message='Use multipart field video with one MP4 file.'))
                filename = (video.filename or '').replace('\\', '/').split('/')[-1]
                if Path(filename).suffix.lower() != '.mp4':
                    raise HTTPException(415, detail=dict(code='wrong_format', message='Only .mp4 uploads are accepted.'))
                path = Path(directory.name) / 'upload.mp4'
                total = 0
                with path.open('wb') as output:
                    while chunk := await video.read(1024**2):
                        total += len(chunk)
                        if total > max_bytes:
                            raise HTTPException(413, detail=dict(code='too_large', message='Upload exceeds the byte limit.'))
                        await asyncio.to_thread(output.write, chunk)
                try:
                    meta, first = await asyncio.to_thread(inspect_video, path)
                    del first
                except InvalidVideo as exc:
                    raise HTTPException(422, detail=dict(code=exc.code, message=str(exc))) from exc
            jid = uuid.uuid4().hex
            jobs[jid] = dict(job_id=jid, state='running', stage='decoding', progress=0.,
                             created=time.time(), duration=meta['duration'])
            task = asyncio.create_task(run_job(jid, directory, path, filename))
            pending.add(task)
            task.add_done_callback(pending.discard)
            handed_off = True
            return dict(job_id=jid, status_url=f'/jobs/{jid}', result_url=f'/jobs/{jid}/result')
        finally:
            if not handed_off:
                directory.cleanup()
                busy = False

    def job(jid):
        purge()
        if jid not in jobs:
            raise HTTPException(404, detail=dict(code='not_found', message='Unknown or expired job.'))
        return jobs[jid]

    @app.get('/jobs/{jid}')
    async def status(jid: str):
        return job(jid)

    @app.get('/jobs/{jid}/result')
    async def result(jid: str):
        info = job(jid)
        if info['state'] == 'error':
            raise HTTPException(422, detail=info['error'])
        if info['state'] != 'done':
            raise HTTPException(409, detail=dict(code='not_ready', message='Job is still running.'))
        return FileResponse(storage / f'{jid}.json', media_type='application/json')

    return app


app = create_app()
