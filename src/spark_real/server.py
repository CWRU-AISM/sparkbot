"""
SPARK real-robot web server (FastAPI).

REST + WebSocket API for the frontend: camera capture and streaming, SAM3
detection, plan generation, robot execution and pipeline status.

Usage:
    python -m spark_real.server --robot {ur10e,franka,bimanual} [--machine <host>] [--port 8888] [--no-robot]
"""

import contextlib
import logging
import os
import sys

# SAM3 weights are cached locally; without this, hf_hub pings huggingface.co
# on every start to recheck them. Export HF_HUB_OFFLINE=0 to fetch updates.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("spark_server")

import faulthandler  # noqa: E402
import threading  # noqa: E402

try:
    faulthandler.enable(file=sys.stderr)
except Exception as _fh_exc:  # noqa: BLE001 - never block boot on diagnostics
    logger.warning("faulthandler.enable failed: %s", _fh_exc)


def _log_thread_exception(args):
    logger.critical(
        "UNCAUGHT EXCEPTION in thread %r",
        getattr(args.thread, "name", "?"),
        exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
    )


threading.excepthook = _log_thread_exception

app = FastAPI(title="SPARK Real Pipeline", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Progress logging

from spark_real.routes import state


class _ProgressHandler(logging.Handler):
    """
    Captures pipeline log messages for the frontend progress endpoint.

    Stamps each record with the monotonic sequence number the frontend
    polls against, and with record.created (when the event occurred) rather
    than the handling time.
    """

    def emit(self, record):
        msg = self.format(record)
        with state.progress_lock:
            state.progress_seq += 1
            state.progress_log.append(
                {"seq": state.progress_seq, "ts": record.created, "msg": msg}
            )
            # Trim from the front. Safe now that clients track `seq`
            # rather than a list index.
            excess = len(state.progress_log) - state.PROGRESS_MAX
            if excess > 0:
                del state.progress_log[:excess]


# ATTACH AT THE PACKAGE ROOT, NOT PER-MODULE. A per-leaf allowlist rots
# whenever a module is split (each mixin takes getLogger(__name__)), and the
# symptom is a /api/progress buffer that never grows while a 133-second run
# executes in silence. Propagation from the "spark_real" root carries records
# from any spark_real.* logger, including ones not yet written. Measured cost
# on a 113-second pick-and-place: ~76 records, under one per second.
# "spark_server" is a separate root and needs its own.
for _logger_name in ["spark_real", "spark_server"]:
    _prog_handler = _ProgressHandler()
    _prog_handler.setLevel(logging.INFO)
    _prog_handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger(_logger_name).addHandler(_prog_handler)


# Register route modules

from spark_real.routes.core import router as core_router  # noqa: E402
from spark_real.routes.control import router as control_router  # noqa: E402
from spark_real.routes.calibration import router as calibration_router  # noqa: E402
from spark_real.routes.streaming import router as streaming_router  # noqa: E402
from spark_real.routes.episode import router as episode_router  # noqa: E402
from spark_real.routes.bt_library import router as bt_library_router  # noqa: E402
from spark_real.routes.perception import router as perception_router  # noqa: E402
from spark_real.routes.bimanual import router as bimanual_router  # noqa: E402

for _r in (
    core_router,
    control_router,
    calibration_router,
    streaming_router,
    episode_router,
    bt_library_router,
    perception_router,
    bimanual_router,
):
    app.include_router(_r)


import atexit  # noqa: E402
import signal  # noqa: E402
from pathlib import Path  # noqa: E402

import tyro  # noqa: E402

from spark_real.config import SparkConfig, load_profile, output_root  # noqa: E402
from spark_real.pipeline import SPARKRealPipeline, PipelineConfig  # noqa: E402

# One-shot latch, not a bool: the callers are a signal handler
# (SIGTERM/SIGINT/SIGHUP), uvicorn's on_event("shutdown"), and atexit, and a
# signal can interrupt the main thread between a check and a set, re-entering
# the whole teardown. Acquired once and never released, so every later caller
# (including a reentrant signal) returns immediately. A Lock held across the
# body would instead deadlock on that same reentry.
_shutdown_latch = threading.Lock()
_SHUTDOWN_BUDGET_S = 10.0  # hard deadline, after which we os._exit


def _release_devices(reason: str):
    """
    Idempotent cleanup of Kinects, RealSense, robot. Safe to call from
    signal handlers and atexit. Closing Kinects without this leaves the
    depth MCU mid-stream and bricks it for the next launch.

    Runs pipeline.shutdown() in a worker thread with a hard deadline.
    Past the deadline, we abandon whatever step is hung: the cameras' own
    close() ran first and each has its own internal watchdog, so they
    release before robot disconnect even gets a chance to wedge.
    """
    if not _shutdown_latch.acquire(blocking=False):
        return

    # BRAKE THE ARM FIRST, so Ctrl-C never exits with a movej still running.
    # One URScript send on the already-open socket,
    # bounded, and every failure swallowed: a stuck camera must never be able
    # to prevent the stop.
    try:
        _pipe = state.pipeline
        _rb = getattr(_pipe, "_robot", None) if _pipe is not None else None
        if _rb is not None and getattr(_rb, "SUPPORTS_URSCRIPT", False):
            _send = getattr(_rb, "_send_script", None)
            if callable(_send):
                _send("stopj(3.0)")
                logger.warning("shutdown: braked the arm (%s)", reason)
    except Exception as _exc:  # noqa: BLE001 - shutdown must continue regardless
        logger.warning("shutdown: brake failed: %s", _exc)

    # Then the stream/websocket capture pool. Its workers sit inside
    # capture_single_camera and would otherwise keep reading cameras that the
    # shutdown below is releasing. cancel_futures drops the queued backlog;
    # wait=False so a worker blocked in a camera read cannot consume the
    # shutdown budget on its own.
    try:
        from spark_real.routes.streaming import _STREAM_POOL

        _STREAM_POOL.shutdown(wait=False, cancel_futures=True)
    except Exception as _exc:  # noqa: BLE001
        logger.warning("shutdown: stream pool shutdown failed: %s", _exc)

    pipe = state.pipeline
    if pipe is None:
        return
    logger.info("releasing devices (%s)", reason)

    def _do():
        try:
            pipe.shutdown()
        except Exception as exc:
            logger.warning("pipeline shutdown raised: %s", exc)

    t = threading.Thread(target=_do, daemon=True, name="spark-shutdown")
    t.start()
    t.join(timeout=_SHUTDOWN_BUDGET_S)
    if t.is_alive():
        logger.warning(
            "shutdown exceeded %.0fs, proceeding to exit anyway. "
            "Cameras should already be released (per-device watchdogs); "
            "robot disconnect was likely the holdup.",
            _SHUTDOWN_BUDGET_S,
        )


@app.on_event("shutdown")
def _on_shutdown():
    """
    FastAPI/Uvicorn graceful path (no signal involved, or SIGTERM
    drained through Uvicorn's signal trap).
    """
    _release_devices("uvicorn shutdown")


def _signal_handler(signum, _frame):
    """
    Catch SIGTERM/SIGINT before Uvicorn's loop can stall on a busy
    pipeline. Calls shutdown synchronously then exits via os._exit so
    we never get stuck waiting for the event loop to drain. SIGKILL
    bypasses this, so never stop the server with `kill -9`.
    """
    name = signal.Signals(signum).name
    try:
        _release_devices(f"signal {name}")
    finally:
        os._exit(0)


signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGHUP, _signal_handler)
atexit.register(_release_devices, "atexit")


# Static files + frontend

FRONTEND_DIR = Path(__file__).parent / "frontend"
VIDEOS_DIR = output_root() / "videos"
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/videos", StaticFiles(directory=str(VIDEOS_DIR)), name="videos")
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    async def serve_frontend():
        return HTMLResponse(
            (FRONTEND_DIR / "index.html").read_text(),
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
            },
        )

else:

    @app.get("/")
    async def serve_placeholder():
        return HTMLResponse("<h1>SPARK Real Pipeline</h1><p>Frontend not built.</p>")


def main():
    cfg = tyro.cli(SparkConfig)
    # /api/initialize rebuilds the pipeline from the launched config.
    state.launch_config = cfg

    if not cfg.no_init:
        profile = load_profile(cfg)
        config = PipelineConfig(**profile.to_pipeline_kwargs())
        state.pipeline = SPARKRealPipeline(config, profile=profile)
        state.pipeline.initialize()

    # Serve WITHOUT uvicorn's signal handling: its graceful path waits for
    # in-flight requests, and a long /api/execute never returns, so Ctrl-C
    # would hang. Our handler releases the devices and exits instead.
    class _KeepOurSignalHandlers(uvicorn.Server):
        def capture_signals(self):
            return contextlib.nullcontext()

    _KeepOurSignalHandlers(uvicorn.Config(app, host=cfg.host, port=cfg.port)).run()


if __name__ == "__main__":
    main()
