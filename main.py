"""
main.py — Entry point. Initialises DB, assets, scheduler, and FastAPI server.

CHANGE: Added Windows asyncio [WinError 10054] suppression.
  ConnectionResetError from _ProactorBasePipeTransport._call_connection_lost
  is a known harmless Windows-specific asyncio teardown artefact that fires
  whenever an outgoing HTTP connection is closed by the remote host while the
  proactor event loop is draining.  It does NOT indicate a real error; it
  spammed the log on every Pollinations.ai / Groq API call.  Filtered here.
"""
import sys
import asyncio

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
import logging
import os
import shutil
import subprocess
import sys
import urllib.request

import uvicorn
from apscheduler.schedulers.background import BackgroundScheduler

import config
from database.db import init_db, recover_stuck_runs

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ─── Suppress Windows asyncio [WinError 10054] noise ─────────────────────────
# _ProactorBasePipeTransport raises ConnectionResetError when a remote HTTP
# server closes the connection while the Windows proactor is draining the pipe.
# This is harmless (the request already succeeded or timed out as expected) but
# logs as ERROR in asyncio's exception handler, filling the log with red noise.
if sys.platform == "win32":
    import asyncio

    _orig_exc_handler = None

    def _win32_asyncio_exception_handler(loop, context):
        exc = context.get("exception")
        if isinstance(exc, ConnectionResetError) and exc.winerror == 10054:
            return  # silently swallow — harmless Windows socket teardown
        if _orig_exc_handler:
            _orig_exc_handler(loop, context)
        else:
            loop.default_exception_handler(context)

    # Patch the running loop's handler after uvicorn starts; also patch the
    # policy so any new loop created by threads inherits the filter.
    _asyncio_filter_installed = False

    def _install_win32_asyncio_filter():
        global _orig_exc_handler, _asyncio_filter_installed
        if _asyncio_filter_installed:
            return
        try:
            loop = asyncio.get_event_loop()
            _orig_exc_handler = loop.get_exception_handler()
            loop.set_exception_handler(_win32_asyncio_exception_handler)
            _asyncio_filter_installed = True
            logger.info("Windows asyncio WinError-10054 filter installed.")
        except Exception as e:
            logger.warning(f"Could not install asyncio filter: {e}")


def check_ffmpeg() -> bool:
    """Verify that FFmpeg and FFprobe are on PATH. Hard-fail if missing."""
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            logger.error(
                f"'{tool}' not found in PATH.\n"
                "Install FFmpeg from https://ffmpeg.org/download.html and add to PATH.\n"
                "Then restart the server."
            )
            return False
        try:
            result = subprocess.run(
                [tool, "-version"],
                capture_output=True, text=True, timeout=10,
            )
            version_line = result.stdout.splitlines()[0] if result.stdout else "unknown"
            logger.info(f"{tool}: {version_line[:80]}")
        except Exception as e:
            logger.warning(f"{tool} version check failed: {e}")
    return True


def download_font_if_missing():
    """Download a free Impact-like font if not present in assets/."""
    os.makedirs(config.ASSETS_DIR, exist_ok=True)
    font_path = config.THUMBNAIL_FONT_PATH

    if os.path.exists(font_path) and os.path.getsize(font_path) > 1000:
        logger.info(f"Font OK: {font_path}")
        return

    logger.info("Impact.ttf not found — downloading Oswald-Bold from Google Fonts…")
    fallback = config.FALLBACK_FONT_PATH
    if os.path.exists(fallback) and os.path.getsize(fallback) > 1000:
        logger.info(f"Fallback font exists: {fallback}")
        return

    try:
        url = (
            "https://github.com/google/fonts/raw/main/ofl/oswald/"
            "Oswald%5Bwght%5D.ttf"
        )
        urllib.request.urlretrieve(url, fallback)
        logger.info(f"Downloaded font to {fallback}")
    except Exception as e:
        logger.warning(f"Font download failed: {e}. Pillow default font will be used.")


def ensure_output_dirs():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.ASSETS_DIR, exist_ok=True)
    logger.info(
        f"Dirs: output={config.OUTPUT_DIR}  assets={config.ASSETS_DIR}"
    )


def scheduled_pipeline_run():
    """APScheduler job — starts a fresh pipeline run."""
    try:
        from graph.pipeline import run_pipeline
        import database.db as db
        run_data = db.create_run(config.NICHE)
        logger.info(f"Scheduled run started: {run_data['id']}")
        run_pipeline(niche=config.NICHE, run_id=run_data["id"])
    except Exception as e:
        logger.error(f"Scheduled pipeline run failed: {e}")


def start_scheduler() -> BackgroundScheduler:
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        scheduled_pipeline_run,
        trigger="interval",
        hours=config.SCHEDULER_INTERVAL_HOURS,
        id="pipeline_auto_run",
        max_instances=1,
        coalesce=True,
    )
    scheduler.start()
    logger.info(f"Scheduler started — pipeline every {config.SCHEDULER_INTERVAL_HOURS}h")
    return scheduler


def main():
    logger.info("=" * 62)
    logger.info("  YouTube Agent Pipeline — Starting Up")
    logger.info("=" * 62)

    # 1. Directories
    ensure_output_dirs()

    # 2. FFmpeg check (hard requirement)
    if not check_ffmpeg():
        logger.error("FFmpeg missing — cannot assemble videos. Exiting.")
        sys.exit(1)

    # 3. Database init
    logger.info("Initialising SQLite database...")
    init_db()

    # 4. Crash recovery — mark RUNNING runs as FAILED so Resume buttons appear
    logger.info("Running crash recovery check...")
    recover_stuck_runs()

    # 5. Assets (fonts)
    logger.info("Checking assets...")
    download_font_if_missing()

    # 6. API key validation
    if not config.GROQ_API_KEY:
        logger.error("GROQ_API_KEY is required. Add it to .env and restart.")
        sys.exit(1)

    for key, name in [
        (config.YOUTUBE_API_KEY,     "YOUTUBE_API_KEY (optional — trend data)"),
        (config.PEXELS_API_KEY,      "PEXELS_API_KEY  (optional — stock video/images)"),
        (config.HUGGINGFACE_API_KEY, "HUGGINGFACE_API_KEY (optional — HF models)"),
    ]:
        if not key:
            logger.warning(f"Not set: {name}")

    # 7. Scheduler
    scheduler = start_scheduler()

    # 8. FastAPI server
    from dashboard.server import app

    logger.info("=" * 62)
    logger.info(f"  Dashboard : http://localhost:{config.DASHBOARD_PORT}")
    logger.info(f"  Niche     : {config.NICHE}")
    logger.info(f"  Length    : {config.VIDEO_LENGTH}")
    logger.info(f"  Voice     : {config.VOICE}")
    logger.info(f"  LLM       : {config.GROQ_MODEL}")
    logger.info(f"  Whisper   : {config.WHISPER_MODEL} (runs in child process)")
    logger.info("=" * 62)

    try:
        uvicorn.run(
            app,
            host=config.DASHBOARD_HOST,
            port=config.DASHBOARD_PORT,
            log_level="info",
        )
    finally:
        scheduler.shutdown()
        logger.info("Scheduler stopped.")

    # Install asyncio filter AFTER uvicorn creates its event loop
    if sys.platform == "win32":
        _install_win32_asyncio_filter()


if __name__ == "__main__":
    main()
