"""
Two-layer logging:
Layer 1 → Better Stack: streams every log line to cloud in real time
Layer 2 → Local file: rotated daily, uploaded to S3 on crash
"""

import sys
import os
import signal
import atexit
import boto3
from datetime import datetime
from pathlib import Path
from loguru import logger
from logtail import LogtailHandler
from dotenv import load_dotenv

load_dotenv()

# Constants
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"illuminate_{datetime.now().strftime('%Y-%m-%d')}.log"

BETTERSTACK_TOKEN = os.getenv("BETTERSTACK_TOKEN")
AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")


def upload_logs_to_s3(reason: str = "manual"):
    """
    Uploads today's log file to S3.
    Called on crash, SIGTERM, or manual trigger.
    """
    if not AWS_S3_BUCKET:
        logger.warning("S3 bucket not configured — skipping crash log upload")
        return

    if not LOG_FILE.exists():
        logger.warning("No log file found to upload")
        return

    try:
        s3 = boto3.client(
            "s3",
            region_name=AWS_REGION,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
        )

        # S3 key: crash-logs/2026-06-25_crash_reason.log
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        s3_key = f"crash-logs/{timestamp}_{reason}.log"

        s3.upload_file(str(LOG_FILE), AWS_S3_BUCKET, s3_key)

        # Print directly — logger may be broken at this point
        print(f"✅ Crash log uploaded to s3://{AWS_S3_BUCKET}/{s3_key}")

    except Exception as e:
        print(f"❌ Failed to upload crash log to S3: {e}")


def handle_signal(signum, frame):
    """Handles SIGTERM and SIGINT (Ctrl+C or server shutdown)."""
    signal_name = signal.Signals(signum).name
    logger.warning(f"Signal received: {signal_name} — uploading logs before exit")
    upload_logs_to_s3(reason=f"signal_{signal_name}")
    sys.exit(0)


def handle_unhandled_exception(exc_type, exc_value, exc_traceback):
    """
    Global exception handler — catches anything that slips through.
    This is your last line of defence before the process dies.
    """
    if issubclass(exc_type, KeyboardInterrupt):
        # Don't treat Ctrl+C as a crash
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.critical(
        "UNHANDLED EXCEPTION — server is crashing",
        exc_info=(exc_type, exc_value, exc_traceback)
    )
    upload_logs_to_s3(reason="unhandled_exception")


def setup_logging():
    """
    Call this once at app startup.
    Sets up all log handlers and crash handlers.
    """

    # Remove loguru's default handler
    logger.remove()

    # Inject request_id="system" into all records that don't have one
    logger.configure(extra={"request_id": "system"})

    # 1. Console — clean, coloured, readable
    logger.add(
        sys.stdout,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{extra[request_id]}</cyan> | "
            "{message}"
        ),
        level="INFO",
        colorize=True
    )

    # 2. Local file — full details, rotated daily
    logger.add(
        str(LOG_FILE),
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{extra[request_id]} | "
            "{module}:{function}:{line} | "
            "{message}"
        ),
        level="DEBUG",
        rotation="00:00",       # New file at midnight
        retention="7 days",     # Keep 7 days locally
        compression="zip",      # Compress old logs
        encoding="utf-8"
    )

    # 3. Better Stack — real-time cloud streaming
    if BETTERSTACK_TOKEN:
        logtail_handler = LogtailHandler(source_token=BETTERSTACK_TOKEN)
        logger.add(
            logtail_handler,
            format="{message}",
            level="INFO"
        )
        logger.info("Better Stack logging enabled")
    else:
        logger.warning("BETTERSTACK_TOKEN not set — cloud logging disabled")

    # 4. Register crash handlers
    sys.excepthook = handle_unhandled_exception
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    atexit.register(lambda: logger.info("Server process exiting normally"))

    logger.info("Logging setup complete")


# Context var for request ID — explained in middleware section
import contextvars
request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="system"
)