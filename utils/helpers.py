"""
utils/helpers.py
----------------
Shared utilities used across all collector modules.
"""

import json
import csv
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from loguru import logger
import pandas as pd


# ── Logging setup ────────────────────────────────────────────────────────────
def setup_logger(log_dir: Path = Path("logs"), level: str = "INFO") -> None:
    """Configure loguru with rotating file + stderr sinks."""
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_dir / "acquisition_{time:YYYY-MM-DD}.log",
        rotation="00:00",       # new file at midnight
        retention="14 days",
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{line} | {message}",
        enqueue=True,           # thread-safe
    )
    logger.info("Logger initialised")


# ── Timestamps ───────────────────────────────────────────────────────────────
def utc_now() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def epoch_ms() -> int:
    """Current time in milliseconds since Unix epoch."""
    return int(datetime.now(timezone.utc).timestamp() * 1000)


# ── File I/O ─────────────────────────────────────────────────────────────────
def save_json(data: Any, path: Path) -> None:
    """Write data as formatted JSON, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    logger.debug(f"Saved JSON → {path}")


def append_csv(row: Dict, path: Path, fieldnames: Optional[List[str]] = None) -> None:
    """Append a single dict row to a CSV file (creates with header if new)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    fields = fieldnames or list(row.keys())

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def load_json(path: Path) -> Any:
    """Load and return a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parquet_append(df: pd.DataFrame, path: Path) -> None:
    """Append DataFrame to a Parquet file (reads existing + concat + write)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_parquet(path)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_parquet(path, index=False, engine="pyarrow")
    logger.debug(f"Parquet updated → {path} ({len(df)} rows total)")


# ── Data validation ───────────────────────────────────────────────────────────
def validate_record(record: Dict, required_keys: List[str]) -> bool:
    """Return True if all required keys are present and non-null."""
    for key in required_keys:
        if key not in record or record[key] is None:
            logger.warning(f"Validation failed – missing '{key}' in record: {record}")
            return False
    return True


def checksum(data: str) -> str:
    """MD5 checksum for deduplication."""
    return hashlib.md5(data.encode()).hexdigest()


# ── Kafka publisher (optional) ────────────────────────────────────────────────
def publish_to_kafka(topic: str, payload: Dict, bootstrap_servers: List[str]) -> bool:
    """
    Publish a dict payload to a Kafka topic.
    Returns True on success, False if Kafka is unavailable.
    """
    try:
        from kafka import KafkaProducer
        producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
            acks="all",
            retries=3,
        )
        future = producer.send(topic, value=payload)
        future.get(timeout=10)
        producer.close()
        logger.debug(f"Kafka → {topic}: {list(payload.keys())}")
        return True
    except Exception as exc:
        logger.warning(f"Kafka publish skipped ({exc.__class__.__name__}): {exc}")
        return False


# ── Rate-limit aware HTTP ─────────────────────────────────────────────────────
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def make_session(retries: int = 3, backoff: float = 1.0) -> requests.Session:
    """
    Return a requests.Session with automatic retry + exponential backoff.
    Retries on 429, 500, 502, 503, 504.
    """
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def safe_get(url: str, params: Dict = None, headers: Dict = None,
             timeout: int = 15) -> Optional[Dict]:
    """GET with retry session; returns parsed JSON or None on failure."""
    session = make_session()
    try:
        resp = session.get(url, params=params, headers=headers, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP {e.response.status_code} for {url}: {e}")
    except requests.exceptions.ConnectionError:
        logger.error(f"Connection error for {url}")
    except requests.exceptions.Timeout:
        logger.error(f"Timeout for {url}")
    except Exception as e:
        logger.error(f"Unexpected error for {url}: {e}")
    return None