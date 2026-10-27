"""
Structured JSON logging setup for FastAPI serving.
"""

from contextvars import ContextVar
from datetime import datetime, timezone
import json
import logging
import sys
from typing import Any, Dict, Optional

current_request_id: ContextVar[Optional[str]] = ContextVar("current_request_id", default=None)


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        req_id = current_request_id.get() or getattr(record, "request_id", None)
        if req_id:
            log_entry["request_id"] = req_id

        for key in ("endpoint", "method", "status_code", "latency_ms", "latency_seconds", "error_category"):
            val = getattr(record, key, None)
            if val is not None:
                log_entry[key] = val

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


def setup_structured_logging(log_level: str = "INFO", log_format: str = "json") -> None:
    """Configures application-wide logging format."""
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    if log_format.lower() == "json":
        handler.setFormatter(JSONFormatter())
    else:
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(name)s: %(message)s")
        handler.setFormatter(formatter)

    root_logger.addHandler(handler)
