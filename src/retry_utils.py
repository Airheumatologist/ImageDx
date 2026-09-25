"""Retry helpers for external API calls."""

import logging
import time
from typing import Callable, Optional, TypeVar


T = TypeVar("T")

# Common transient HTTP status codes.
RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


def _status_code_from_exception(exc: Exception) -> Optional[int]:
    """Extract HTTP status code from common exception types."""
    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        return status_code

    response = getattr(exc, "response", None)
    if response is not None:
        return getattr(response, "status_code", None)

    return None


def is_retryable_exception(exc: Exception) -> bool:
    """Return True for transient/network/rate-limit errors."""
    status_code = _status_code_from_exception(exc)
    if status_code is not None:
        return status_code in RETRYABLE_STATUS_CODES

    # Fallback for network-level errors without status codes.
    error_text = f"{type(exc).__name__}: {exc}".lower()
    transient_markers = ("timeout", "connection", "temporar", "rate limit", "too many requests")
    return any(marker in error_text for marker in transient_markers)
def retry_with_exponential_backoff(
    fn: Callable[[], T],
    *,
    max_attempts: int,
    base_delay: float,
    operation_name: str,
    logger: logging.Logger,
) -> T:
    """
    Retry function calls with exponential backoff for transient errors.

    Args:
        fn: Function to execute.
        max_attempts: Total attempts (initial try + retries).
        base_delay: Base delay in seconds. Delay grows as base_delay * 2^attempt.
        operation_name: Operation label for logs.
        logger: Logger instance.
    """
    max_attempts = max(1, max_attempts)
    base_delay = max(0.0, base_delay)

    for attempt_idx in range(max_attempts):
        try:
            return fn()
        except Exception as exc:
            is_last_attempt = attempt_idx == max_attempts - 1
            if is_last_attempt or not is_retryable_exception(exc):
                raise

            delay_seconds = base_delay * (2 ** attempt_idx)
            logger.warning(
                "%s failed with retryable error (%s). Retrying in %.1fs (%d/%d)...",
                operation_name,
                exc,
                delay_seconds,
                attempt_idx + 1,
                max_attempts - 1,
            )
            time.sleep(delay_seconds)

    # Should never be reached due to early returns/raises above.
    raise RuntimeError(f"{operation_name} retry loop exited unexpectedly")
