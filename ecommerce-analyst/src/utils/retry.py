"""
Retry + graceful-degradation wrappers for all external API calls.
Uses exponential back-off so transient rate-limits don't crash the pipeline.
"""
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)
import logging
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Retry on any Exception (covers network errors, rate limits, quota exceeded)
llm_retry = retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)


def safe_llm_call(fn, *args, fallback=None, **kwargs):
    """
    Call fn(*args, **kwargs) with retry logic.
    If all retries fail, return `fallback` instead of crashing.
    """
    try:
        return llm_retry(fn)(*args, **kwargs)
    except Exception as e:
        logger.error(f"LLM call failed after retries: {e}")
        if fallback is not None:
            return fallback
        return (
            "⚠️ The AI service is temporarily unavailable. "
            "Please check your API key and try again."
        )
