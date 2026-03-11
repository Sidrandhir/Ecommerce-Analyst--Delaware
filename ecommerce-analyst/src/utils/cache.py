"""
Disk-based cache for LLM responses.
Reduces API costs by caching identical queries for CACHE_TTL_SECONDS.
"""
import hashlib
import json
import os
from functools import wraps
from typing import Any, Callable, Optional

import diskcache

from src.config import CACHE_DIR, CACHE_ENABLED, CACHE_TTL_SECONDS
from src.utils.logger import get_logger

logger = get_logger(__name__)

_cache: Optional[diskcache.Cache] = None


def _get_cache() -> diskcache.Cache:
    global _cache
    if _cache is None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        _cache = diskcache.Cache(CACHE_DIR)
    return _cache


def make_cache_key(prefix: str, *args, **kwargs) -> str:
    payload = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return f"{prefix}:{digest}"


def cached_llm_call(prefix: str = "llm"):
    """Decorator that caches the return value of any function."""
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs) -> Any:
            if not CACHE_ENABLED:
                return fn(*args, **kwargs)

            key = make_cache_key(prefix, *args, **kwargs)
            cache = _get_cache()

            if key in cache:
                logger.debug(f"Cache HIT: {key}")
                return cache[key]

            result = fn(*args, **kwargs)
            cache.set(key, result, expire=CACHE_TTL_SECONDS)
            logger.debug(f"Cache SET: {key}")
            return result
        return wrapper
    return decorator


def get_cached(key: str) -> Optional[Any]:
    if not CACHE_ENABLED:
        return None
    try:
        return _get_cache().get(key)
    except Exception:
        return None


def set_cached(key: str, value: Any) -> None:
    if not CACHE_ENABLED:
        return
    try:
        _get_cache().set(key, value, expire=CACHE_TTL_SECONDS)
    except Exception as e:
        logger.warning(f"Cache write failed: {e}")


def clear_cache() -> None:
    _get_cache().clear()
    logger.info("Cache cleared.")
