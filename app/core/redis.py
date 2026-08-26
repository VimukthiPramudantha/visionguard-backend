# app/core/redis.py
import os
import logging
import json
import time
import fnmatch
from typing import Any, Optional
import redis.asyncio as redis
from redis.exceptions import ConnectionError as RedisConnectionError
from dotenv import load_dotenv

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

logger = logging.getLogger("visionguard.redis")

_redis_client: Optional[redis.Redis] = None
_redis_available = True

_memory_cache = {}

try:
    _redis_client = redis.from_url(REDIS_URL, decode_responses=True)
except Exception as e:
    logger.info(f"Failed to initialize Redis client: {e}. Falling back to in-memory caching.")
    _redis_client = None
    _redis_available = False

def _get_memory_cache(key: str) -> Optional[Any]:
    if key in _memory_cache:
        item = _memory_cache[key]
        if time.time() < item["expires_at"]:
            return item["value"]
        else:
            del _memory_cache[key]
    return None

def _set_memory_cache(key: str, value: Any, expire_seconds: int = 3600):
    _memory_cache[key] = {
        "value": value,
        "expires_at": time.time() + expire_seconds
    }

def _delete_memory_cache(key: str):
    _memory_cache.pop(key, None)

def _clear_memory_cache_pattern(pattern: str):
    keys_to_delete = [k for k in _memory_cache.keys() if fnmatch.fnmatch(k, pattern)]
    for k in keys_to_delete:
        _memory_cache.pop(k, None)

async def get_cache(key: str) -> Optional[Any]:
    """Retrieve value from cache. Falls back to in-memory cache if Redis is unavailable."""
    global _redis_available
    if _redis_client is None or not _redis_available:
        return _get_memory_cache(key)
    try:
        val = await _redis_client.get(key)
        if val:
            return json.loads(val)
    except (RedisConnectionError, OSError) as e:
        logger.info(f"Redis connection failed. Falling back to in-memory caching. Error: {e}")
        _redis_available = False
        return _get_memory_cache(key)
    except Exception as e:
        logger.warning(f"Redis get failed for key {key}: {e}")
    return None

async def set_cache(key: str, value: Any, expire_seconds: int = 3600) -> bool:
    """Store value in cache. Falls back to in-memory cache if Redis is unavailable."""
    global _redis_available
    if _redis_client is None or not _redis_available:
        _set_memory_cache(key, value, expire_seconds)
        return True
    try:
        serialized = json.dumps(value)
        await _redis_client.set(key, serialized, ex=expire_seconds)
        return True
    except (RedisConnectionError, OSError) as e:
        logger.info(f"Redis connection failed. Falling back to in-memory caching. Error: {e}")
        _redis_available = False
        _set_memory_cache(key, value, expire_seconds)
        return True
    except Exception as e:
        logger.warning(f"Redis set failed for key {key}: {e}")
    return False

async def delete_cache(key: str) -> bool:
    """Delete a key from cache. Falls back to in-memory cache if Redis is unavailable."""
    global _redis_available
    if _redis_client is None or not _redis_available:
        _delete_memory_cache(key)
        return True
    try:
        await _redis_client.delete(key)
        return True
    except (RedisConnectionError, OSError) as e:
        logger.info(f"Redis connection failed. Falling back to in-memory caching. Error: {e}")
        _redis_available = False
        _delete_memory_cache(key)
        return True
    except Exception as e:
        logger.warning(f"Redis delete failed for key {key}: {e}")
    return False

async def clear_cache_pattern(pattern: str) -> bool:
    """Delete keys matching pattern. Falls back to in-memory cache if Redis is unavailable."""
    global _redis_available
    if _redis_client is None or not _redis_available:
        _clear_memory_cache_pattern(pattern)
        return True
    try:
        keys = await _redis_client.keys(pattern)
        if keys:
            await _redis_client.delete(*keys)
        return True
    except (RedisConnectionError, OSError) as e:
        logger.info(f"Redis connection failed. Falling back to in-memory caching. Error: {e}")
        _redis_available = False
        _clear_memory_cache_pattern(pattern)
        return True
    except Exception as e:
        logger.warning(f"Redis clear pattern failed for pattern {pattern}: {e}")
    return False

def get_cache_status() -> dict:
    """Return diagnostic information about the current cache state."""
    import time
    now = time.time()
    valid_keys = [k for k, v in _memory_cache.items() if v["expires_at"] > now]
    return {
        "redis_url": REDIS_URL,
        "redis_client_initialized": _redis_client is not None,
        "redis_available": _redis_available,
        "cache_backend": "redis" if (_redis_client is not None and _redis_available) else "in-memory",
        "in_memory_cache_key_count": len(valid_keys),
        "in_memory_cache_keys": valid_keys,
    }
