# app/core/redis.py
import os
import logging
import json
from typing import Any, Optional
import redis.asyncio as redis
from redis.exceptions import ConnectionError as RedisConnectionError
from dotenv import load_dotenv

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

logger = logging.getLogger("visionguard.redis")

_redis_client: Optional[redis.Redis] = None
_redis_available = True

try:
    _redis_client = redis.from_url(REDIS_URL, decode_responses=True)
except Exception as e:
    logger.warning(f"Failed to initialize Redis client: {e}. Caching will be disabled.")
    _redis_client = None
    _redis_available = False

async def get_cache(key: str) -> Optional[Any]:
    """Retrieve value from cache. Returns None if key not found or error occurs."""
    global _redis_available
    if _redis_client is None or not _redis_available:
        return None
    try:
        val = await _redis_client.get(key)
        if val:
            return json.loads(val)
    except (RedisConnectionError, OSError) as e:
        logger.warning(f"Redis connection failed, disabling caching. get({key}) error: {e}")
        _redis_available = False
    except Exception as e:
        logger.warning(f"Redis get failed for key {key}: {e}")
    return None

async def set_cache(key: str, value: Any, expire_seconds: int = 3600) -> bool:
    """Store value in cache. Returns True on success, False on failure."""
    global _redis_available
    if _redis_client is None or not _redis_available:
        return False
    try:
        serialized = json.dumps(value)
        await _redis_client.set(key, serialized, ex=expire_seconds)
        return True
    except (RedisConnectionError, OSError) as e:
        logger.warning(f"Redis connection failed, disabling caching. set({key}) error: {e}")
        _redis_available = False
    except Exception as e:
        logger.warning(f"Redis set failed for key {key}: {e}")
    return False

async def delete_cache(key: str) -> bool:
    """Delete a key from cache. Returns True on success, False on failure."""
    global _redis_available
    if _redis_client is None or not _redis_available:
        return False
    try:
        await _redis_client.delete(key)
        return True
    except (RedisConnectionError, OSError) as e:
        logger.warning(f"Redis connection failed, disabling caching. delete({key}) error: {e}")
        _redis_available = False
    except Exception as e:
        logger.warning(f"Redis delete failed for key {key}: {e}")
    return False

async def clear_cache_pattern(pattern: str) -> bool:
    """Delete keys matching pattern. Returns True on success, False on failure."""
    global _redis_available
    if _redis_client is None or not _redis_available:
        return False
    try:
        keys = await _redis_client.keys(pattern)
        if keys:
            await _redis_client.delete(*keys)
        return True
    except (RedisConnectionError, OSError) as e:
        logger.warning(f"Redis connection failed, disabling caching. keys({pattern}) error: {e}")
        _redis_available = False
    except Exception as e:
        logger.warning(f"Redis clear pattern failed for pattern {pattern}: {e}")
    return False
