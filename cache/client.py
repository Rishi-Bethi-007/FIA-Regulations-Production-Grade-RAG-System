# cache/redis_client.py
from __future__ import annotations

import redis
from config import REDIS_HOST, REDIS_PORT


_redis_client: redis.Redis | None = None


def get_redis() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            decode_responses=False,
        )
        # Fail fast if Redis is unreachable
        _redis_client.ping()
    return _redis_client
