import redis
from ..config import settings

def get_redis():
    if not settings.redis_host:
        return None
    return redis.Redis(host=settings.redis_host, port=settings.redis_port, decode_responses=True)