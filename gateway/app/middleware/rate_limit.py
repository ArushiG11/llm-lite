import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from ..storage.redis_cache import get_redis

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, limit_per_minute=60):
        super().__init__(app)
        self.limit = limit_per_minute
        self.r = get_redis()

    async def dispatch(self, request: Request, call_next):
        if not self.r:
            return await call_next(request)

        api_key = request.headers.get("x-api-key", "anonymous")
        key = f"rl:{api_key}:{int(time.time()//60)}"
        count = self.r.incr(key)
        if count == 1:
            self.r.expire(key, 70)

        if count > self.limit:
            from fastapi.responses import JSONResponse
            return JSONResponse({"error": "rate_limited"}, status_code=429)

        return await call_next(request)