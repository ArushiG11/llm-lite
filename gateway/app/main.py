from fastapi import FastAPI
from .api import router
from .config import settings
from .observability.otel import setup_otel

app = FastAPI(title="LLM Inference Gateway")
setup_otel(app, settings.service_name, settings.otel_exporter_otlp_endpoint)
app.include_router(router)

@app.get("/healthz")
def healthz():
    return {"ok": True}

from .middleware.rate_limit import RateLimitMiddleware
app.add_middleware(RateLimitMiddleware, limit_per_minute=60)