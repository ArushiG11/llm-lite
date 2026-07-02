"""
FastAPI entrypoint: loads config and inference services once at startup.
"""

from __future__ import annotations

import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from .api.routes import router
from .core.config import get_settings
from .core.limiter import limiter
from .services.inference_service import InferenceService


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    app.state.inference = InferenceService(settings)
    if app.state.inference.is_ready:
        print(f"Inference ready (checkpoint: {settings.resolved_checkpoint_path()})")
    else:
        print(f"Inference not ready: {app.state.inference.load_error}")
    yield


app = FastAPI(
    title="llm-lite inference API",
    description="Synchronous + streaming inference over a trained GPTMini checkpoint and BPE tokenizer.",
    version="2.0.0",
    lifespan=lifespan,
)

# Rate limiter state (slowapi reads from app.state.limiter).
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS: allow the Vite dev server and any production origin.
# In production, replace "*" with your actual domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://localhost:80"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# All inference routes live under /v1 — matches the Vite proxy config and client.ts.
app.include_router(router, prefix="/v1")


# /healthz is separate from /v1 so the UI can poll it before the model is
# confirmed ready without needing an auth or version prefix.
@app.get("/healthz", tags=["health"])
async def healthz(request: Request):
    """Combined liveness + readiness check for the UI.

    Returns { ok, model_loaded, ckpt } so the frontend can show a status badge
    without making two separate calls (/health + /ready).
    """
    svc = getattr(request.app.state, "inference", None)
    model_loaded = svc is not None and svc.is_ready
    ckpt = str(svc._loaded.checkpoint_path) if model_loaded and svc._loaded else ""
    return {"ok": True, "ckpt": ckpt, "model_loaded": model_loaded}


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation error",
            "code": "validation_error",
            "detail": exc.errors(),
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(_request: Request, exc: HTTPException):
    """Return JSON bodies for HTTPException (including FastAPI defaults)."""
    if isinstance(exc.detail, dict):
        body = exc.detail
    else:
        body = {"error": str(exc.detail), "code": "http_error"}
    return JSONResponse(status_code=exc.status_code, content=body)


@app.exception_handler(Exception)
async def unhandled_exception_handler(_request: Request, exc: Exception):
    """Avoid raw tracebacks in responses; log server-side."""
    if isinstance(exc, (RequestValidationError, HTTPException)):
        raise exc
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "code": "internal_error",
            "reason": str(exc),
        },
    )
