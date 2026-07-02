"""HTTP routes: validation and delegation to InferenceService."""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

from ..services.inference_service import (
    InferenceError,
    InferenceNotReadyError,
    InferenceService,
)
from .schemas import (
    DetokenizeRequest,
    DetokenizeResponse,
    GenerateMetadata,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    ModelInfoResponse,
    ModelLimits,
    ReadyResponse,
    TokenizeRequest,
    TokenizeResponse,
)

router = APIRouter()


def _get_service(request: Request) -> InferenceService:
    svc = getattr(request.app.state, "inference", None)
    if svc is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": "Inference service unavailable", "code": "no_service"},
        )
    return svc


@router.get("/health", response_model=HealthResponse, tags=["health"])
async def health() -> HealthResponse:
    """Liveness: process is running (does not guarantee the model is loaded)."""
    return HealthResponse(status="ok")


@router.get("/ready", response_model=ReadyResponse, tags=["health"])
async def ready(request: Request) -> ReadyResponse:
    """Readiness: model and tokenizer are loaded. Returns 503 if not ready."""
    svc = _get_service(request)
    if svc.is_ready:
        return ReadyResponse(ready=True, detail=None)
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={
            "ready": False,
            "detail": svc.load_error or "Model and tokenizer are not loaded.",
        },
    )


@router.get("/model-info", response_model=ModelInfoResponse, tags=["model"])
async def model_info(request: Request) -> ModelInfoResponse:
    """Static model/tokenizer metadata and server-side limits."""
    svc = _get_service(request)
    try:
        info = svc.model_info()
    except InferenceNotReadyError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": str(e), "code": "not_ready"},
        ) from e
    lim = info.pop("limits")
    return ModelInfoResponse(**info, limits=ModelLimits(**lim))


@router.post("/generate", response_model=GenerateResponse, tags=["inference"])
async def generate_endpoint(request: Request, body: GenerateRequest) -> GenerateResponse:
    """Sample text with GPTMini.generate() (autoregressive, returns when done).

    Runs in a thread pool so the event loop stays unblocked during CPU-bound
    inference. Rate-limited to 20 requests/minute per IP.
    """
    svc = _get_service(request)
    try:
        result = await run_in_threadpool(
            svc.generate,
            prompt=body.prompt,
            max_new_tokens=body.max_new_tokens,
            temperature=body.temperature,
            top_k=body.top_k,
            top_p=body.top_p,
            repetition_penalty=body.repetition_penalty,
            repeat_window=body.repeat_window,
            seed=body.seed,
        )
    except InferenceNotReadyError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": e.message, "code": "not_ready"},
        ) from e
    except InferenceError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": str(e), "code": e.code},
        ) from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Inference failed", "code": "inference_failed", "reason": str(e)},
        ) from e

    meta = GenerateMetadata(
        model_version=result.model_version,
        tokenizer_path=result.tokenizer_path,
        tokenizer_version=result.tokenizer_version,
        latency_ms=result.latency_ms,
        prompt_token_count=result.prompt_token_count,
        context_token_count=result.context_token_count,
        new_tokens_generated=result.new_tokens_generated,
    )
    return GenerateResponse(text=result.text, metadata=meta)


@router.post("/stream", tags=["inference"])
async def stream_endpoint(request: Request, body: GenerateRequest):
    """Stream tokens via Server-Sent Events as they are generated.

    Each event: data: <json-encoded text chunk>\\n\\n
    Final event: data: [DONE]\\n\\n

    iterate_in_threadpool runs each next() call of the sync generator in a
    thread-pool worker so the event loop is never blocked during inference.
    Rate-limited to 10 requests/minute per IP (streams are long-lived).
    """
    svc = _get_service(request)

    async def event_stream():
        gen = svc.generate_stream(
            prompt=body.prompt,
            max_new_tokens=body.max_new_tokens,
            temperature=body.temperature,
            top_k=body.top_k,
            top_p=body.top_p,
            repetition_penalty=body.repetition_penalty,
            repeat_window=body.repeat_window,
            seed=body.seed,
        )
        try:
            async for chunk in iterate_in_threadpool(gen):
                yield f"data: {json.dumps(chunk)}\n\n"
        except (InferenceNotReadyError, InferenceError) as e:
            yield f"data: {json.dumps('[ERROR] ' + str(e))}\n\n"
        except Exception as e:  # noqa: BLE001
            yield f"data: {json.dumps('[ERROR] ' + str(e))}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/tokenize", response_model=TokenizeResponse, tags=["tokenizer"])
async def tokenize_endpoint(request: Request, body: TokenizeRequest) -> TokenizeResponse:
    """Encode text to token ids."""
    svc = _get_service(request)
    try:
        result = await run_in_threadpool(svc.tokenize, body.text)
    except InferenceNotReadyError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": e.message, "code": "not_ready"},
        ) from e
    except InferenceError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": str(e), "code": e.code},
        ) from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Tokenization failed", "code": "tokenize_failed", "reason": str(e)},
        ) from e

    return TokenizeResponse(
        ids=result.ids,
        token_count=result.token_count,
        latency_ms=result.latency_ms,
        tokenizer_path=result.tokenizer_path,
        tokenizer_version=result.tokenizer_version,
    )


@router.post("/detokenize", response_model=DetokenizeResponse, tags=["tokenizer"])
async def detokenize_endpoint(request: Request, body: DetokenizeRequest) -> DetokenizeResponse:
    """Decode token ids to text."""
    svc = _get_service(request)
    try:
        result = await run_in_threadpool(svc.detokenize, body.ids)
    except InferenceNotReadyError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": e.message, "code": "not_ready"},
        ) from e
    except InferenceError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": str(e), "code": e.code},
        ) from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Detokenization failed", "code": "detokenize_failed", "reason": str(e)},
        ) from e

    return DetokenizeResponse(
        text=result.text,
        token_count=result.token_count,
        latency_ms=result.latency_ms,
        tokenizer_path=result.tokenizer_path,
        tokenizer_version=result.tokenizer_version,
    )
