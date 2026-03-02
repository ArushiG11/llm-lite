import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .schemas import GenerateRequest, GenerateResponse
from .model import InferenceEngine

CKPT_PATH = os.getenv("LLMLITE_CKPT", "models/transformer/ckpt.pt")
UI_DIST = Path(__file__).resolve().parent.parent.parent / "ui" / "dist"

app = FastAPI(
    title="llm-lite API",
    description="Inference API for the trained causal transformer. Use /v1/generate or /v1/stream.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine: InferenceEngine | None = None


def _ensure_engine():
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check that the checkpoint exists and the server started correctly.",
        )


@app.on_event("startup")
def _load_model():
    global engine
    ckpt = Path(CKPT_PATH)
    if not ckpt.is_file():
        print(f"Warning: Checkpoint not found at {CKPT_PATH}. Train first: python scripts/train_transformer_causal.py")
        return
    try:
        engine = InferenceEngine(CKPT_PATH)
        print(f"Loaded model from {CKPT_PATH}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        engine = None


@app.get("/", tags=["root"])
def root():
    """Root: links to docs, health, and UI."""
    return {
        "message": "llm-lite API",
        "docs": "/docs",
        "health": "/healthz",
        "ui": "/ui",
        "generate": "POST /v1/generate",
        "stream": "POST /v1/stream",
    }


@app.get("/ui", tags=["ui"])
@app.get("/ui/", tags=["ui"])
def serve_ui():
    """Serve the llm-lite web UI (Phase 3 — React app)."""
    index = UI_DIST / "index.html"
    if not index.is_file():
        raise HTTPException(
            status_code=404,
            detail="UI not built. Run: cd ui && npm install && npm run build",
        )
    return FileResponse(index)


@app.get("/healthz", tags=["health"])
def healthz():
    """Health check. Use for readiness probes."""
    return {
        "ok": engine is not None,
        "ckpt": CKPT_PATH,
        "model_loaded": engine is not None,
    }


@app.post("/v1/generate", response_model=GenerateResponse, tags=["generate"])
def generate(req: GenerateRequest):
    """Generate text in one shot. Returns full prompt + completion."""
    _ensure_engine()
    prompt_ids = engine.encode(req.prompt)

    if len(prompt_ids) > 4096:
        prompt_ids = prompt_ids[-4096:]

    out_ids = engine.generate_ids(
        prompt_ids=prompt_ids,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        repetition_penalty=req.repetition_penalty,
        repeat_window=req.repeat_window,
        seed=req.seed,
    )
    return GenerateResponse(text=engine.decode(out_ids))


@app.post("/v1/stream", tags=["generate"])
def stream(req: GenerateRequest):
    """Stream generated tokens as Server-Sent Events (SSE). One data event per token."""
    _ensure_engine()
    prompt_ids = engine.encode(req.prompt)

    if len(prompt_ids) > 4096:
        prompt_ids = prompt_ids[-4096:]

    def event_stream():
        ids = prompt_ids[:]
        for _ in range(req.max_new_tokens):
            new_ids = engine.generate_ids(
                prompt_ids=ids,
                max_new_tokens=1,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
                repetition_penalty=req.repetition_penalty,
                repeat_window=req.repeat_window,
                seed=req.seed,
            )
            next_id = new_ids[-1]
            ids.append(next_id)
            chunk = engine.decode([next_id])
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if UI_DIST.is_dir():
    app.mount("/ui", StaticFiles(directory=str(UI_DIST), html=True), name="ui")