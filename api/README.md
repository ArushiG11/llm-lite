# Phase 2: Backend (Inference API)

**Goal:** Run your trained transformer as a **service** so a user (or a UI) can call it over HTTP.

This API loads the checkpoint from `train_transformer_causal.py`, exposes **generate** and **stream** endpoints, and is ready for a frontend or CLI client.

---

## Prerequisites

1. **Trained checkpoint**  
   From repo root:
   ```bash
   python scripts/train_transformer_causal.py
   ```
   This produces `models/transformer/ckpt.pt` (and embeds the tokenizer path in the checkpoint).

2. **Dependencies**  
   From repo root (same venv as training):
   ```bash
   pip install -r api/requirements.txt
   ```

---

## Run the service

From **repo root** (so paths like `models/transformer/ckpt.pt` resolve):

```bash
# With reload (development)
uvicorn api.app.main:app --reload --port 8000 --reload-exclude '.venv'

# Without reload (stable, e.g. for a UI)
uvicorn api.app.main:app --port 8000
```

Server will be at **http://127.0.0.1:8000**.

- **Root:** http://127.0.0.1:8000/ → JSON with links  
- **UI:** http://127.0.0.1:8000/ui — Phase 3 React app (run `cd ui && npm run build` first).  
- **Docs (Swagger):** http://127.0.0.1:8000/docs  
- **Health:** http://127.0.0.1:8000/healthz  

---

## Environment

| Variable       | Default                      | Description                    |
|----------------|------------------------------|--------------------------------|
| `LLMLITE_CKPT` | `models/transformer/ckpt.pt` | Path to the transformer ckpt. |

Example:
```bash
export LLMLITE_CKPT=models/transformer/ckpt.pt
uvicorn api.app.main:app --port 8000
```

---

## Endpoints

### `POST /v1/generate`

One-shot generation. Request body (JSON):

| Field              | Type   | Default | Description                    |
|--------------------|--------|--------|--------------------------------|
| `prompt`           | string | `""`   | Input text.                    |
| `max_new_tokens`   | int    | 200    | Max tokens to generate (1–1000). |
| `temperature`      | float  | 0.9    | Sampling temperature (0–2).    |
| `top_k`            | int    | 50     | Top-k sampling (0–500).      |
| `top_p`            | float  | 0.9    | Nucleus sampling (0–1).      |
| `repetition_penalty` | float | 1.15   | Penalize recent tokens.       |
| `repeat_window`    | int    | 64     | Window for repetition penalty.|
| `seed`             | int?   | 42     | Random seed (optional).       |

Response: `{"text": "prompt + generated continuation"}`.

**Example (curl):**
```bash
curl -s -X POST http://127.0.0.1:8000/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"The meaning of life is","max_new_tokens":80,"temperature":0.9,"top_k":50,"top_p":0.9}'
```

### `POST /v1/stream`

Same request body as `/v1/generate`, but the response is **Server-Sent Events (SSE)**: one `data: <chunk>` per new token, then `data: [DONE]`. Use this for a typing-effect UI.

**Example (curl):**
```bash
curl -s -N -X POST http://127.0.0.1:8000/v1/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Hello,","max_new_tokens":50}'
```

### `GET /healthz`

Returns `{"ok": true, "ckpt": "<path>"}`. Use for readiness checks.

---

## Summary

| Item        | Description                                      |
|------------|--------------------------------------------------|
| **Goal**   | Serve the trained transformer over HTTP.         |
| **Stack**  | FastAPI + PyTorch + HuggingFace tokenizers.      |
| **Model**  | Same as `scripts/generate_transformer.py` (GPTMini). |
| **Clients**| Any HTTP client: curl, browser, or the **Phase 3 UI** at `/ui`. |
