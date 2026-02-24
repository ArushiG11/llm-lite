from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from .config import settings
from .providers.openai_provider import OpenAIProvider

router = APIRouter()

def get_provider():
    if settings.provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        return OpenAIProvider(settings.openai_api_key, settings.openai_base_url)
    raise RuntimeError(f"Unknown provider: {settings.provider}")

@router.post("/v1/chat/completions")
async def chat_completions(payload: dict):
    provider = get_provider()

    async def event_stream():
        try:
            async for line in provider.chat_stream(payload):
                # Ensure SSE framing
                if line.startswith("data:"):
                    yield line if line.endswith("\n\n") else (line + "\n")
                else:
                    yield "data: " + line + "\n\n"
        except Exception as e:
            yield f"data: {{" + f"\"error\": {repr(str(e))}" + "}}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")