import json
import httpx
from typing import AsyncIterator, Dict, Any
from .base import LLMProvider

class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    async def chat_stream(self, payload: Dict[str, Any]) -> AsyncIterator[str]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        url = f"{self.base_url}/chat/completions"
        payload = {**payload, "stream": True}

        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", url, headers=headers, json=payload) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    # pass-through (OpenAI already emits `data: ...`)
                    yield line + "\n"