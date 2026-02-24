from abc import ABC, abstractmethod
from typing import AsyncIterator, Dict, Any

class LLMProvider(ABC):
    @abstractmethod
    async def chat_stream(self, payload: Dict[str, Any]) -> AsyncIterator[str]:
        """Yield SSE data lines (already formatted as `data: ...\n\n`)."""
        ...