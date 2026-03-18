from pydantic import BaseModel, Field
from typing import Optional

class GenerateRequest(BaseModel):
    prompt: str = Field(default="", description="User prompt text")
    max_new_tokens: int = Field(default=200, ge=1, le=1000)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0, description="Lower = more focused, less random (0.7-0.9 typical)")
    top_k: int = Field(default=40, ge=0, le=500, description="Sample from top-k tokens (lower = more coherent)")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Nucleus sampling threshold")
    repetition_penalty: float = Field(default=1.15, ge=1.0, le=2.0, description="Discourage repeating recent tokens")
    repeat_window: int = Field(default=64, ge=0, le=512)
    seed: Optional[int] = Field(default=42)

class GenerateResponse(BaseModel):
    text: str