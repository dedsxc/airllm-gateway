import time
from typing import List, Optional, Literal
from pydantic import BaseModel, Field

# --- OpenAI Format Objects ---
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: int = Field(default=512, ge=1, description="Number tokens max")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool = False

class Choice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[Choice]
    usage: dict

class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "airllm"

class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard]