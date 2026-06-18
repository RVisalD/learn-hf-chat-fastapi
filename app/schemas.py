from pydantic import BaseModel, Field
from typing import Optional, List


class Message(BaseModel):
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    messages: List[Message] = Field(..., description="Conversation history")
    max_new_tokens: Optional[int] = Field(None, ge=1, le=4096)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    stream: bool = Field(False, description="Enable streaming (SSE)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "temperature": 0.7,
                "max_new_tokens": 256,
                "messages": [
                    {"role": "user", "content": "Explain what a transformer model is in simple terms."},
                    {"role": "assistant", "content": "I am a large language model created by Alibaba Cloud. I am called Qwen."},
                    
                ],
            }
        }
    }


class ChatResponse(BaseModel):
    model: str
    generated_text: str
    usage: Optional[dict] = None


class ModelInfo(BaseModel):
    model_id: str
    description: str
    task: str