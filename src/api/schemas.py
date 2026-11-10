# src/api/schemas.py
from pydantic import BaseModel, Field, validator

class ChatRequest(BaseModel):
    session_id: str = Field(..., description="UUID for the conversation session")
    query: str = Field(..., min_length=1)
    
    @validator('query')
    def validate_length(cls, v):
        if len(v) > 2000:
            raise ValueError("Query too long")
        return v

class ChatResponse(BaseModel):
    answer: str
    references: list[str] = []
    processing_time: float

class FeedbackRequest(BaseModel):
    message_id: str
    score: int # 1 or -1
    comment: str | None = None