from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4
from pydantic import BaseModel, Field

class Collection(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    user_id: UUID

class Document(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    collection_id: UUID
    file_name: str
    file_path: str
    file_size: int
    mime_type: str
    uploaded_at: datetime = Field(default_factory=datetime.now)
    processed: bool = False
    processed_at: Optional[datetime] = None
    summary: Optional[str] = None
    extraction_metadata: Optional[Dict[str, Any]] = None
    user_id: UUID

class TimelineEvent(BaseModel):
    """Timeline event model."""
    id: Optional[UUID] = None
    document_id: UUID
    event_date: datetime
    title: str
    description: str
    importance: float
    category: Optional[str] = None
    actors: List[str] = []
    location: Optional[str] = None
    confidence_score: float
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class DocumentEmbedding(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    document_id: UUID
    chunk_index: int
    content: str
    embedding: List[float]
    source_element_id: str
    element_type: str
    element_metadata: Dict[str, Any] = {}

class ProcessingResult(BaseModel):
    document_id: UUID
    success: bool
    summary: Optional[str] = None
    events: List[TimelineEvent] = []
    embeddings: List[DocumentEmbedding] = []
    error: Optional[str] = None

class TextChunk(BaseModel):
    content: str
    element_ids: List[str]
    metadata: Dict[str, Any] = {} 