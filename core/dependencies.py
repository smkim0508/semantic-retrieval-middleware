from typing import Any, Optional
from common.logger import logger
from fastapi import Request, Depends
from sqlalchemy.ext.asyncio import AsyncEngine
from models.embeddings.gemini_embedding_client import GenAITextEmbeddingClient

# conveniently return any app lifetime dependencies to be used in routes
def get_main_db_engine(request: Request) -> AsyncEngine:
    """
    FastAPI dependency to get the shared main DB engine from the application state.
    """
    return request.app.state.main_db_engine

def get_gemini_text_embedding_client(request: Request) -> GenAITextEmbeddingClient:
    """
    FastAPI dependency to get the shared Google Gen AI text embedding client from the application state.
    """
    return request.app.state.gemini_text_embedding_client
