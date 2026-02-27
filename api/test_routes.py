# test routes
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import text

from common.logger import logger
from common.timer import async_timer
from core.dependencies import get_main_db_engine, get_gemini_text_embedding_client, get_memory_retriever
from db.crud import store_vector
from db.session import get_async_session_maker
from memory_interface import MemoryInterface
from models.embeddings.gemini_embedding_client import GenAITextEmbeddingClient
from sqlalchemy.ext.asyncio import AsyncEngine

router = APIRouter(prefix="/test", tags=["Test"])

# request body using pydantic
class EmbedStoreRequest(BaseModel):
    text: str

@router.get("/retrieve")
async def test_retrieve(
    query: str = Query(description="Text query to retrieve semantically similar results"),
    limit: int = 5,
    memory: MemoryInterface = Depends(get_memory_retriever),
):
    logger.info(f"[test/retrieve] query='{query}', limit={limit}")

    async with async_timer("memory.retrieve"):
        results = await memory.retrieve(query=query, limit=limit)

    return {
        "query": query,
        "results": results,
        "count": len(results),
    }

@router.post("/embed-and-store")
async def test_embed_and_store(
    body: EmbedStoreRequest,
    embedding_client: GenAITextEmbeddingClient = Depends(get_gemini_text_embedding_client),
    engine: AsyncEngine = Depends(get_main_db_engine),
):
    logger.info(f"[test/embed-and-store] text='{body.text}'")

    async with async_timer("aembed_text"):
        embeddings = await embedding_client.aembed_text([body.text], task_type="RETRIEVAL_DOCUMENT")

    if not embeddings:
        logger.error("[test/embed-and-store] embedding failed")
        return {"error": "Embedding failed"}

    vector = embeddings[0]
    logger.info(f"[test/embed-and-store] vector length: {len(vector)}")

    async with async_timer("store_vector"):
        row = await store_vector(vector=vector, text=body.text, engine=engine)

    logger.info(f"[test/embed-and-store] stored row id: {row.id}")

    return {
        "text": body.text,
        "stored_id": row.id,
        "vector_length": len(vector),
    }
