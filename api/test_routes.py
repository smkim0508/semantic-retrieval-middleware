# test routes
import json
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import text
import redis.asyncio as aioredis

from common.logger import logger
from common.timer import async_timer
from core.dependencies import get_main_db_engine, get_gemini_text_embedding_client, get_memory_retriever, get_redis_client
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

@router.get("/redis-cache")
async def get_redis_cache(
    redis_client: aioredis.Redis = Depends(get_redis_client),
):
    """
    Returns all key-value pairs currently stored in Redis.
    """
    keys = await redis_client.keys("*")
    if not keys:
        return {"count": 0, "entries": {}}

    values = await redis_client.mget(*keys)
    entries = {
        key: json.loads(value) if value else None
        for key, value in zip(keys, values)
    }
    logger.info(f"Returning {len(entries)} cached entries")
    return {"count": len(entries), "entries": entries}

@router.post("/clear-cache")
async def clear_cache(
    redis_client: aioredis.Redis = Depends(get_redis_client),
    memory: MemoryInterface = Depends(get_memory_retriever),
):
    """
    Clears all cache layers:
    - 1) in-memory exact cache (L1)
    - 2) Redis exact cache (L2) 
    - 3) semantic cache (L3) 
    """
    await redis_client.flushdb()
    memory._exact_cache.clear()
    memory._semantic_cache.clear()
    logger.info("All cache layers cleared")
    return {"message": "All cache layers cleared"}

# TODO: make redis cache client wrapper to save/load easily
