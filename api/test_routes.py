# test routes
import json
import time
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import text
import redis.asyncio as aioredis

from common.logger import logger
from common.timer import async_timer
from core.dependencies import get_main_db_engine, get_gemini_text_embedding_client, get_memory_retriever, get_redis_client, get_extended_memory_retriever
from db.crud import store_vector
from db.session import get_async_session_maker
from memory_interface import MemoryInterface
from memory_interface_extended import ExtendedMemoryInterface
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

    start = time.perf_counter()
    results = await memory.retrieve(query=query, limit=limit)
    elapsed_ms = (time.perf_counter() - start) * 1000

    logger.info(f"total elapsed time: {round(elapsed_ms, 2)}")

    return {
        "query": query,
        "results": results,
        "count": len(results),
        "elapsed_ms": round(elapsed_ms, 2),
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

@router.get("/retrieve-reranked")
async def test_retrieve_reranked(
    query: str = Query(description="Text query to retrieve and rerank semantically similar results"),
    limit: int = 5,
    retrieval_size: int = 50,
    memory: MemoryInterface = Depends(get_memory_retriever),
):
    logger.info(f"[test/retrieve-reranked] query='{query}', limit={limit}, retrieval_size={retrieval_size}")

    start = time.perf_counter()
    results = await memory.retrieve_and_rerank(query=query, limit=limit, retrieval_size=retrieval_size)
    elapsed_ms = (time.perf_counter() - start) * 1000

    logger.info(f"total elapsed time: {round(elapsed_ms, 2)}")

    return {
        "query": query,
        "results": results,
        "count": len(results),
        "elapsed_ms": round(elapsed_ms, 2),
    }

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
    memory._cache_fetch_sizes.clear()
    logger.info("All cache layers cleared")
    return {"message": "All cache layers cleared"}

@router.post("/store-warm")
async def test_store_warm(
    body: EmbedStoreRequest,
    extended_memory: ExtendedMemoryInterface = Depends(get_extended_memory_retriever),
):
    """
    Stores text via the warm-buffer path:
    1. Immediately writes a ground_truth row (is_synced=False).
    2. Embeds the text and holds it in the in-memory warm buffer.
    3. Auto-flushes if the buffer reaches the flush threshold.
    Returns the assigned ground_truth_id and current buffer size.
    """
    logger.info(f"[test/store-warm] text='{body.text}'")

    start = time.perf_counter()
    gt_id = await extended_memory.store_via_warm_buffer(body.text)
    elapsed_ms = (time.perf_counter() - start) * 1000

    logger.info(f"[test/store-warm] ground_truth_id={gt_id}, buffer_size={len(extended_memory._warm_buffer)}")

    return {
        "text": body.text,
        "ground_truth_id": gt_id,
        "buffer_size": len(extended_memory._warm_buffer),
        "elapsed_ms": round(elapsed_ms, 2),
    }

@router.post("/flush-warm")
async def test_flush_warm(
    extended_memory: ExtendedMemoryInterface = Depends(get_extended_memory_retriever),
):
    """
    Manually triggers a warm-buffer flush:
        - Deletes stale VectorDB rows for updated items.
        - Bulk-inserts new VectorDB rows.
        - Marks ground_truth rows as is_synced=True.
        - Clears the in-memory buffer.
    """
    logger.info("[test/flush-warm] manual flush triggered")

    start = time.perf_counter()
    count = await extended_memory.flush_warm_buffer()
    elapsed_ms = (time.perf_counter() - start) * 1000

    logger.info(f"[test/flush-warm] flushed {count} entries in {round(elapsed_ms, 2)}ms")

    return {
        "flushed_count": count,
        "elapsed_ms": round(elapsed_ms, 2),
    }

# TODO: make redis cache client wrapper to save/load easily
