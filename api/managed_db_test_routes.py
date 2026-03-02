# test routes for the managed vector DB path (warm-buffer write + ground-truth validation)
import time
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from common.logger import logger
from common.timer import async_timer
from core.dependencies import get_extended_memory_retriever
from memory_interface_extended import ExtendedMemoryInterface

router = APIRouter(prefix="/test/managed", tags=["Test - Managed Vector DB"])

class EmbedStoreRequest(BaseModel):
    text: str

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
    logger.info(f"[managed/store-warm] text='{body.text}'")

    start = time.perf_counter()
    gt_id = await extended_memory.store_via_warm_buffer(body.text)
    elapsed_ms = (time.perf_counter() - start) * 1000

    logger.info(f"[managed/store-warm] ground_truth_id={gt_id}, buffer_size={len(extended_memory._warm_buffer)}")

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
      - Deletes stale VectorDBManaged rows for updated items.
      - Bulk-inserts new VectorDBManaged rows.
      - Marks ground_truth rows as is_synced=True.
      - Clears the in-memory buffer.
    """
    logger.info("[managed/flush-warm] manual flush triggered")

    start = time.perf_counter()
    count = await extended_memory.flush_warm_buffer()
    elapsed_ms = (time.perf_counter() - start) * 1000

    logger.info(f"[managed/flush-warm] flushed {count} entries in {round(elapsed_ms, 2)}ms")

    return {
        "flushed_count": count,
        "elapsed_ms": round(elapsed_ms, 2),
    }

@router.get("/retrieve")
async def test_managed_retrieve(
    query: str = Query(description="Text query to retrieve semantically similar results from managed vector DB"),
    limit: int = 5,
    extended_memory: ExtendedMemoryInterface = Depends(get_extended_memory_retriever),
):
    """
    Retrieves results from the managed vector DB (VectorDBManaged).
    When the warm buffer is non-empty, also searches it and validates candidates
    against the ground_truth table before returning.
    """
    logger.info(f"[managed/retrieve] query='{query}', limit={limit}")

    start = time.perf_counter()
    results = await extended_memory.retrieve(query=query, limit=limit)
    elapsed_ms = (time.perf_counter() - start) * 1000

    logger.info(f"[managed/retrieve] total elapsed: {round(elapsed_ms, 2)}ms")

    return {
        "query": query,
        "results": results,
        "count": len(results),
        "warm_buffer_size": len(extended_memory._warm_buffer),
        "elapsed_ms": round(elapsed_ms, 2),
    }

@router.get("/retrieve-reranked")
async def test_managed_retrieve_reranked(
    query: str = Query(description="Text query to retrieve and rerank results from managed vector DB"),
    limit: int = 5,
    retrieval_size: int = 50,
    extended_memory: ExtendedMemoryInterface = Depends(get_extended_memory_retriever),
):
    """
    Retrieves and cross-encoder reranks results from the managed vector DB (VectorDBManaged).
    When the warm buffer is non-empty, also searches it and validates candidates
    against the ground_truth table before reranking.
    """
    logger.info(f"[managed/retrieve-reranked] query='{query}', limit={limit}, retrieval_size={retrieval_size}")

    start = time.perf_counter()
    results = await extended_memory.retrieve_and_rerank(query=query, limit=limit, retrieval_size=retrieval_size)
    elapsed_ms = (time.perf_counter() - start) * 1000

    logger.info(f"[managed/retrieve-reranked] total elapsed: {round(elapsed_ms, 2)}ms")

    return {
        "query": query,
        "results": results,
        "count": len(results),
        "warm_buffer_size": len(extended_memory._warm_buffer),
        "elapsed_ms": round(elapsed_ms, 2),
    }
