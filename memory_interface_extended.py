# Extended memory interface: warm-buffer write path + ground-truth validation on retrieval.
# Inherits all existing caching/retrieval behaviour from MemoryInterface and overrides
# retrieve / retrieve_and_rerank to incorporate warm-buffer freshness when active.
import asyncio
import json
import numpy as np
from sqlalchemy.ext.asyncio import AsyncEngine
import redis.asyncio as aioredis
from typing import Optional, cast

from memory_interface import MemoryInterface
from models.embeddings.gemini_embedding_client import GenAITextEmbeddingClient
from models.reranker.cross_encoder import CEReranker
from db.crud import (
    find_similar,
    find_similar_extended,
    create_ground_truth,
    get_ground_truth_sync_status,
    flush_warm_entries,
)
from common.logger import logger
from common.timer import async_timer


class ExtendedMemoryInterface(MemoryInterface):
    """
    Extends MemoryInterface with a warm-buffer write path and ground-truth validation.
    NOTE: this is the PoC for freshness-aware memory caching + retrieval interface.

    ── Warm-buffer write path ─────────────────────────────────────────────────────
    store_via_warm_buffer(text):
      - Immediately writes to the ground_truth structured table (is_synced=False).
        This is the authoritative record; no vector embedding is required here.
      - Embeds the text and holds {ground_truth_id, vector, text} in an in-memory
        _warm_buffer list.
      - Auto-flushes to the vector DB when buffer size >= _warm_buffer_flush_threshold.

    flush_warm_buffer():
      - Atomically moves all warm-buffer entries into the vector DB and marks their
        ground_truth rows as is_synced=True.
      - Clears the buffer. Called by the background periodic-flush task in lifespan.py
        and exposed as a manual endpoint via POST /test/flush-warm.

    ── Retrieval when warm buffer is active ──────────────────────────────────────
    When _warm_buffer is non-empty, retrieve / retrieve_and_rerank SKIP all caches
    and perform a fresh validated retrieval:
      1. Fetch 2 times requested candidates from the vector DB (includes id + ground_truth_id).
      2. Cosine-similarity search over the warm buffer (brute-force; buffer is small).
      3. Combine results (warm-buffer entries win dedup on same ground_truth_id).
      4. Validate vector-DB candidates against the ground_truth table:
           - ground_truth_id is NULL -> legacy/direct-path item, always valid.
           - ground_truth_id not NULL -> valid only if ground_truth.is_synced=True.
            (is_synced=False -> stale; the warm buffer holds the newer version.)
         Warm-buffer entries are always valid.
      5. If valid candidates < requested size, top up with a second DB fetch
         (SQL OFFSET past the first batch) + the same validation logic.
      6. Cache the final validated texts normally (warm for when buffer empties).

    When _warm_buffer is empty the parent's three-tier cache hierarchy is used
    unchanged (L1 LRU -> L2 Redis -> L3 semantic deque -> vector DB).

    ── Cache-coherence note ──────────────────────────────────────────────────────
    After a flush the warm buffer is empty, so subsequent requests use caches.
    If a cached result pre-dates new data from a flush, clear the caches via
    POST /test/clear-cache.
    """

    def __init__(
        self,
        embedding_client: GenAITextEmbeddingClient,
        main_db_engine: AsyncEngine,
        redis_client: aioredis.Redis,
        cross_encoder_reranker: CEReranker,
        warm_buffer_flush_threshold: int = 100,
    ):
        super().__init__(
            embedding_client=embedding_client,
            main_db_engine=main_db_engine,
            redis_client=redis_client,
            cross_encoder_reranker=cross_encoder_reranker,
        )
        # warm buffer: [{ground_truth_id, vector, text}]
        self._warm_buffer: list[dict] = []
        self._warm_buffer_flush_threshold = warm_buffer_flush_threshold
        # background periodic flush task handle
        self._flush_task: Optional[asyncio.Task] = None

    # ── warm buffer helpers ─────────────────────────────────────────────────────

    def _search_warm_buffer(self, query_vector: list[float], limit: int) -> list[dict]:
        """
        Brute-force cosine-similarity search over the warm buffer.
        Returns up to 'limit' entries sorted by descending similarity, each tagged
        with source='warm_buffer' for downstream validation.
        """
        if not self._warm_buffer:
            return []
        scored = [
            (self._cosine_similarity(query_vector, entry["vector"]), entry)
            for entry in self._warm_buffer
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "id": None,
                "text": entry["text"],
                "ground_truth_id": entry["ground_truth_id"],
                "source": "warm_buffer",
            }
            for _, entry in scored[:limit]
        ]

    async def _validate_candidates(self, candidates: list[dict]) -> list[str]:
        """
        Validates retrieval candidates against the ground_truth table and returns
        only the texts of valid candidates, preserving ranking order.

        Rules:
          - source='warm_buffer' -> always valid (freshest data).
          - source='vector_db', gt_id=None -> always valid (legacy direct-store item).
          - source='vector_db', gt_id≠None -> valid only if ground_truth.is_synced=True.
            (is_synced=False -> the item was updated; warm buffer holds newer version.)
        """
        gt_ids_to_check = [
            c["ground_truth_id"]
            for c in candidates
            if c["source"] == "vector_db" and c["ground_truth_id"] is not None
        ]
        gt_status = await get_ground_truth_sync_status(gt_ids_to_check, self.main_db_engine)

        valid_texts: list[str] = []
        for c in candidates:
            if c["source"] == "warm_buffer":
                valid_texts.append(c["text"])
            elif c["ground_truth_id"] is None:
                valid_texts.append(c["text"]) # legacy item, always valid
            elif gt_status.get(c["ground_truth_id"], False):
                valid_texts.append(c["text"]) # is_synced=True -> valid
            # else: is_synced=False or record missing -> stale, silently drop
        return valid_texts

    async def _fetch_fresh_validated(self, query_vector: list[float], target_count: int) -> list[str]:
        """
        Core validated-retrieval routine used when the warm buffer is non-empty.

        1. Fetches 2 times target_count candidates from the vector DB (with id/ground_truth_id).
        2. Cosine-searches the warm buffer for up to 2 times target_count candidates.
        3. Merges both sets; warm-buffer entries win dedup on matching ground_truth_id.
        4. Validates the merged set against the ground_truth table.
        5. If still < target_count valid texts, performs a second DB fetch with SQL
        - OFFSET = fetch_count and applies the same validation.
        Returns up to target_count validated, deduplicated texts.
        """
        fetch_count = target_count * 2

        # ── first pass ─────────────────────────────────────────────────────────
        async with async_timer("find_similar_extended"):
            db_raw = await find_similar_extended(query_vector, self.main_db_engine, fetch_count)
        db_candidates = [{**c, "source": "vector_db"} for c in db_raw]
        warm_candidates = self._search_warm_buffer(query_vector, fetch_count)

        # Merge; warm buffer first so it wins dedup on the same ground_truth_id.
        seen_gt_ids: set[int] = set()
        seen_db_ids: set[int] = set()
        deduped: list[dict] = []

        for c in warm_candidates + db_candidates:
            if c["source"] == "warm_buffer":
                gt_id = c["ground_truth_id"]
                if gt_id is not None:
                    if gt_id in seen_gt_ids:
                        continue
                    seen_gt_ids.add(gt_id)
            else: # vector_db
                db_id = c["id"]
                if db_id in seen_db_ids:
                    continue
                seen_db_ids.add(db_id)
                gt_id = c["ground_truth_id"]
                if gt_id is not None:
                    if gt_id in seen_gt_ids:
                        continue # warm buffer already covering this logical item
                    seen_gt_ids.add(gt_id)
            deduped.append(c)

        async with async_timer("validate_candidates"):
            valid_texts = await self._validate_candidates(deduped)

        # ── top-up pass ────────────────────────────────────────────────────────
        if len(valid_texts) < target_count:
            logger.info(
                f"[warm retrieval] {len(valid_texts)}/{target_count} valid after first pass;"
                f"top-up with offset={fetch_count}"
            )
            async with async_timer("find_similar_extended_topup"):
                extra_raw = await find_similar_extended(
                    query_vector, self.main_db_engine, fetch_count, offset=fetch_count
                )
            extra_candidates = [
                {**c, "source": "vector_db"}
                for c in extra_raw
                if c["id"] not in seen_db_ids
            ]
            async with async_timer("validate_candidates_topup"):
                extra_valid = await self._validate_candidates(extra_candidates)

            seen_texts: set[str] = set(valid_texts)
            for t in extra_valid:
                if t not in seen_texts:
                    valid_texts.append(t)
                    seen_texts.add(t)

        return valid_texts[:target_count]

    # ── warm buffer write path ──────────────────────────────────────────────────

    async def store_via_warm_buffer(self, text: str) -> int:
        """
        Stores new text through the warm-buffer path:
        1. Immediately writes a GroundTruth row (is_synced=False) — canonical record.
        2. Embeds the text and appends {ground_truth_id, vector, text} to the buffer.
        3. Auto-flushes to the vector DB if buffer size >= _warm_buffer_flush_threshold.
        Returns the ground_truth_id assigned to this entry.
        """
        # 1. Write canonical record immediately
        gt_row = await create_ground_truth(text, self.main_db_engine)
        # gt_row.id is always an int after commit+refresh, but cast for type hinting
        gt_id = cast(int, gt_row.id)
        logger.info(f"[warm buffer] ground truth created: id={gt_id}")

        # 2. Embed and hold in warm buffer
        embeddings = await self.embedding_client.aembed_text([text], task_type="RETRIEVAL_DOCUMENT")
        if not embeddings:
            logger.error(
                f"[warm buffer] embedding failed; ground_truth_id={gt_id} remains in GT only"
            )
            return gt_id

        self._warm_buffer.append({
            "ground_truth_id": gt_id,
            "vector": embeddings[0],
            "text": text,
        })
        logger.info(f"[warm buffer] entry added; buffer_size={len(self._warm_buffer)}")

        # 3. Auto-flush if threshold reached
        if len(self._warm_buffer) >= self._warm_buffer_flush_threshold:
            logger.info(
                f"[warm buffer] threshold ({self._warm_buffer_flush_threshold}) reached; auto-flushing"
            )
            await self.flush_warm_buffer()

        return gt_id

    async def flush_warm_buffer(self) -> int:
        """
        Atomically flushes all warm-buffer entries into the vector DB:
            - Deletes any existing VectorDB rows with the same ground_truth_id (handles updates).
            - Bulk-inserts new VectorDB rows with ground_truth_id set.
            - Marks the corresponding GroundTruth rows as is_synced=True.
            - Clears the in-memory warm buffer.
        Returns the number of entries flushed.
        """
        if not self._warm_buffer:
            logger.info("[warm buffer] flush called but buffer is empty; skipping")
            return 0

        entries_to_flush = list(self._warm_buffer) # snapshot before clearing
        logger.info(f"[warm buffer] flushing {len(entries_to_flush)} entries to vector DB")

        async with async_timer("flush_warm_entries"):
            count = await flush_warm_entries(entries_to_flush, self.main_db_engine)

        self._warm_buffer.clear()
        logger.info(f"[warm buffer] flush complete; {count} entries synced, buffer cleared")
        return count

    # ── overridden retrieval methods ────────────────────────────────────────────

    async def retrieve(self, query: str, limit: int = 5) -> list[str]:
        """
        Baseline retrieval without reranking, backed by VectorDBManaged.

        When warm buffer is non-empty:
            - Bypasses all caches. Uses _fetch_fresh_validated (2x fetch + GT validation top-up), then caches the validated texts for when the buffer empties.

        When warm buffer is empty:
            - Full L1->L2->L3->VectorDBManaged cache hierarchy (mirrors MemoryInterface, but queries vector_db_managed instead of vector_db).
        """
        cache_key = self._make_cache_key(query, "plain")

        if not self._warm_buffer:
            # L1
            if cache_key in self._exact_cache:
                cached_results = self._exact_cache[cache_key]
                if len(cached_results) >= limit:
                    logger.info(f"[L1 cache] exact hit: {query}")
                    self._exact_cache.move_to_end(cache_key)
                    return cached_results[:limit]

            # L2 Redis
            cached = await self.redis_client.get(cache_key)
            if cached:
                results = json.loads(cached)
                if len(results) >= limit:
                    logger.info(f"[L2 cache] Redis hit: {query}")
                    self._set_exact_cache(cache_key, results)
                    return results[:limit]

            embeddings = self.embedding_client.embed_text([query], task_type="RETRIEVAL_QUERY")
            if not embeddings:
                return []
            query_vector = embeddings[0]

            # L3 semantic cache
            semantic_hit = self._find_semantic_cache_hit(query_vector, rerank=False, size_needed=limit)
            if semantic_hit:
                logger.info(f"[L3 cache] semantic hit: {query}")
                self._set_exact_cache(cache_key, semantic_hit)
                await self.redis_client.set(cache_key, json.dumps(semantic_hit))
                return semantic_hit[:limit]

            # DB miss — query VectorDBManaged
            logger.info(f"no cache hit, retrieving from managed db: {query}")
            async with async_timer("find_similar_extended"):
                db_rows = await find_similar_extended(query_vector, self.main_db_engine, limit)
            results = [row["text"] for row in db_rows]
            self._set_exact_cache(cache_key, results)
            self._semantic_cache.append((query_vector, results, False, limit))
            await self.redis_client.set(cache_key, json.dumps(results))
            return results

        # warm buffer active — bypass caches
        logger.info(f"[warm buffer active] retrieve bypassing caches: '{query}'")
        embeddings = self.embedding_client.embed_text([query], task_type="RETRIEVAL_QUERY")
        if not embeddings:
            return []
        query_vector = embeddings[0]

        results = await self._fetch_fresh_validated(query_vector, limit)

        # Populate caches so they're warm once the buffer is flushed
        self._set_exact_cache(cache_key, results)
        self._semantic_cache.append((query_vector, results, False, limit))
        await self.redis_client.set(cache_key, json.dumps(results))
        return results

    async def retrieve_and_rerank(self, query: str, limit: int = 5, retrieval_size: int = 50) -> list[str]:
        """
        Retrieval with cross-encoder reranking, backed by VectorDBManaged.

        When warm buffer is non-empty:
            - Bypasses all caches. Fetches 2x retrieval_size validated candidates via
            - _fetch_fresh_validated, reranks them, caches, and returns top-k.

        When warm buffer is empty:
            - Full L1 -> L2 -> L3 -> VectorDBManaged cache hierarchy (mirrors MemoryInterface
            - but queries vector_db_managed instead of vector_db).
        """
        assert retrieval_size >= limit, f"retrieval_size ({retrieval_size}) must be >= limit ({limit})"
        cache_key = self._make_cache_key(query, "reranked")

        if not self._warm_buffer:
            original_fetch_rs = self._cache_fetch_sizes.get(cache_key, 0)

            # L1
            if cache_key in self._exact_cache:
                cached_results = self._exact_cache[cache_key]
                db_exhausted = original_fetch_rs >= retrieval_size and len(cached_results) < original_fetch_rs
                if len(cached_results) >= retrieval_size or db_exhausted:
                    logger.info(f"[L1 cache] exact hit: {query}")
                    self._exact_cache.move_to_end(cache_key)
                    return cached_results[:limit]

            # L2 Redis
            async with async_timer("redis_get"):
                cached = await self.redis_client.get(cache_key)
            if cached:
                data = json.loads(cached)
                results, redis_fetch_rs = data["results"], data["fetch_rs"]
                db_exhausted = redis_fetch_rs >= retrieval_size and len(results) < redis_fetch_rs
                if len(results) >= retrieval_size or db_exhausted:
                    logger.info(f"[L2 cache] Redis hit: {query}")
                    self._set_exact_cache(cache_key, results, fetch_rs=redis_fetch_rs)
                    return results[:limit]

            async with async_timer("embed_text"):
                embeddings = self.embedding_client.embed_text([query], task_type="RETRIEVAL_QUERY")
            if not embeddings:
                return []
            query_vector = embeddings[0]

            # L3 semantic cache
            semantic_hit = self._find_semantic_cache_hit(query_vector, rerank=True, size_needed=retrieval_size)
            if semantic_hit:
                logger.info(f"[L3 cache] semantic hit: {query}")
                async with async_timer("rerank"):
                    reranked = self._rerank(query, semantic_hit)
                self._set_exact_cache(cache_key, reranked, fetch_rs=retrieval_size)
                await self.redis_client.set(cache_key, json.dumps({"results": reranked, "fetch_rs": retrieval_size}))
                return reranked[:limit]

            # DB miss — query VectorDBManaged
            logger.info(f"no cache hit, retrieving from managed db: {query}")
            async with async_timer("find_similar_extended"):
                db_rows = await find_similar_extended(query_vector, self.main_db_engine, retrieval_size)
            results = [row["text"] for row in db_rows]
            async with async_timer("rerank"):
                reranked = self._rerank(query, results)
            self._set_exact_cache(cache_key, reranked, fetch_rs=retrieval_size)
            self._semantic_cache.append((query_vector, reranked, True, retrieval_size))
            await self.redis_client.set(cache_key, json.dumps({"results": reranked, "fetch_rs": retrieval_size}))
            return reranked[:limit]

        # warm buffer active — bypass caches
        logger.info(f"[warm buffer active] retrieve_and_rerank bypassing caches: '{query}'")
        async with async_timer("embed_text"):
            embeddings = self.embedding_client.embed_text([query], task_type="RETRIEVAL_QUERY")
        if not embeddings:
            return []
        query_vector = embeddings[0]

        candidates = await self._fetch_fresh_validated(query_vector, retrieval_size)
        async with async_timer("rerank"):
            reranked = self._rerank(query, candidates)

        # Populate caches
        self._set_exact_cache(cache_key, reranked, fetch_rs=retrieval_size)
        self._semantic_cache.append((query_vector, reranked, True, retrieval_size))
        await self.redis_client.set(cache_key, json.dumps({"results": reranked, "fetch_rs": retrieval_size}))
        return reranked[:limit]

    # ── periodic flush lifecycle ────────────────────────────────────────────────

    async def _run_periodic_flush(self, interval_seconds: int) -> None:
        """
        Internal coroutine that sleeps 'interval_seconds' then flushes the warm buffer,
        repeating until cancelled. Errors are logged but do not stop the loop.
        """
        while True:
            try:
                await asyncio.sleep(interval_seconds)
                count = await self.flush_warm_buffer()
                if count > 0:
                    logger.info(f"[periodic flush] {count} warm-buffer entries synced to vector DB")
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"[periodic flush] unexpected error: {exc}")

    def start_periodic_flush(self, interval_seconds: int = 300) -> None:
        """
        Starts the background periodic flush task.
        Safe to call multiple times; does nothing if a task is already running.
        Intended to be called from lifespan on service startup.
        """
        if self._flush_task and not self._flush_task.done():
            logger.warning("[periodic flush] task already running; ignoring start request")
            return
        self._flush_task = asyncio.create_task(self._run_periodic_flush(interval_seconds))
        logger.info(f"[periodic flush] started (interval={interval_seconds}s)")

    async def stop_periodic_flush(self) -> None:
        """
        Cancels and awaits the background periodic flush task.
        Intended to be called from lifespan on service shutdown.
        """
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        logger.info("[periodic flush] stopped")
