# the main memory retrieval interface
import json
import numpy as np
from collections import deque, OrderedDict
import redis.asyncio as aioredis
from sqlalchemy.ext.asyncio import AsyncEngine
from models.embeddings.gemini_embedding_client import GenAITextEmbeddingClient
from models.reranker.cross_encoder import CEReranker
from db.crud import find_similar
from typing import Optional
from common.logger import logger

class MemoryInterface:
    """
    Wraps the embedding client and vector db to retrieve semantically similar text.
    Caches:
    1. Exact match cache - skips embedding client entirely on identical query texts (LRU via OrderedDict, max 50)
    2. Semantic cache — skips db retrieval if a sufficiently similar query was seen before (FIFO deque, max 10)

    NOTE:
    - semantic cache deque stores (vector, results, reranked_bool) tuples
        - reranked_bool prevents cross-contamination between plain and reranked entries
        - len(results) encodes the candidate pool size (retrieval_size for reranked, limit for plain)
    - L1/L2 cache keys are namespaced: plain::{query} and reranked::{query}
        - upward compatible: len(cached) >= size_needed check allows larger cached entries to serve smaller requests
        - on miss due to insufficient size, DB re-fetch overwrites the entry with the larger result set
    """

    def __init__(self, embedding_client: GenAITextEmbeddingClient, main_db_engine: AsyncEngine, redis_client: aioredis.Redis, cross_encoder_reranker: CEReranker):
        # clients/engines
        self.embedding_client = embedding_client
        self.main_db_engine = main_db_engine
        self.redis_client = redis_client
        self.cross_encoder_reranker = cross_encoder_reranker
        # caches and cache settings
        self._exact_cache: OrderedDict[str, list[str]] = OrderedDict()  # L1: in-memory LRU
        self._semantic_cache: deque[tuple[list[float], list[str], bool]] = deque(maxlen=10) # (query_vector, results, reranked_bool)
        self._cosine_similarity_threshold = 0.90 # threshold for semantic cache
        self._exact_cache_max = 50 # threshold for max number of items in exact query cache
    
    # utils for caches below
    def _make_cache_key(self, query: str, namespace: str) -> str:
        """
        Simple helper to format the cache key for L1 and L2 caches.
        Namespace separates retrieve_plain ("plain") from retrieve_and_rerank ("reranked").
        Neither limit nor retrieval_size is included — the len(cached) >= size_needed check at
        read time makes entries upward-compatible across different limit/retrieval_size values.
        """
        return f"{namespace}::{query}"

    def _set_exact_cache(self, key: str, value: list[str]) -> None:
        """
        Simple helper to insert or or update elements in LRU exact cache, evicting the oldest entry if at capacity.
        """
        if key in self._exact_cache:
            self._exact_cache.move_to_end(key)
        self._exact_cache[key] = value
        if len(self._exact_cache) > self._exact_cache_max:
            self._exact_cache.popitem(last=False) # evict LRU

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """
        Simple helper to compute cosine similarity between two vectors using numpy.
        NOTE: prevents zero-divisions; not likely in real scenarios (real embeddings), but possible when testing.
        """
        va, vb = np.array(a), np.array(b)
        norm_a, norm_b = np.linalg.norm(va), np.linalg.norm(vb)
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(np.dot(va, vb) / (norm_a * norm_b))

    def _find_semantic_cache_hit(self, query_vector: list[float], rerank: bool, size_needed: int) -> Optional[list[str]]:
        """
        Simple helper to loop through semantic cache to find query hit via cos. sim. threshold.
        - rerank: must match the cached entry's reranked_bool to prevent cross-contamination
        - size_needed: len(cached_results) must be >= size_needed to ensure sufficient candidate pool
            - for retrieve_and_rerank: size_needed = retrieval_size
            - for retrieve_plain / retrieve: size_needed = limit
        - returns the cached results if hit, None otherwise
        NOTE: since the max size of semantic cache is small (10), we find the best match after iterating through the entire list
        """
        best_results = None
        best_score = -1.0
        for cached_vector, cached_results, cached_reranked in self._semantic_cache:
            if cached_reranked != rerank:
                continue
            if len(cached_results) < size_needed:
                continue
            score = self._cosine_similarity(query_vector, cached_vector)
            if score >= self._cosine_similarity_threshold and score > best_score:
                best_score = score
                best_results = cached_results
        return best_results

    # main retrieval methods
    # two separate retrieval methods for rerank vs not
    async def retrieve(self, query: str, limit: int = 5) -> list[str]:
        """
        Baseline retrieval without reranking. 
        Embeds the natural language query and returns the top-k most similar texts from the DB.
    
        Cache hierarchy:
            L1: in-memory LRU (OrderedDict) — fastest, ephemeral
            L2: Redis — persistent across restarts (cache promotion should be both L1 and L2, since redis persists)
            L3: semantic cache (deque) - skips DB if a similar query was seen, ephemeral
            fallback: vector DB query
        NOTE: separate cache exists for reranked results
        NOTE: task_type is set to RETRIEVAL_QUERY since gemini embeddings prefer diff content types for diff tasks.
        """
        cache_key = self._make_cache_key(query, "plain")

        # 1) L1 — upward compatible: serve if cached has enough results
        if cache_key in self._exact_cache:
            cached_results = self._exact_cache[cache_key]
            if len(cached_results) >= limit:
                logger.info(f"[L1 cache] exact hit: {query}")
                self._exact_cache.move_to_end(cache_key)
                return cached_results[:limit]

        # 2) L2 Redis — upward compatible: serve if cached has enough results
        cached = await self.redis_client.get(cache_key)
        if cached:
            results = json.loads(cached)
            if len(results) >= limit:
                logger.info(f"[L2 cache] Redis hit: {query}")
                self._set_exact_cache(cache_key, results)
                return results[:limit]

        # otherwise, embed the query
        embeddings = self.embedding_client.embed_text([query], task_type="RETRIEVAL_QUERY")
        if not embeddings:
            return []
        query_vector = embeddings[0]

        # 3) semantic cache — skip db retrieval if similar query was seen before
        semantic_cache_result = self._find_semantic_cache_hit(query_vector, rerank=False, size_needed=limit)
        if semantic_cache_result:
            logger.info(f"[L3 cache] semantic hit: {query}")
            self._set_exact_cache(cache_key, semantic_cache_result)
            await self.redis_client.set(cache_key, json.dumps(semantic_cache_result))
            return semantic_cache_result[:limit]

        # 4) cache miss — retrieve from db and populate all caches
        logger.info(f"no cache hit, retrieving from db: {query}")
        results = await find_similar(query_vector=query_vector, engine=self.main_db_engine, limit=limit)
        self._set_exact_cache(cache_key, results)
        self._semantic_cache.append((query_vector, results, False))
        await self.redis_client.set(cache_key, json.dumps(results))
        return results

    async def retrieve_and_rerank(self, query: str, limit: int = 5, retrieval_size: int = 50) -> list[str]:
        """
        Retrieval with cross-encoder reranking. Returns top-k results reranked by CE score.
        Uses the full 3-tier cache.
        NOTE: caches hold already-reranked results, so only L3 hits re-rank on the exact query.
        NOTE: task_type is set to RETRIEVAL_QUERY since gemini embeddings prefer diff content types for diff tasks.
        """
        assert retrieval_size >= limit, f"retrieval_size ({retrieval_size}) must be >= limit ({limit})"
        cache_key = self._make_cache_key(query, "reranked")

        # 1) L1 — upward compatible: serve if cached has enough candidates to satisfy retrieval_size
        if cache_key in self._exact_cache:
            cached_results = self._exact_cache[cache_key]
            if len(cached_results) >= retrieval_size:
                logger.info(f"[L1 cache] exact hit: {query}")
                # NOTE: update LRU if there's a cache hit, as long as size is sufficiently large
                self._exact_cache.move_to_end(cache_key)
                return cached_results[:limit]

        # 2) L2 Redis — upward compatible: serve if cached has enough candidates to satisfy retrieval_size
        cached = await self.redis_client.get(cache_key)
        if cached:
            results = json.loads(cached)
            if len(results) >= retrieval_size:
                logger.info(f"[L2 cache] Redis hit: {query}")
                self._set_exact_cache(cache_key, results)
                return results[:limit]

        # otherwise, embed the query
        embeddings = self.embedding_client.embed_text([query], task_type="RETRIEVAL_QUERY")
        if not embeddings:
            return []
        query_vector = embeddings[0]

        # 3) semantic cache — valid if reranked entry with enough candidates; re-rank on exact query
        semantic_cache_result = self._find_semantic_cache_hit(query_vector, rerank=True, size_needed=retrieval_size)
        if semantic_cache_result:
            logger.info(f"[L3 cache] semantic hit: {query}")
            reranked = self._rerank(query, semantic_cache_result)
            self._set_exact_cache(cache_key, reranked)
            await self.redis_client.set(cache_key, json.dumps(reranked))
            return reranked[:limit]

        # 4) cache miss — retrieve from db, rerank, populate all caches
        logger.info(f"no cache hit, retrieving from db: {query}")
        results = await find_similar(query_vector=query_vector, engine=self.main_db_engine, limit=retrieval_size)
        reranked = self._rerank(query, results)
        # NOTE: stores all reranked results to caches.
        self._set_exact_cache(cache_key, reranked)
        self._semantic_cache.append((query_vector, reranked, True))
        await self.redis_client.set(cache_key, json.dumps(reranked))
        # returns just the top-k requested
        return reranked[:limit]

    # helper methods for reranking
    def _rerank(self, query, docs) -> list[str]:
        """
        Thin wrapper around cross-encoder reranker.
        - Takes a query and list of docs and returns reranked list results (list[str]).
        """
        pairs = [(query, doc) for doc in docs]
        reranked_results = self.cross_encoder_reranker.rerank(pairs)
        # parse just the docs from reranked results and return
        return [doc for _, doc in reranked_results]
    