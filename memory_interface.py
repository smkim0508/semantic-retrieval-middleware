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
    """

    def __init__(self, embedding_client: GenAITextEmbeddingClient, main_db_engine: AsyncEngine, redis_client: aioredis.Redis, cross_encoder_reranker: CEReranker):
        # clients/engines
        self.embedding_client = embedding_client
        self.main_db_engine = main_db_engine
        self.redis_client = redis_client
        self.cross_encoder_reranker = cross_encoder_reranker
        # caches and cache settings
        self._exact_cache: OrderedDict[str, list[str]] = OrderedDict()  # L1: in-memory LRU
        self._semantic_cache: deque[tuple[list[float], list[str]]] = deque(maxlen=10) # query_vector, results tuple
        self._cosine_similarity_threshold = 0.90 # threshold for semantic cache
        self._exact_cache_max = 50 # threshold for max number of items in exact query cache
    
    # utils for caches below
    def _make_cache_key(self, query: str, limit: int) -> str:
        """
        Simple helper to format the cache key for L1 and L2 caches.
        """
        return f"{query}::{limit}"

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

    def _find_semantic_cache_hit(self, query_vector: list[float]) -> Optional[list[str]]:
        """
        Simple helper to loop through semantic cache to find query hit via cos. sim. threshold.
        - If cache hit, returns just the cached similar text results
        - returns None if no semantic cache hit
        """
        for cached_vector, cached_results in self._semantic_cache:
            if self._cosine_similarity(query_vector, cached_vector) >= self._cosine_similarity_threshold:
                return cached_results
        return None

    # main retrieval methods
    async def retrieve(self, query: str, limit: int = 5, rerank: bool = True) -> list[str]:
        """
        Embeds the natural language query and returns the top-k most similar texts from the db.
        Cache hierarchy:
            L1: in-memory LRU (OrderedDict) — fastest, ephemeral
            L2: Redis — persistent across restarts (cache promotion should be both L1 and L2, since redis persists)
            L3: semantic cache (deque) - skips DB if a similar query was seen, ephemeral
            fallback: vector DB query
            NOTE: caches hold reranked results.

        This retrieval method also (optionally) reranks the results using a cross-encoder reranker before returning.
        - NOTE: one caveat, initial retrieval only fetches top k requested results (limit) and reranks.
        - This is not necessarily the top k most similar texts post-reranking from the entire DB.

        NOTE: task_type is set to RETRIEVAL_QUERY since gemini embeddings prefer diff content types for diff tasks.
        """
        cache_key = self._make_cache_key(query, limit)

        # 1) L1 exact match — skip embedding entirely
        if cache_key in self._exact_cache:
            logger.info(f"[L1 cache] exact hit: {query}")
            self._exact_cache.move_to_end(cache_key)
            return self._exact_cache[cache_key]

        # 2) L2 Redis exact match — persistent across restarts, still skips embedding
        cached = await self.redis_client.get(cache_key)
        if cached:
            logger.info(f"[L2 cache] Redis hit: {query}")
            results = json.loads(cached)
            self._set_exact_cache(cache_key, results) # promote to L1
            return results

        # otherwise, embed the query
        embeddings = self.embedding_client.embed_text([query], task_type="RETRIEVAL_QUERY")
        if not embeddings:
            return []
        query_vector = embeddings[0]

        # 3) semantic cache — skip db retrieval if similar query was seen before
        # NOTE: current helper loops through all the cached vectors, but it is possible to implement this via numpy matrix multiplication to one-shot all cosine similarities
        # - above optimization not yet implemented since cache size is negligibly small (most case) and may be beneficial if recent cache computed first and returns
        semantic_cache_result = self._find_semantic_cache_hit(query_vector)
        if semantic_cache_result:
            logger.info(f"[L3 cache] semantic hit: {query}")
            # NOTE: for semantic cache only, we re-rank again on the exact query, since cached results are reranked on a different query.
            semantic_cache_result = self._rerank(query, semantic_cache_result) if rerank else semantic_cache_result
            # promote to both exact caches so future identical queries skip embedding
            self._set_exact_cache(cache_key, semantic_cache_result)
            await self.redis_client.set(cache_key, json.dumps(semantic_cache_result))
            return semantic_cache_result

        # 4) cache miss — retrieve from db and populate all caches
        logger.info(f"no cache hit, retrieving from db: {query}")
        results = await find_similar(query_vector=query_vector, engine=self.main_db_engine, limit=limit)
        # rerank results via cross-encoder and save them to caches
        results = self._rerank(query, results) if rerank else results
        logger.info(f"saving retrieved results to cache for {query}")
        self._set_exact_cache(cache_key, results)
        self._semantic_cache.append((query_vector, results))
        await self.redis_client.set(cache_key, json.dumps(results))
        return results
    
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
    