# the main memory retrieval interface
import json
import numpy as np
from collections import deque, OrderedDict
import redis.asyncio as aioredis
from sqlalchemy.ext.asyncio import AsyncEngine
from models.embeddings.gemini_embedding_client import GenAITextEmbeddingClient
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

    def __init__(self, embedding_client: GenAITextEmbeddingClient, main_db_engine: AsyncEngine, redis_client: aioredis.Redis):
        self.embedding_client = embedding_client
        self.main_db_engine = main_db_engine
        self.redis_client = redis_client
        self._exact_cache: OrderedDict[str, list[str]] = OrderedDict()  # L1: in-memory LRU
        self._semantic_cache: deque[tuple[list[float], list[str]]] = deque(maxlen=10) # query_vector, results tuple
        self._cosine_similarity_threshold = 0.90 # threshold for semantic cache
        self._exact_cache_max = 50 # threshold for max number of items in exact query cache

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
        NOTE: prevents zero-divisions.
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

    async def retrieve(self, query: str, limit: int = 5) -> list[str]:
        """
        Embeds the natural language query and returns the top-k most similar texts from the db.
        Cache hierarchy:
          L1: in-memory LRU (OrderedDict) — fastest, ephemeral
          L2: Redis — persistent across restarts (cache promotion should be both L1 and L2, since redis persists)
          L3: semantic cache (deque) - skips DB if a similar query was seen, ephemeral
          fallback: vector DB query
        NOTE: task_type is set to RETRIEVAL_QUERY since gemini embeddings prefer diff content types for diff tasks.
        """
        # 1) L1 exact match — skip embedding entirely
        if query in self._exact_cache:
            logger.info(f"[L1 cache] exact hit: {query}")
            self._exact_cache.move_to_end(query)
            return self._exact_cache[query]

        # 2) L2 Redis exact match — persistent across restarts, still skips embedding
        cached = await self.redis_client.get(query)
        if cached:
            logger.info(f"[L2 cache] Redis hit: {query}")
            results = json.loads(cached)
            self._set_exact_cache(query, results)  # promote to L1
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
            # promote to both exact caches so future identical queries skip embedding
            self._set_exact_cache(query, semantic_cache_result)
            await self.redis_client.set(query, json.dumps(semantic_cache_result))
            return semantic_cache_result

        # 4) cache miss — retrieve from db and populate all caches
        results = await find_similar(query_vector=query_vector, engine=self.main_db_engine, limit=limit)
        self._set_exact_cache(query, results)
        self._semantic_cache.append((query_vector, results))
        await self.redis_client.set(query, json.dumps(results))
        return results
