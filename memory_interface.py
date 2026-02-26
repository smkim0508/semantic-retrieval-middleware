# the main memory retrieval interface
import numpy as np
from collections import deque, OrderedDict
from sqlalchemy.engine import Engine
from models.embeddings.gemini_embedding_client import GenAITextEmbeddingClient
from db.crud import find_similar
from typing import Optional

class MemoryInterface:
    """
    Wraps the embedding client and vector db to retrieve semantically similar text.
    Caches:
    1. Exact match cache - skips embedding client entirely on identical query texts (LRU via OrderedDict, max 50)
    2. Semantic cache — skips db retrieval if a sufficiently similar query was seen before (FIFO deque, max 10)
    """

    def __init__(self, embedding_client: GenAITextEmbeddingClient, main_db_engine: Engine):
        self.embedding_client = embedding_client
        self.main_db_engine = main_db_engine
        self._exact_cache: OrderedDict[str, list[str]] = OrderedDict()
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
        """
        va, vb = np.array(a), np.array(b)
        return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))

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

    def retrieve(self, query: str, limit: int = 5) -> list[str]:
        """
        Embeds the natural language query and returns the top-k most similar texts from the db.
        NOTE: task_type is set to RETRIEVAL_QUERY since gemini embeddings prefer diff content types for diff tasks.
        """
        # 1) exact match — skip embedding entirely
        if query in self._exact_cache:
            print(f"Exact cache hit: {query}")
            # if exact cache hit directly, just promote to most recently used spot in cache
            self._exact_cache.move_to_end(query)
            return self._exact_cache[query]

        # otherwise, embed the query
        embeddings = self.embedding_client.embed_text([query], task_type="RETRIEVAL_QUERY")
        if not embeddings:
            return []
        query_vector = embeddings[0]

        # 2) semantic cache — skip db retrieval if similar query was seen before
        # NOTE: current helper loops through all the cached vectors, but it is possible to implement this via numpy matrix multiplication to one-shot all cosine similarities
        # - above optimization not yet implemented since cache size is negligibly small (most case) and may be beneficial if recent cache computed first and returns
        semantic_cache_result = self._find_semantic_cache_hit(query_vector)
        if semantic_cache_result:
            print(f"Semantic cache hit on: {query}")
            # NOTE: if we have a semantic cache hit, we also promote to exact cache w/ similar vector results (not exact)
            # - this is logical as even without this promotion, the next query will be the same semantic cache hit anyways
            self._set_exact_cache(query, semantic_cache_result)
            return semantic_cache_result

        # 3) cache miss — retrieve from db and populate both caches
        results = find_similar(query_vector=query_vector, engine=self.main_db_engine, limit=limit)
        self._set_exact_cache(query, results)
        self._semantic_cache.append((query_vector, results))
        return results
