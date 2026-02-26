# the main memory retrieval interface
import numpy as np
from sqlalchemy.engine import Engine
from models.embeddings.gemini_embedding_client import GenAITextEmbeddingClient
from db.crud import find_similar

class MemoryInterface:
    """
    Wraps the embedding client and vector db to retrieve semantically similar text.
    Caches:
    1. Exact match cache - skips embedding client entirely on identical query texts
    2. Semantic cache — skips db retrieval if a sufficiently similar query was seen before

    TODO: optimizations via LRU and deque
    """
    def __init__(self, embedding_client: GenAITextEmbeddingClient, main_db_engine: Engine):
        self.embedding_client = embedding_client
        self.main_db_engine = main_db_engine
        self._exact_cache: dict[str, list[str]] = {}
        self._semantic_cache: list[tuple[list[float], list[str]]] = [] # query_vector, results tuple
        self._cosine_similarity_threshold = 0.90 # threshold for semantic cache

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """
        Simple helper to compute cosine similarity between two vectors using numpy
        """
        va, vb = np.array(a), np.array(b)
        return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))

    def retrieve(self, query: str, limit: int = 5) -> list[str]:
        """
        Embeds the natural language query and returns the top-k most similar texts from the db.
        NOTE: task_type is set to RETRIEVAL_QUERY since gemini embeddings prefer diff content types for diff tasks.
        """
        # 1) exact match — skip embedding entirely
        if query in self._exact_cache:
            print(f"Exact cache hit: {query}")
            return self._exact_cache[query]

        # otherwise, embed the query
        embeddings = self.embedding_client.embed_text([query], task_type="RETRIEVAL_QUERY")
        if not embeddings:
            return []
        query_vector = embeddings[0]

        # 2) semantic cache — skip db retrieval if similar query was seen before
        # NOTE: loops through all the cached vectors, but it is possible to implement this via numpy matrix multiplication to one-shot all cosine similarities
        # - above optimization not yet implemented since cache size is negligibly small (most case) and may be beneficial if recent cache computed first and returns
        for cached_vector, cached_results in self._semantic_cache:
            if self._cosine_similarity(query_vector, cached_vector) >= self._cosine_similarity_threshold:
                print(f"Semantic cache hit: {query}")
                # NOTE: if we have a semantic cache hit, we also promote to exact cache w/ similar vector results (not exact)
                # - this is logical as even without this promotion, the next query will be the same semantic cache hit anyways
                self._exact_cache[query] = cached_results
                return cached_results

        # 3) cache miss — retrieve from db and populate both caches
        results = find_similar(query_vector=query_vector, engine=self.main_db_engine, limit=limit)
        self._exact_cache[query] = results
        self._semantic_cache.append((query_vector, results))
        return results
