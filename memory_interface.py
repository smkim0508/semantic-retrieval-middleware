# the main memory retrieval interface
from sqlalchemy.engine import Engine
from models.embeddings.gemini_embedding_client import GenAITextEmbeddingClient
from db.crud import find_similar

class MemoryInterface:
    """
    Wraps the embedding client and vector db to retrieve semantically similar text.
    - Uses exact match cache as first check before hitting the embedding API + db.
    """
    def __init__(self, embedding_client: GenAITextEmbeddingClient, main_db_engine: Engine):
        self.embedding_client = embedding_client
        self.main_db_engine = main_db_engine
        # local exact query match cache; used to skip embedding query vector if hit
        self._exact_cache: dict[str, list[str]] = {}

    def retrieve(self, query: str, limit: int = 5) -> list[str]:
        """
        Embeds the natural language query and returns the top-k most similar texts from the db.
        NOTE: task_type is set to RETRIEVAL_QUERY since gemini embeddings prefer diff content types for diff tasks.
        
        Caches:
        1. Uses exact match cache as first check before hitting the embedding client
        2. TODO: semantic cache to prevent vector retrieval from embedding client when hit with previously similar queries
        """
        # first check the exact match cache
        if query in self._exact_cache:
            print(f"Cache hit for query: {query}")
            return self._exact_cache[query]

        # otherwise, embed using client
        embeddings = self.embedding_client.embed_text([query], task_type="RETRIEVAL_QUERY")
        if not embeddings:
            return []
        query_vector = embeddings[0]

        # then find the similar texts from vector db
        # TODO: implement semantic cache before this
        results = find_similar(query_vector=query_vector, engine=self.main_db_engine, limit=limit)
        # save the results to local caches (exact and semantic)
        self._exact_cache[query] = results
        return results
