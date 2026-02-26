# the main memory retrieval interface
from sqlalchemy.engine import Engine
from models.embeddings.gemini_embedding_client import GenAITextEmbeddingClient
from db.crud import find_similar

class MemoryInterface:
    """
    Wraps the embedding client and vector db to retrieve semantically similar text.
    - NOTE: this is where cache will be implemented in the future.
    - i.e. only dispatches to db retrieval method if cache miss
    """
    def __init__(self, embedding_client: GenAITextEmbeddingClient, main_db_engine: Engine):
        self.embedding_client = embedding_client
        self.main_db_engine = main_db_engine

    def retrieve(self, query: str, limit: int = 5) -> list[str]:
        """
        Embeds the natural language query and returns the top-k most similar texts from the db.
        NOTE: task_type is set to RETRIEVAL_QUERY since gemini embeddings prefer diff content types for diff tasks.
        """
        embeddings = self.embedding_client.embed_text([query], task_type="RETRIEVAL_QUERY")
        if not embeddings:
            return [] # empty when no match, not error
        query_vector = embeddings[0]
        return find_similar(query_vector=query_vector, engine=self.main_db_engine, limit=limit)
