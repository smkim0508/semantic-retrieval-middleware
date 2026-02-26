# gemini text embedding client

from google import genai
from google.genai import types
from google.genai.types import ContentEmbedding
import os
from pathlib import Path
from dotenv import load_dotenv

from typing import Type, List, Optional

# task types
from models.embeddings.embedding_task_types import VALID_GEMINI_TASK_TYPES

# TODO: if using proper, async embedding calls in the future, use tenacity for retries

class GenAITextEmbeddingClient():
    """
    Core Google GenAI Text Embedding Client.
    """
    def __init__(
        self,
        model_name: str = "gemini-embedding-001", # google genai's default text embedding model
        content_type: str = "RETRIEVAL_DOCUMENT", # choose to differ embedding style based on desired tasks
        embedding_size: int = 1536, # NOTE: use at max 1536 embeddings for now, since pgvector supports upto 2000 dims. Default is 3072, could be explored later.
        *,
        api_key: str | None = None,
    ):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.content_type = content_type # content/task type to specialized embeddings
        self.embedding_size = embedding_size

    def embed_text(self, contents: list[str]) -> Optional[list[list[float]]]:
        """
        Simple helper to embed a list of text strings using gemini client.
        """
        result = self.client.models.embed_content(
            model=self.model_name,
            contents=contents,
            # NOTE: content type changes the embedding vectors as well
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
        )

        if result and result.embeddings:
            print(f"Embedding successful!")
            return [e.values for e in result.embeddings if e.values is not None]
        
        return None
