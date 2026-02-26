# task types for task-specific embedding generation

from enum import Enum

# valid task types for text embedding models
# check for more: https://ai.google.dev/gemini-api/docs/embeddings#task-types

VALID_GEMINI_TASK_TYPES = frozenset({
    "RETRIEVAL_QUERY",
    "RETRIEVAL_DOCUMENT",
    "SEMANTIC_SIMILARITY",
    "CLASSIFICATION",
    "CLUSTERING",
    "QUESTION_ANSWERING",
    "FACT_VERIFICATION",
})
