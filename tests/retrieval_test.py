# test to retrieve top-k similar texts from the db using MemoryInterface
from dotenv import load_dotenv
from pathlib import Path
import os

from sqlalchemy import create_engine

from models.embeddings.gemini_embedding_client import GenAITextEmbeddingClient
from memory_interface import MemoryInterface

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

MAIN_DB_USER = os.getenv("MAIN_DB_USER")
MAIN_DB_PW = os.getenv("MAIN_DB_PW")
MAIN_DB_HOST = os.getenv("MAIN_DB_HOST")
MAIN_DB_PORT = os.getenv("MAIN_DB_PORT")
MAIN_DB_NAME = os.getenv("MAIN_DB_NAME")

MAIN_DB_URL = f"postgresql+psycopg2://{MAIN_DB_USER}:{MAIN_DB_PW}@{MAIN_DB_HOST}:{MAIN_DB_PORT}/{MAIN_DB_NAME}?sslmode=require"

if __name__ == "__main__":
    # test inputs
    sample_query = "fox jumping over a duck"
    top_k = 5

    main_db_engine = create_engine(MAIN_DB_URL)
    embedding_client = GenAITextEmbeddingClient(api_key=os.getenv("GEMINI_API_KEY"))
    memory = MemoryInterface(embedding_client=embedding_client, main_db_engine=main_db_engine)

    print(f"Query: {sample_query}")
    results = memory.retrieve(query=sample_query, limit=top_k)

    if not results:
        print("No results returned.")
        exit(0) # exit prematurely if nothing returned

    print(f"Requested for top {top_k} on query: {sample_query}, fetched {len(results)} results: {results}")
