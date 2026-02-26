# test to embed a sample text and store it in vector db via ORM mapping
from dotenv import load_dotenv
from pathlib import Path
import os

from sqlalchemy import create_engine

from db.crud import store_vector
from models.embeddings.gemini_embedding_client import GenAITextEmbeddingClient

# load env + construct db connection url
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

MAIN_DB_USER = os.getenv("MAIN_DB_USER")
MAIN_DB_PW = os.getenv("MAIN_DB_PW")
MAIN_DB_HOST = os.getenv("MAIN_DB_HOST")
MAIN_DB_PORT = os.getenv("MAIN_DB_PORT")
MAIN_DB_NAME = os.getenv("MAIN_DB_NAME")

MAIN_DB_URL = f"postgresql+psycopg2://{MAIN_DB_USER}:{MAIN_DB_PW}@{MAIN_DB_HOST}:{MAIN_DB_PORT}/{MAIN_DB_NAME}?sslmode=require"

if __name__ == "__main__":

    # test input
    sample_text = "The quick brown fox jumps over the lazy dog."

    engine = create_engine(MAIN_DB_URL)
    embedding_client = GenAITextEmbeddingClient(api_key=os.getenv("GEMINI_API_KEY"))

    print(f"Embedding: {sample_text}")
    embeddings = embedding_client.embed_text([sample_text])

    if not embeddings:
        print("Embedding failed, no result returned.")
        exit(1) # exit prematurely if embedding fails

    vector = embeddings[0] # we only gave one text in the list to embed
    print(f"Embedded vector length: {len(vector)}")

    row = store_vector(vector=vector, text=sample_text, engine=engine)
    print(f"Stored row {row.id} successfully.")
