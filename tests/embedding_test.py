# test to embed a sample text and store it in vector db via ORM mapping
from dotenv import load_dotenv
from pathlib import Path
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.model import VectorDB
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
        exit(1)

    vector = embeddings[0]
    print(f"Embedded vector length: {len(vector)}")

    obj = VectorDB(vector=vector, text=sample_text)
    with sessionmaker(engine)() as session:
        session.add(obj)
        session.commit()
        session.refresh(obj)

    print(f"Stored row {obj.id} successfully.")
