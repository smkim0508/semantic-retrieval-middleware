# test to retrieve top-k similar texts from the db
from dotenv import load_dotenv
from pathlib import Path
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.model import VectorDB
from models.embeddings.gemini_embedding_client import GenAITextEmbeddingClient

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

    engine = create_engine(MAIN_DB_URL)
    embedding_client = GenAITextEmbeddingClient(api_key=os.getenv("GEMINI_API_KEY"))

    print(f"Query: {sample_query}")
    embeddings = embedding_client.embed_text([sample_query], task_type="RETRIEVAL_QUERY")

    if not embeddings:
        print("Embedding failed, no result returned.")
        exit(1)

    query_vector = embeddings[0]

    with sessionmaker(engine)() as session:
        rows = (
            session.query(VectorDB.text)
            .order_by(VectorDB.vector.cosine_distance(query_vector))
            .limit(top_k)
            .all()
        )

    results = [row.text for row in rows]

    if not results:
        print("No results returned.")
        exit(0)

    print(f"Requested for top {top_k} on query: {sample_query}, fetched {len(results)} results: {results}")
