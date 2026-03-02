# script to fill vector db with randomly generated embeddings
import uuid
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os
from tqdm import tqdm

from db.model import VectorDB

VECTOR_DIM = 1536

def generate_unit_vector(dim: int) -> list[float]:
    v = np.random.randn(dim).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()

BATCH_SIZE = 500

def prefill(n: int, engine):
    """
    Fills the DB with randomly generated embeddings (np.random) and text (uuid).
    - Used to test DB performance with large number of rows.
    """
    Session = sessionmaker(bind=engine)
    with Session() as session:
        for offset in tqdm(range(0, n, BATCH_SIZE), desc="Inserting batches"):
            batch_size = min(BATCH_SIZE, n - offset)
            rows = [
                VectorDB(vector=generate_unit_vector(VECTOR_DIM), text=str(uuid.uuid4()))
                for _ in range(batch_size)
            ]
            session.add_all(rows)
            session.commit()
    print(f"Successfully inserted {n} rows!")

if __name__ == "__main__":
    load_dotenv()
    MAIN_DB_USER = os.getenv("MAIN_DB_USER")
    MAIN_DB_PW   = os.getenv("MAIN_DB_PW")
    MAIN_DB_HOST = os.getenv("MAIN_DB_HOST")
    MAIN_DB_PORT = os.getenv("MAIN_DB_PORT")
    MAIN_DB_NAME = os.getenv("MAIN_DB_NAME")

    DB_URL = f"postgresql+psycopg2://{MAIN_DB_USER}:{MAIN_DB_PW}@{MAIN_DB_HOST}:{MAIN_DB_PORT}/{MAIN_DB_NAME}?sslmode=require"
    engine = create_engine(DB_URL)

    N = 10000  # number of rows to insert
    prefill(N, engine)
