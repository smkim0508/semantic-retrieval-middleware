# script to fill vector db with randomly generated embeddings
import uuid
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os
from tqdm import tqdm

from db.model import VectorDB, VectorDBManaged, GroundTruth
from models.embeddings.gemini_embedding_client import GenAITextEmbeddingClient

VECTOR_DIM = 1536

def generate_unit_vector(dim: int) -> list[float]:
    v = np.random.randn(dim).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()

BATCH_SIZE = 500

def prefill(n: int, engine):
    """
    Fills both vector tables with randomly generated embeddings (np.random) and text (uuid).
    - VectorDB: plain insert (no ground-truth link) for the legacy retrieval path.
    - VectorDBManaged + GroundTruth: linked insert (is_synced=True) for the managed path.
    Same vector is shared between both table rows for each text.
    """
    Session = sessionmaker(bind=engine)
    with Session() as session:
        for offset in tqdm(range(0, n, BATCH_SIZE), desc="Inserting batches"):
            batch_size = min(BATCH_SIZE, n - offset)
            texts = [str(uuid.uuid4()) for _ in range(batch_size)]
            vectors = [generate_unit_vector(VECTOR_DIM) for _ in range(batch_size)]

            # VectorDB: plain insert, no ground-truth link (legacy path)
            session.add_all([
                VectorDB(vector=vec, text=text)
                for text, vec in zip(texts, vectors)
            ])

            # GroundTruth + VectorDBManaged: linked insert (managed path)
            gt_rows = [GroundTruth(text=text, is_synced=True) for text in texts]
            session.add_all(gt_rows)
            session.flush() # populate gt_rows[i].id without committing

            session.add_all([
                VectorDBManaged(vector=vec, text=gt.text, ground_truth_id=gt.id)
                for gt, vec in zip(gt_rows, vectors)
            ])
            session.commit()

    print(f"Successfully inserted {n} rows into VectorDB and VectorDBManaged!")

# test inputs designed to be simple but nuanced
SAMPLE_TEXTS = [
    "A quick brown fox jumps over the lazy dog.",
    "The fox leaps gracefully over a sleeping duck.",
    "A red fox hops across the pond near some ducks.",
    "The brown fox narrowly clears the duck sitting on the log.",
    "Three ducks scatter as a fox bounds through the reeds.",
    "A fox and a duck rest peacefully side by side by the river.",
    "The fox stalks quietly toward the unsuspecting duck.",
    "A young fox trips and tumbles over a startled duck.",
    "The duck dives into the water to escape the approaching fox.",
    "A fox circles the pond while two ducks swim slowly away.",
]

def prefill_real_embeddings(engine, embedding_client: GenAITextEmbeddingClient) -> None:
    """
    Embeds SAMPLE_TEXTS using the Gemini client and stores each into both vector tables.
    - VectorDB: plain insert (no ground-truth link) for the legacy retrieval path.
    - VectorDBManaged + GroundTruth: linked insert (is_synced=True) for the managed path.
    The texts are thematically similar (fox + duck) with subtle scene differences.
    - Useful for verifying that reranked results differ from plain retrieval.
    """
    vectors = embedding_client.embed_text(SAMPLE_TEXTS, task_type="RETRIEVAL_DOCUMENT")
    if not vectors:
        print("Embedding failed; aborting prefill_real_embeddings.")
        return
    if len(vectors) != len(SAMPLE_TEXTS):
        print(f"Expected {len(SAMPLE_TEXTS)} vectors, got {len(vectors)}; aborting.")
        return

    Session = sessionmaker(bind=engine)
    with Session() as session:
        for text, vector in zip(SAMPLE_TEXTS, vectors):
            # VectorDB: plain insert (legacy path)
            session.add(VectorDB(vector=vector, text=text))

            # VectorDBManaged: linked insert (managed path)
            gt = GroundTruth(text=text, is_synced=True)
            session.add(gt)
            session.flush() # populate gt.id before linking
            session.add(VectorDBManaged(vector=vector, text=text, ground_truth_id=gt.id))
        session.commit()

    print(f"Successfully inserted {len(SAMPLE_TEXTS)} real-embedding rows into VectorDB and VectorDBManaged.")

if __name__ == "__main__":
    load_dotenv()
    MAIN_DB_USER = os.getenv("MAIN_DB_USER")
    MAIN_DB_PW = os.getenv("MAIN_DB_PW")
    MAIN_DB_HOST = os.getenv("MAIN_DB_HOST")
    MAIN_DB_PORT = os.getenv("MAIN_DB_PORT")
    MAIN_DB_NAME = os.getenv("MAIN_DB_NAME")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    assert GEMINI_API_KEY, "GEMINI_API_KEY must be set in .env"

    DB_URL = f"postgresql+psycopg2://{MAIN_DB_USER}:{MAIN_DB_PW}@{MAIN_DB_HOST}:{MAIN_DB_PORT}/{MAIN_DB_NAME}?sslmode=require"
    engine = create_engine(DB_URL)

    embedding_client = GenAITextEmbeddingClient(api_key=GEMINI_API_KEY)

    N = 10000 # number of rows to insert
    prefill(N, engine)
    prefill_real_embeddings(engine, embedding_client)
