# CRUD operations on the vector db
from db.model import VectorDB

# using sync session for testing now, async TBD
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import Session
from sqlalchemy.engine import Engine

engine = create_engine("postgresql+psycopg2://user:pass@localhost/db")

def find_similar(query_vector: list[float], engine: Engine, limit: int = 5) -> list[str]:
    """
    Returns the text of the top-k most similar vectors to query_vector, ranked by cosine similarity.
    """
    with sessionmaker(engine)() as session:
        rows = (
            session.query(VectorDB.text)
            .order_by(VectorDB.vector.cosine_distance(query_vector))
            .limit(limit)
            .all()
        )
    return [row.text for row in rows]

def store_vector(vector: list[float], text: str, engine: Engine) -> VectorDB:
    """
    Takes a text string and its vector embedding and stores it in the vector db
    """
    obj = VectorDB(vector=vector, text=text)
    with sessionmaker(engine)() as session:
        session.add(obj)
        session.commit()
        session.refresh(obj)
    return obj
