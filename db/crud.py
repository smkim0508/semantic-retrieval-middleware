# CRUD operations on the vector db
from db.model import VectorDB

# using sync session for testing now, async TBD
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import Session
from sqlalchemy.engine import Engine

engine = create_engine("postgresql+psycopg2://user:pass@localhost/db")

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
