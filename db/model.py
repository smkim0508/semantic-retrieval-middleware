# baseline ORM model for the vector db table using SQLAlchemy
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Text, DateTime, Index, Column, Integer
from pgvector.sqlalchemy import Vector
from datetime import datetime
from typing import Optional

# A dedicated Base for all models that map to tables in the main database.
class MainDB_Base(DeclarativeBase):
    pass

class VectorDB(MainDB_Base):
    """
    Simple vector db table, used to experiment with semantic retrieval
    """
    __tablename__ = "vector_db"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    vector = Column(Vector(1536), nullable=False)
    text = Column(Text, nullable=False)

    # index for faster similarity search, NOTE: only on cosine similarity for now
    __table_args__ = (
        Index(
            "ix_vector_db_hnsw",
            "vector",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"vector": "vector_cosine_ops"},
        ),
    )
