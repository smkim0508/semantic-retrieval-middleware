# baseline ORM model for the vector db table using SQLAlchemy
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Text, DateTime, Index, Column, Integer, Boolean, ForeignKey
from pgvector.sqlalchemy import Vector
from datetime import datetime
from typing import Optional

# A dedicated Base for all models that map to tables in the main database.
class MainDB_Base(DeclarativeBase):
    pass

class GroundTruth(MainDB_Base):
    """
    Canonical structured store for items entering via the warm buffer path.
    Written immediately on store_via_warm_buffer(); no embedding/index.

    is_synced=False: item is pending in the warm buffer, not yet in vector_db.
    is_synced=True: item has been flushed from the warm buffer into vector_db.

    VectorDB rows reference this table via ground_truth_id. 
    - A VectorDB row whose ground_truth has is_synced=False is stale
    - (superseded by an update still pending in the warm buffer) and is dropped during retrieval)

    NOTE: for warm buffer retrieval
    """
    __tablename__ = "ground_truth"
    id = Column(Integer, primary_key=True, autoincrement=True)
    text = Column(Text, nullable=False)
    is_synced = Column(Boolean, nullable=False, default=False)

class VectorDB(MainDB_Base):
    """
    Simple vector db table, used to experiment with semantic retrieval.
    Legacy direct-store path — no ground-truth tracking.
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

class VectorDBManaged(MainDB_Base):
    """
    Managed vector DB table for the warm-buffer write path.
    Every row is linked to a GroundTruth row via non-nullable ground_truth_id.
    Valid only after the corresponding GroundTruth row has is_synced=True (set on flush).
    """
    __tablename__ = "vector_db_managed"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    vector = Column(Vector(1536), nullable=False)
    text = Column(Text, nullable=False)
    ground_truth_id = Column(Integer, ForeignKey("ground_truth.id"), nullable=False, index=True)

    # index for faster similarity search, NOTE: only on cosine similarity for now
    __table_args__ = (
        Index(
            "ix_vector_db_managed_hnsw",
            "vector",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"vector": "vector_cosine_ops"},
        ),
    )
