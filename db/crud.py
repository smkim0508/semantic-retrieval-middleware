# CRUD operations on the vector db
from db.model import VectorDB, VectorDBManaged, GroundTruth
from db.session import get_async_session_maker

from sqlalchemy import select, delete, update
from sqlalchemy.ext.asyncio import AsyncEngine

async def find_similar(query_vector: list[float], engine: AsyncEngine, limit: int = 5) -> list[str]:
    """
    Returns the text of the top-k most similar vectors to query_vector, ranked by cosine similarity.
    """
    async with get_async_session_maker(engine)() as session:
        result = await session.execute(
            select(VectorDB.text)
            .order_by(VectorDB.vector.cosine_distance(query_vector))
            .limit(limit)
        )
    return [row.text for row in result.all()]

async def find_similar_extended(
    query_vector: list[float], engine: AsyncEngine, limit: int, offset: int = 0
) -> list[dict]:
    """
    Queries VectorDBManaged and returns dicts with {id, text, ground_truth_id} for each result.
    Used internally by the warm-buffer retrieval path for ground-truth validation.
    offset: skip this many top results (for top-up queries after the first pass).

    NOTE: for warm buffer retrieval
    """
    async with get_async_session_maker(engine)() as session:
        result = await session.execute(
            select(VectorDBManaged.id, VectorDBManaged.text, VectorDBManaged.ground_truth_id)
            .order_by(VectorDBManaged.vector.cosine_distance(query_vector))
            .limit(limit)
            .offset(offset)
        )
    return [
        {"id": row.id, "text": row.text, "ground_truth_id": row.ground_truth_id}
        for row in result.all()
    ]

async def store_vector(vector: list[float], text: str, engine: AsyncEngine) -> VectorDB:
    """
    Takes a text string and its vector embedding and stores it in the vector db.
    ground_truth_id is left NULL (legacy direct-store path; always treated as valid on retrieval).
    """
    obj = VectorDB(vector=vector, text=text)
    async with get_async_session_maker(engine)() as session:
        session.add(obj)
        await session.commit()
        await session.refresh(obj)
    return obj

async def create_ground_truth(text: str, engine: AsyncEngine) -> GroundTruth:
    """
    Creates a new GroundTruth row with is_synced=False.
    Called immediately when data enters via the warm-buffer path.
    
    NOTE: for warm buffer retrieval
    """
    obj = GroundTruth(text=text, is_synced=False)
    async with get_async_session_maker(engine)() as session:
        session.add(obj)
        await session.commit()
        await session.refresh(obj)
    return obj

async def get_ground_truth_sync_status(ids: list[int], engine: AsyncEngine) -> dict[int, bool]:
    """
    Batch-fetches the is_synced status for a list of ground_truth IDs.
    Returns a dict mapping {ground_truth_id: is_synced}.
    IDs not found in the table are omitted (caller treats them as invalid).

    NOTE: for warm buffer retrieval
    """
    if not ids:
        return {}
    async with get_async_session_maker(engine)() as session:
        result = await session.execute(
            select(GroundTruth.id, GroundTruth.is_synced)
            .where(GroundTruth.id.in_(ids))
        )
    return {row.id: row.is_synced for row in result.all()}

async def flush_warm_entries(entries: list[dict], engine: AsyncEngine) -> int:
    """
    Atomically flushes warm-buffer entries into the vector DB:
    1. Delete any existing VectorDB rows with the same ground_truth_id (stale entries from updates).
    2. Bulk-insert new VectorDB rows with ground_truth_id set.
    3. Mark the corresponding GroundTruth rows as is_synced=True.
    Returns the number of entries flushed.

    NOTE: for warm buffer retrieval
    """
    if not entries:
        return 0

    gt_ids = [e["ground_truth_id"] for e in entries]

    async with get_async_session_maker(engine)() as session:
        # 1. Remove stale VectorDBManaged rows for these ground_truth_ids (handles update case)
        await session.execute(
            delete(VectorDBManaged).where(VectorDBManaged.ground_truth_id.in_(gt_ids))
        )

        # 2. Insert fresh VectorDBManaged rows
        new_rows = [
            VectorDBManaged(vector=e["vector"], text=e["text"], ground_truth_id=e["ground_truth_id"])
            for e in entries
        ]
        session.add_all(new_rows)

        # 3. Mark ground truth rows as synced
        await session.execute(
            update(GroundTruth)
            .where(GroundTruth.id.in_(gt_ids))
            .values(is_synced=True)
        )

        await session.commit()

    return len(entries)
