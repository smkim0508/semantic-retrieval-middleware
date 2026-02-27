# CRUD operations on the vector db
from db.model import VectorDB
from db.session import get_async_session_maker

from sqlalchemy import select
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

async def store_vector(vector: list[float], text: str, engine: AsyncEngine) -> VectorDB:
    """
    Takes a text string and its vector embedding and stores it in the vector db.
    """
    obj = VectorDB(vector=vector, text=text)
    async with get_async_session_maker(engine)() as session:
        session.add(obj)
        await session.commit()
        await session.refresh(obj)
    return obj
