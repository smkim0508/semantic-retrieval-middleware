from contextlib import asynccontextmanager, AsyncExitStack
from fastapi import FastAPI
import redis.asyncio as aioredis
from core.config import get_service_settings
from common.logger import logger
from db.session import create_db_engine_context, parse_db_settings_from_service, DBSettings, DBType
from models.embeddings.gemini_embedding_client import GenAITextEmbeddingClient
from memory_interface import MemoryInterface

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the service's startup and shutdown events.
    Uses the AsyncExitStack to clean up resources.
    Register resources to the app state to be used as dependencies.

    NOTE:
    - Use stack.enter_async_context when the resource has __aenter__ and __aexit__ support
    - Use stack.push_async_context to register the clean up method only
    """
    # default start up message
    logger.info(f"Starting semantic retrieval middleware!")

    # initialize resources during start up
    logger.info("Initializing service resources...")
    settings = get_service_settings()

    async with AsyncExitStack() as stack:
        
        # TODO: add any initialization code here

        # Main db engine
        # parse main db settings, and create engine
        main_db_settings = parse_db_settings_from_service(settings, DBType.MainDB)
        app.state.main_db_engine = await stack.enter_async_context(
            create_db_engine_context(
                db_settings=main_db_settings
            )
        )
        logger.info("Main database engine initialized.")

        # Text embedding models, for now only Google GenAI
        gemini_text_embedding_client = GenAITextEmbeddingClient(api_key=settings.GEMINI_API_KEY)
        app.state.gemini_text_embedding_client = gemini_text_embedding_client
        logger.info(f"Text embedding client (GOOGLE GENAI) initialized.")

        # Redis client for persistent cache
        redis_client = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
        app.state.redis_client = redis_client
        stack.push_async_callback(redis_client.aclose)
        logger.info(f"Redis client initialized at {settings.REDIS_URL}.")

        # Memory retriever interface
        memory_retriever = MemoryInterface(
            main_db_engine=app.state.main_db_engine,
            embedding_client=gemini_text_embedding_client,
            redis_client=redis_client,
        )
        app.state.memory_retriever = memory_retriever
        logger.info(f"Memory retriever initialized.")

        try:
            # lets FastAPI process requests during yield
            yield
        finally:
            # explicit resource clean up, otherwise automatically cleaned via exit stack
            logger.info("Shutting down service resources...")
            # TODO: add any explicit cleanup / shutdown code here

        # The AsyncExitStack will automatically call the __aexit__ or registered cleanup
        # methods for all resources entered or pushed to it, in reverse order.
        logger.info("All global resources have been gracefully closed.")
