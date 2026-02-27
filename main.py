# main FastAPI app

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import Any, Optional, Dict
from common.logger import logger
from core.config import get_service_settings
from core.lifespan import lifespan

# dependencies
from core.dependencies import (
    get_main_db_engine,
    get_gemini_text_embedding_client
)

# logging middleware
from core.logging_middleware import LoggingMiddleware

# API / routes
# TODO

# allow docs for testing
docs_config: dict[str, Any] = {
    "docs_url": "/docs",
    "redoc_url": "/redoc",
    "openapi_url": "/openapi.json"
}

# main app, asgi entrypoint
app = FastAPI(
    title="Semantic-Retrieval-Middleware",
    description="Experimental service for testing latency-aware vector DB retrieval middleware",
    version="0.1.0",
    lifespan=lifespan,
    **docs_config,
)

# add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# add logging middleware for requests
app.add_middleware(LoggingMiddleware)

# test routes
# TODO

# test endpoint
@app.get("/", tags=["Application"])
async def root():
    return {"message": "Hello World"}
