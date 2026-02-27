# timing utilities for async coroutines
# measures wall-clock execution time of individual async calls (not cumulative await time)

import time
from functools import wraps
from contextlib import asynccontextmanager
from typing import Callable, AsyncGenerator
from common.logger import logger

# NOTE: these are just wrappers to help timing simpler

def atimed(label: str | None = None) -> Callable:
    """
    Decorator that measures and logs the wall-clock execution time of an async function.
    Usage:
        @atimed("embed_text")
        async def embed_text(...): ...

        @atimed()  # uses function's qualified name as label
        async def my_func(...): ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            name = label or func.__qualname__
            start = time.perf_counter()
            result = await func(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.info(f"[timer] {name}: {elapsed_ms:.2f}ms")
            return result
        return wrapper
    return decorator

@asynccontextmanager
async def async_timer(label: str) -> AsyncGenerator[None, None]:
    """
    Async context manager that measures and logs wall-clock time of a block.
    Usage:
        async with async_timer("find_similar"):
            results = await find_similar(...)
    """
    start = time.perf_counter()
    yield
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(f"[timer] {label}: {elapsed_ms:.2f}ms")
