# used to log and measure latency, especially for async operations
# baseline logger modeled from my other projects
# TODO: implement async time-sensitive logging (or stick to sync)

import logging
import uuid
from contextvars import ContextVar
from typing import Dict

# The context variable that will store the unique ID for a given request or task
# Given a generic name because it will hold either a request_id or a task_id depending on the context.
# NOTE: import session_id_var and set it before using logging
session_id_var: ContextVar[str] = ContextVar("session_id", default="")

class SessionIdFilter(logging.Filter):
    """
    A logging filter that injects the correlation_id from the context variable
    into the log record.
    """
    def filter(self, record):
        # Retrieve the ID from the context variable and add it to the log record
        record.session_id = session_id_var.get() or "N/A" # Default to N/A if not set
        return True

def get_session_id() -> str:
    """
    Retrieves the current session ID from the context variable.
    """
    return session_id_var.get()

class CustomLogger:
    """A factory class for creating and configuring loggers."""
    _loggers: Dict[str, logging.Logger] = {}

    @classmethod
    def get_logger(cls, name: str = "app", log_level: int = logging.INFO) -> logging.Logger:
        if name in cls._loggers:
            return cls._loggers[name]

        logger = logging.getLogger(name)
        logger.setLevel(log_level)

        if logger.handlers:
            logger.handlers.clear()
        
        # Add our custom filter to all loggers created by this factory
        logger.addFilter(SessionIdFilter())
        
        # The log format now includes our custom correlation_id field
        log_format = logging.Formatter(
            '%(asctime)s - [%(session_id)s] - [%(filename)s:%(lineno)d] - %(name)s - %(levelname)s - %(message)s', 
            '%Y-%m-%d %H:%M:%S'
        )
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(log_format)
        
        logger.addHandler(console_handler)
        
        cls._loggers[name] = logger
        return logger
    
# Create a default logger instance that can be imported and used anywhere
# NOTE: import this logger to log with logger.info("message"), .warning("message"), .error("message")
logger = CustomLogger.get_logger()
