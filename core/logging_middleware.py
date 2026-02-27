import uuid
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from common.logger import session_id_var

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """
        This middleware is called for every incoming HTTP request.
        It generates a unique ID and sets it in the context variable.
        NOTE: this should not be confused with background task ids.
        """
        # Generate a unique ID for the request
        request_id = str(uuid.uuid4())
        session_id_var.set(f"request_id:{request_id}")
        
        # Process the request
        response = await call_next(request)
        
        # Add the request ID to the response headers so it can be traced
        # from the client or in logs from other systems (like a load balancer)
        response.headers["X-Request-ID"] = request_id

        # Clear the context variable
        session_id_var.set("")
        
        return response
