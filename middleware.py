"""
Request ID Middleware.

Every incoming request gets a unique ID.
This ID appears in every log line for that request.

So instead of:
  INFO | Fetching weather
  INFO | Running optimizer
  ERROR | Weather API failed   ← which request was this???

You get:
  INFO | req_a3f9 | Fetching weather
  INFO | req_a3f9 | Running optimizer
  ERROR | req_a3f9 | Weather API failed  ← NOW you know exactly
"""

import uuid
import time
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from loguru import logger
from logger_setup import request_id_ctx


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Generate short unique ID for this request
        request_id = f"req_{uuid.uuid4().hex[:6]}"
        request.state.request_id = request_id

        # Store in context var so logger can access it
        token = request_id_ctx.set(request_id)

        start_time = time.time()

        # Bind request_id to all logs within this request
        with logger.contextualize(request_id=request_id):
            logger.info(f"→ {request.method} {request.url.path}")

            try:
                response: Response = await call_next(request)
                duration_ms = round((time.time() - start_time) * 1000, 2)
                logger.info(
                    f"← {request.method} {request.url.path} "
                    f"[{response.status_code}] {duration_ms}ms"
                )
                response.headers["X-Request-ID"] = request_id
                return response

            except Exception as e:
                duration_ms = round((time.time() - start_time) * 1000, 2)
                logger.error(
                    f"← {request.method} {request.url.path} "
                    f"[500] {duration_ms}ms — {str(e)}"
                )
                raise

            finally:
                request_id_ctx.reset(token)