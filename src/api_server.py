"""FastAPI server for Medical RAG pipeline."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from .config import (
    API_INFLIGHT_ACQUIRE_TIMEOUT_MS,
    API_INFLIGHT_DRAIN_POLL_SECONDS,
    API_EXECUTOR_MAX_WORKERS,
    API_MAX_INFLIGHT_REQUESTS,
    API_SHUTDOWN_GRACE_SECONDS,
    CORS_ALLOWED_ORIGINS,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pipeline = None
_shutdown_event = threading.Event()
_inflight_lock = threading.Lock()
_inflight_requests = 0
_stream_threads_lock = threading.Lock()
_stream_threads: set[threading.Thread] = set()
_default_executor: Optional[ThreadPoolExecutor] = None
_request_capacity_semaphore: Optional[asyncio.Semaphore] = (
    asyncio.Semaphore(API_MAX_INFLIGHT_REQUESTS) if API_MAX_INFLIGHT_REQUESTS > 0 else None
)


def make_json_serializable(obj: Any) -> Any:
    """Recursively convert objects to JSON-safe values."""
    import math
    import numpy as np

    if obj is None:
        return None

    try:
        if isinstance(obj, (float, np.floating)):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return float(obj)
    except (TypeError, ValueError):
        pass

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, (str, int, bool)):
        return obj

    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]

    if hasattr(obj, "to_dict"):
        try:
            return make_json_serializable(obj.to_dict())
        except Exception:
            pass

    if isinstance(obj, np.ndarray):
        return make_json_serializable(obj.tolist())

    if hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes)):
        try:
            iter(obj)
            return [make_json_serializable(item) for item in obj]
        except (TypeError, ValueError):
            pass

    if hasattr(obj, "__dict__"):
        try:
            return make_json_serializable(obj.__dict__)
        except Exception:
            pass

    try:
        return str(obj)
    except Exception:
        return None


class ChatRequest(BaseModel):
    query: str
    stream: bool = False
    thread_context: Optional[dict[str, Any]] = None


class ChatResponse(BaseModel):
    query: str
    report_title: str
    answer: str
    follow_up_questions: list[str] = Field(default_factory=list)
    sections: list = Field(default_factory=list)
    sources: list = Field(default_factory=list)
    evidence_hierarchy: dict = Field(default_factory=dict)
    full_text_articles: list = Field(default_factory=list)
    retrieval_stats: dict = Field(default_factory=dict)
    status: str


class AuthenticatedService(BaseModel):
    service_id: str


def _followup_cap_exceeded(thread_context: Optional[dict[str, Any]]) -> bool:
    if not isinstance(thread_context, dict):
        return False
    is_follow_up = bool(thread_context.get("is_follow_up"))
    try:
        follow_up_count = int(thread_context.get("follow_up_count", 0))
    except (TypeError, ValueError):
        follow_up_count = 0
    return is_follow_up and follow_up_count >= 3


def _current_inflight_requests() -> int:
    with _inflight_lock:
        return _inflight_requests


def _request_error_detail(request: Request, message: str) -> dict[str, str]:
    return {
        "message": message,
        "request_id": getattr(request.state, "request_id", ""),
        "retry_hint": "Retry with exponential backoff.",
    }


def _release_request_capacity_slot() -> None:
    if _request_capacity_semaphore is not None:
        _request_capacity_semaphore.release()


async def _acquire_request_capacity_slot(request: Request) -> int:
    if _request_capacity_semaphore is None:
        return 0

    queue_wait_start = time.perf_counter()
    timeout_seconds = max(0.001, API_INFLIGHT_ACQUIRE_TIMEOUT_MS / 1000.0)
    try:
        await asyncio.wait_for(_request_capacity_semaphore.acquire(), timeout=timeout_seconds)
    except asyncio.TimeoutError as exc:
        queue_wait_ms = int((time.perf_counter() - queue_wait_start) * 1000)
        logger.warning(
            json.dumps(
                {
                    "event": "request_backpressure_reject",
                    "request_id": getattr(request.state, "request_id", ""),
                    "endpoint": request.url.path,
                    "queue_wait_ms": queue_wait_ms,
                    "acquire_timeout_ms": API_INFLIGHT_ACQUIRE_TIMEOUT_MS,
                    "max_inflight_requests": API_MAX_INFLIGHT_REQUESTS,
                },
                sort_keys=True,
            )
        )
        raise HTTPException(
            status_code=429,
            detail=_request_error_detail(
                request,
                "Server is handling maximum in-flight requests. Please retry shortly.",
            ),
        ) from exc

    queue_wait_ms = int((time.perf_counter() - queue_wait_start) * 1000)
    logger.info(
        json.dumps(
            {
                "event": "request_backpressure_acquired",
                "request_id": getattr(request.state, "request_id", ""),
                "endpoint": request.url.path,
                "queue_wait_ms": queue_wait_ms,
                "max_inflight_requests": API_MAX_INFLIGHT_REQUESTS,
            },
            sort_keys=True,
        )
    )
    return queue_wait_ms


async def require_service_auth(
    request: Request,
) -> AuthenticatedService:
    request.state.service_id = "production-app"
    return AuthenticatedService(service_id="production-app")


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Initialize shared resources at startup and drain in-flight traffic on shutdown."""
    global pipeline, _default_executor
    _shutdown_event.clear()
    logger.info("Starting Medical RAG API server")
    _default_executor = ThreadPoolExecutor(
        max_workers=API_EXECUTOR_MAX_WORKERS,
        thread_name_prefix="api-executor",
    )
    asyncio.get_running_loop().set_default_executor(_default_executor)
    logger.info(
        json.dumps(
            {
                "event": "api_startup_concurrency_config",
                "api_executor_max_workers": API_EXECUTOR_MAX_WORKERS,
                "api_max_inflight_requests": API_MAX_INFLIGHT_REQUESTS,
                "api_inflight_acquire_timeout_ms": API_INFLIGHT_ACQUIRE_TIMEOUT_MS,
            },
            sort_keys=True,
        )
    )

    skip_pipeline_init = os.getenv("SKIP_PIPELINE_INIT", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    if not skip_pipeline_init:
        try:
            from .rag_pipeline import MedicalRAGPipeline

            pipeline = MedicalRAGPipeline()
            logger.info("Pipeline initialized")
        except Exception as exc:
            logger.error("Failed to initialize pipeline: %s", exc)
            raise
    else:
        logger.warning("Skipping pipeline initialization (SKIP_PIPELINE_INIT is enabled)")

    yield

    logger.info("Shutdown requested, draining in-flight requests")
    _shutdown_event.set()
    deadline = time.monotonic() + max(1, API_SHUTDOWN_GRACE_SECONDS)

    while time.monotonic() < deadline and _current_inflight_requests() > 0:
        time.sleep(max(0.05, API_INFLIGHT_DRAIN_POLL_SECONDS))

    with _stream_threads_lock:
        stream_threads = list(_stream_threads)

    for thread in stream_threads:
        remaining = max(0.0, deadline - time.monotonic())
        if remaining <= 0:
            break
        thread.join(timeout=remaining)

    remaining_inflight = _current_inflight_requests()
    if remaining_inflight > 0:
        logger.warning(
            "Shutdown grace period elapsed with %d in-flight request(s)",
            remaining_inflight,
        )
    if _default_executor is not None:
        _default_executor.shutdown(wait=False)
        _default_executor = None
    logger.info("Shutdown complete")


app = FastAPI(
    title="Elixir AI Medical RAG API",
    description="Versioned Medical RAG API",
    version="1.1.0",
    openapi_url="/api/v1/openapi.json",
    docs_url="/api/v1/docs",
    redoc_url=None,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """Attach request ids and emit structured per-request logs."""
    global _inflight_requests

    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    request.state.request_id = request_id

    with _inflight_lock:
        _inflight_requests += 1

    start_time = time.perf_counter()
    status_code = 500
    response = None
    unhandled_error = None

    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    except Exception as exc:
        unhandled_error = type(exc).__name__
        raise
    finally:
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        log_event = {
            "request_id": request_id,
            "service_id": getattr(request.state, "service_id", "anonymous"),
            "endpoint": request.url.path,
            "method": request.method,
            "status": status_code,
            "latency_ms": elapsed_ms,
            "stream_flag": request.url.path.endswith("/stream"),
            "upstream_error_type": unhandled_error,
        }
        logger.info(json.dumps(log_event, default=str, sort_keys=True))
        with _inflight_lock:
            _inflight_requests = max(0, _inflight_requests - 1)


@app.get("/api/v1/health", tags=["health"])
@app.get("/health", include_in_schema=False)
async def health_check():
    return {
        "status": "healthy",
        "pipeline_ready": pipeline is not None,
        "shutting_down": _shutdown_event.is_set(),
    }


@app.post("/api/v1/chat", response_model=ChatResponse, tags=["chat"])
@app.post("/api/chat", include_in_schema=False)
async def chat(
    request: Request,
    payload: ChatRequest,
    _auth: AuthenticatedService = Depends(require_service_auth),
):
    if _shutdown_event.is_set():
        raise HTTPException(status_code=503, detail="Server is shutting down")

    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    if _followup_cap_exceeded(payload.thread_context):
        raise HTTPException(
            status_code=400,
            detail="Follow-up limit reached for this thread. Start a new thread to continue.",
        )

    slot_acquired = False
    try:
        await _acquire_request_capacity_slot(request)
        slot_acquired = True
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            pipeline.answer,
            payload.query,
            payload.thread_context,
        )
        return make_json_serializable(result)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Chat error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=_request_error_detail(request, "Chat generation failed"),
        )
    finally:
        if slot_acquired:
            _release_request_capacity_slot()


@app.post("/api/v1/chat/stream", tags=["chat"])
@app.post("/api/chat/stream", include_in_schema=False)
async def chat_stream(
    request: Request,
    payload: ChatRequest,
    _auth: AuthenticatedService = Depends(require_service_auth),
):
    if _shutdown_event.is_set():
        raise HTTPException(status_code=503, detail="Server is shutting down")

    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    if _followup_cap_exceeded(payload.thread_context):
        raise HTTPException(
            status_code=400,
            detail="Follow-up limit reached for this thread. Start a new thread to continue.",
        )

    request_id = getattr(request.state, "request_id", "")
    await _acquire_request_capacity_slot(request)
    slot_released = False

    def _release_slot_once() -> None:
        nonlocal slot_released
        if not slot_released:
            _release_request_capacity_slot()
            slot_released = True

    async def event_generator():
        queue: asyncio.Queue[Any] = asyncio.Queue()
        loop = asyncio.get_running_loop()
        stop_event = threading.Event()

        def producer() -> None:
            try:
                for event in pipeline.answer_streaming(payload.query, payload.thread_context):
                    if _shutdown_event.is_set() or stop_event.is_set():
                        break
                    asyncio.run_coroutine_threadsafe(queue.put(event), loop)
            except Exception as exc:
                logger.error("Streaming producer error: %s", exc, exc_info=True)
                asyncio.run_coroutine_threadsafe(
                    queue.put(
                        {
                            "step": "error",
                            "status": "error",
                            "message": "Streaming generation failed",
                            "request_id": request_id,
                            "retry_hint": "Retry with exponential backoff.",
                        }
                    ),
                    loop,
                )
            finally:
                asyncio.run_coroutine_threadsafe(queue.put("[DONE]"), loop)

        thread = threading.Thread(target=producer, name=f"stream-{request_id}")
        with _stream_threads_lock:
            _stream_threads.add(thread)
        thread.start()

        try:
            while True:
                if await request.is_disconnected():
                    stop_event.set()
                    break

                if _shutdown_event.is_set():
                    stop_event.set()

                try:
                    event = await asyncio.wait_for(queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    if stop_event.is_set() and not thread.is_alive():
                        break
                    continue

                if event == "[DONE]":
                    break

                serializable_event = make_json_serializable(event)
                yield f"data: {json.dumps(serializable_event)}\n\n"

            yield "data: [DONE]\n\n"
        except Exception as exc:
            logger.error("Streaming consumer error: %s", exc, exc_info=True)
            failure_event = {
                "step": "error",
                "status": "error",
                "message": "Stream terminated unexpectedly",
                "request_id": request_id,
                "retry_hint": "Retry with exponential backoff.",
            }
            yield f"data: {json.dumps(failure_event)}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            stop_event.set()
            await asyncio.to_thread(thread.join, 1.0)
            with _stream_threads_lock:
                _stream_threads.discard(thread)
            _release_slot_once()
    try:
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            background=BackgroundTask(_release_slot_once),
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    except Exception:
        _release_slot_once()
        raise


@app.post("/api/v1/debug/decompose", tags=["debug"])
@app.post("/api/query/decompose", include_in_schema=False)
async def decompose_query(
    request: Request,
    payload: ChatRequest,
    _auth: AuthenticatedService = Depends(require_service_auth),
):
    if _shutdown_event.is_set():
        raise HTTPException(status_code=503, detail="Server is shutting down")

    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    slot_acquired = False
    try:
        await _acquire_request_capacity_slot(request)
        slot_acquired = True
        result = pipeline.preprocess_query(payload.query)
        return {
            "original_query": result.original_query,
            "primary_query": result.primary_query,
            "keyword_query": result.keyword_query,
            "retrieval_queries": result.retrieval_queries,
            "decomposed": result.decomposed.model_dump() if result.decomposed else None,
        }
    except Exception as exc:
        logger.error("Decomposition error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=_request_error_detail(request, "Query decomposition failed"),
        )
    finally:
        if slot_acquired:
            _release_request_capacity_slot()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
