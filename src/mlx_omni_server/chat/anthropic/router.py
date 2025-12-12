import json
import logging
import asyncio
import queue
import threading
from typing import Any, Generator, Optional, AsyncGenerator
from pathlib import Path

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from mlx_omni_server.chat.anthropic.anthropic_messages_adapter import (
    AnthropicMessagesAdapter,
)

from ..mlx.chat_generator import ChatGenerator
from .anthropic_schema import MessagesRequest, MessagesResponse
from .models_service import AnthropicModelsService
from .schema import AnthropicModelList

logger = logging.getLogger(__name__)

# File logger for debugging Claude Code requests
_file_logger = None
def get_file_logger():
    global _file_logger
    if _file_logger is None:
        _file_logger = logging.getLogger("anthropic_debug")
        _file_logger.setLevel(logging.DEBUG)
        log_path = Path("/tmp/anthropic_debug.log")
        handler = logging.FileHandler(log_path, mode='w')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        _file_logger.addHandler(handler)
    return _file_logger
router = APIRouter(tags=["anthropic"])
models_service = AnthropicModelsService()

# Legacy caching variables removed - now using shared wrapper_cache
# This eliminates duplicate caching logic and enables sharing between endpoints


@router.get("/models", response_model=AnthropicModelList)
@router.get("/v1/models", response_model=AnthropicModelList)
async def list_anthropic_models(
    before_id: Optional[str] = Query(
        default=None,
        title="Before Id",
        description="ID of the object to use as a cursor for pagination. When provided, returns the page of results immediately before this object.",
    ),
    after_id: Optional[str] = Query(
        default=None,
        title="After Id",
        description="ID of the object to use as a cursor for pagination. When provided, returns the page of results immediately after this object.",
    ),
    limit: int = Query(
        default=20,
        ge=1,
        le=1000,
        title="Limit",
        description="Number of items to return per page. Defaults to 20. Ranges from 1 to 1000.",
    ),
) -> AnthropicModelList:
    """List available models in Anthropic format."""
    return models_service.list_models(
        limit=limit, after_id=after_id, before_id=before_id
    )


@router.post("/messages/count_tokens")
@router.post("/v1/messages/count_tokens")
async def count_tokens(raw_request: Request):
    """Count tokens for a messages request (stub implementation)."""
    try:
        body = await raw_request.json()
        # Return a stub response - actual token counting would require tokenizer
        # For now, estimate based on message length
        messages = body.get("messages", [])
        total_chars = sum(
            len(str(m.get("content", ""))) for m in messages
        )
        # Rough estimate: ~4 chars per token
        estimated_tokens = total_chars // 4 + 100  # Add base tokens for overhead

        return JSONResponse(content={
            "input_tokens": estimated_tokens
        })
    except Exception as e:
        logger.error(f"Failed to count tokens: {e}")
        return JSONResponse(content={"input_tokens": 1000})  # Fallback


@router.post("/messages")
@router.post("/v1/messages")
async def create_message(raw_request: Request):
    """Create an Anthropic Messages API completion"""
    flog = get_file_logger()

    # Parse and validate request manually to get better error messages
    try:
        body = await raw_request.json()
        flog.debug(f"=== NEW REQUEST ===")
        flog.debug(f"stream: {body.get('stream')}, model: {body.get('model')}, max_tokens: {body.get('max_tokens')}")
        flog.debug(f"messages count: {len(body.get('messages', []))}")
        flog.debug(f"tools count: {len(body.get('tools', []))}")
        # Log first message content (truncated)
        if body.get('messages'):
            first_msg = body['messages'][0]
            content = str(first_msg.get('content', ''))[:200]
            flog.debug(f"first message role: {first_msg.get('role')}, content: {content}...")
        request = MessagesRequest.model_validate(body)
    except ValidationError as e:
        flog.error(f"Validation error: {e}")
        logger.error(f"Validation error: {e}")
        # Log which fields failed
        for error in e.errors():
            flog.error(f"  Field: {error['loc']}, Error: {error['msg']}")
            logger.error(f"  Field: {error['loc']}, Error: {error['msg']}, Input type: {type(error.get('input'))}")
        raise
    except Exception as e:
        flog.error(f"Failed to parse request: {e}")
        logger.error(f"Failed to parse request: {e}")
        raise

    try:
        anthropic_model = _create_anthropic_model(
            request.model,
            # Extract extra params if needed - for now use defaults
            None,  # adapter_path
            None,  # draft_model
        )
    except ModelNotLoadedError as e:
        flog.error(f"Model not loaded: {e}")
        logger.error(f"Model not loaded: {e}")
        return JSONResponse(
            content={
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": str(e)
                }
            },
            status_code=400
        )

    if not request.stream:
        flog.debug(f"Non-streaming request: model={request.model}, max_tokens={request.max_tokens}")
        try:
            # Run synchronous generate in thread pool to avoid blocking event loop
            completion = await asyncio.to_thread(anthropic_model.generate, request)
            flog.debug(f"Non-streaming success: {completion.stop_reason}")
            return JSONResponse(content=completion.model_dump(exclude_none=True))
        except Exception as e:
            flog.error(f"Non-streaming generation failed: {e}")
            # Return a minimal valid response to prevent Claude Code from hanging
            return JSONResponse(
                content={
                    "id": f"msg_error",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": ""}],
                    "model": request.model,
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0}
                },
                status_code=200
            )

    flog.debug(f"Starting streaming request")

    # For streaming, use a queue to pass events from sync generator to async
    async def async_anthropic_event_generator() -> AsyncGenerator[str, None]:
        event_queue = queue.Queue()
        done_sentinel = object()
        event_count = 0

        def sync_generator():
            nonlocal event_count
            try:
                for event in anthropic_model.generate_stream(request):
                    event_queue.put(event)
                    event_count += 1
            finally:
                event_queue.put(done_sentinel)

        # Start sync generator in background thread
        thread = threading.Thread(target=sync_generator, daemon=True)
        thread.start()

        # Yield events as they arrive (non-blocking)
        local_count = 0
        while True:
            try:
                # Poll queue with small timeout to stay async-friendly
                event = await asyncio.to_thread(event_queue.get, timeout=0.1)
                if event is done_sentinel:
                    break

                local_count += 1
                # Use mode='json' to properly serialize enums to their string values
                event_data = event.model_dump(mode='json', exclude_none=True)

                # For message_start and message_delta, ensure stop_reason/stop_sequence are included as null
                if event.type.value == "message_start" and "message" in event_data:
                    event_data["message"].setdefault("stop_reason", None)
                    event_data["message"].setdefault("stop_sequence", None)
                elif event.type.value == "message_delta" and "delta" in event_data:
                    event_data["delta"].setdefault("stop_sequence", None)

                # Log important events
                if event.type.value in ["message_start", "message_delta", "message_stop"]:
                    logger.info(f"SSE [{local_count}] {event.type.value}: {json.dumps(event_data)[:200]}")
                elif event.type.value == "content_block_start":
                    logger.info(f"SSE [{local_count}] content_block_start: {event.content_block}")
                elif event.type.value == "content_block_delta" and local_count <= 3:
                    logger.info(f"SSE [{local_count}] content_block_delta: {json.dumps(event_data)}")

                yield f"event: {event.type.value}\n"
                yield f"data: {json.dumps(event_data)}\n\n"
            except queue.Empty:
                # No event yet, yield control to event loop
                await asyncio.sleep(0)
                continue

        logger.info(f"SSE stream finished with {local_count} events")
        thread.join(timeout=1.0)

    return StreamingResponse(
        async_anthropic_event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


class ModelNotLoadedError(Exception):
    """Raised when requested model is not loaded in cache."""
    pass


def _create_anthropic_model(
    model_id: str,
    adapter_path: Optional[str] = None,
    draft_model: Optional[str] = None,
    require_loaded: bool = True,
) -> AnthropicMessagesAdapter:
    """Create an Anthropic Messages adapter based on the model parameters.

    Uses the shared wrapper cache to get or create ChatGenerator instance.
    This avoids expensive model reloading when the same model configuration
    is used across different requests or API endpoints.

    Args:
        model_id: Model name/path
        adapter_path: Optional LoRA adapter path
        draft_model: Optional draft model for speculative decoding
        require_loaded: If True (default), raise error if model not already loaded.
                       This prevents concurrent model loading which crashes Metal.
    """
    # IMPORTANT: Resolve alias FIRST before checking if loaded
    # This allows aliases like "claude-haiku-*" to map to loaded models
    try:
        from patches import resolve_alias, get_draft_model_for, reload_aliases
        # Reload aliases to pick up any changes made via API
        reload_aliases()
        original_model_id = model_id
        model_id = resolve_alias(model_id)
        if model_id != original_model_id:
            logger.info(f"Resolved alias '{original_model_id}' -> '{model_id}'")
        # Also get draft model from config if not provided
        if draft_model is None:
            draft_model = get_draft_model_for(original_model_id)
            if draft_model:
                draft_model = resolve_alias(draft_model)
    except ImportError:
        pass  # patches module not available

    if require_loaded:
        # Only get if already loaded - prevents concurrent GPU access crashes
        wrapper = ChatGenerator.get_if_loaded(
            model_id=model_id,
            adapter_path=adapter_path,
            draft_model_id=draft_model,
        )
        if wrapper is None:
            raise ModelNotLoadedError(
                f"Model '{model_id}' is not loaded. "
                f"Load it first via POST /api/models/load?model_id={model_id}"
            )
    else:
        # Legacy behavior - load model if not cached
        wrapper = ChatGenerator.get_or_create(
            model_id=model_id,
            adapter_path=adapter_path,
            draft_model_id=draft_model,
        )

    # Create AnthropicMessagesAdapter with the cached wrapper directly
    return AnthropicMessagesAdapter(wrapper=wrapper)
