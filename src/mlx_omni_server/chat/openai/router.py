import json
import asyncio
import queue
import threading
from typing import Generator, Optional, AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

from mlx_omni_server.chat.mlx.chat_generator import ChatGenerator
from mlx_omni_server.chat.openai.openai_adapter import OpenAIAdapter
from mlx_omni_server.chat.openai.schema import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)

router = APIRouter(tags=["chat—completions"])


@router.post("/chat/completions", response_model=ChatCompletionResponse)
@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    """Create a chat completion (non-blocking)"""

    text_model = _create_text_model(
        request.model,
        request.get_extra_params().get("adapter_path"),
        request.get_extra_params().get("draft_model"),
    )

    if not request.stream:
        # Run synchronous generate in thread pool to avoid blocking
        completion = await asyncio.to_thread(text_model.generate, request)
        return JSONResponse(content=completion.model_dump(exclude_none=True))

    # For streaming, use a queue to pass chunks from sync generator to async
    async def async_event_generator() -> AsyncGenerator[str, None]:
        chunk_queue = queue.Queue()
        done_sentinel = object()

        def sync_generator():
            try:
                for chunk in text_model.generate_stream(request):
                    chunk_queue.put(chunk)
            finally:
                chunk_queue.put(done_sentinel)

        # Start sync generator in background thread
        thread = threading.Thread(target=sync_generator, daemon=True)
        thread.start()

        # Yield chunks as they arrive (non-blocking)
        while True:
            try:
                # Poll queue with small timeout to stay async-friendly
                chunk = await asyncio.to_thread(chunk_queue.get, timeout=0.1)
                if chunk is done_sentinel:
                    break
                yield f"data: {json.dumps(chunk.model_dump(exclude_none=True))}\n\n"
            except queue.Empty:
                # No chunk yet, yield control to event loop
                await asyncio.sleep(0)
                continue

        yield "data: [DONE]\n\n"
        thread.join(timeout=1.0)

    return StreamingResponse(
        async_event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


def _create_text_model(
    model_id: str,
    adapter_path: Optional[str] = None,
    draft_model: Optional[str] = None,
) -> OpenAIAdapter:
    """Create a text model based on the model parameters.

    Uses the shared wrapper cache to get or create ChatGenerator instance.
    This avoids expensive model reloading when the same model configuration
    is used across different requests or API endpoints.
    """
    # Get cached or create new ChatGenerator
    wrapper = ChatGenerator.get_or_create(
        model_id=model_id,
        adapter_path=adapter_path,
        draft_model_id=draft_model,
    )

    # Create OpenAIAdapter with the cached wrapper directly
    return OpenAIAdapter(wrapper=wrapper)


# Legacy caching variables removed - now using shared wrapper_cache
# This eliminates duplicate caching logic and enables sharing between endpoints
