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
from mlx_omni_server.chat.gpu_lock import mlx_generation_lock

router = APIRouter(tags=["chat—completions"])


class ModelNotLoadedError(Exception):
    """Raised when requested model is not loaded in cache."""
    pass


@router.post("/chat/completions", response_model=ChatCompletionResponse)
@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    """Create a chat completion (non-blocking)"""

    # Try to get model - will raise error if not loaded
    try:
        with mlx_generation_lock:
            text_model = _create_text_model(
                request.model,
                request.get_extra_params().get("adapter_path"),
                request.get_extra_params().get("draft_model"),
            )
    except ModelNotLoadedError as e:
        return JSONResponse(
            content={
                "error": {
                    "message": str(e),
                    "type": "invalid_request_error",
                    "code": "model_not_loaded"
                }
            },
            status_code=400
        )

    if not request.stream:
        # Run synchronous generate in thread pool to avoid blocking
        # Lock ensures only one generation runs at a time on GPU
        def generate_with_lock():
            with mlx_generation_lock:
                return text_model.generate(request)
        completion = await asyncio.to_thread(generate_with_lock)
        return JSONResponse(content=completion.model_dump(exclude_none=True))

    # For streaming, collect all chunks synchronously then yield
    async def async_event_generator() -> AsyncGenerator[str, None]:
        def generate_all_chunks():
            """Generate all chunks synchronously while holding lock"""
            chunks = []
            with mlx_generation_lock:
                for chunk in text_model.generate_stream(request):
                    chunks.append(chunk)
            return chunks

        # Run generation in thread pool to not block event loop
        all_chunks = await asyncio.to_thread(generate_all_chunks)

        import logging
        logger = logging.getLogger("mlx-studio.router")
        logger.info(f"[ROUTER] Yielding {len(all_chunks)} chunks to client")

        chunk_count = 0
        for chunk in all_chunks:
            chunk_count += 1
            data = chunk.model_dump(exclude_none=True)
            chunk_str = f"data: {json.dumps(data)}\n\n"

            # Log last 3 chunks with more detail
            if chunk_count >= len(all_chunks) - 2:
                logger.warning(f"[ROUTER] Chunk {chunk_count}/{len(all_chunks)}: {chunk_str[:200]}...")

            yield chunk_str

        logger.warning("[ROUTER] Sending DONE marker to client")
        yield "data: [DONE]\n\n"
        logger.warning("[ROUTER] Stream complete - all data sent")

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
    require_loaded: bool = False,
):
    """Create a text model based on the model parameters.

    Uses the shared wrapper cache to get or create ChatGenerator instance.
    This avoids expensive model reloading when the same model configuration
    is used across different requests or API endpoints.

    Supports both MLX and GGUF backends - automatically dispatches based on
    model configuration or file extension.

    Args:
        model_id: Model name/path
        adapter_path: Optional LoRA adapter path
        draft_model: Optional draft model for speculative decoding
        require_loaded: If True, raise error if model not already loaded.
                       If False (default), auto-load the model on demand.

    Returns:
        OpenAIAdapter for MLX backend or GGUFOpenAIAdapter for GGUF backend
    """
    import logging
    logger = logging.getLogger(__name__)
    backend = "mlx"  # Default backend

    # IMPORTANT: Resolve alias FIRST before checking if loaded
    # This allows aliases like "claude-haiku-*" to map to loaded models
    try:
        from patches import resolve_alias_with_backend, get_draft_model_for, reload_aliases
        # Reload aliases to pick up any changes made via API
        reload_aliases()
        original_model_id = model_id
        model_id, backend = resolve_alias_with_backend(model_id)
        if model_id != original_model_id:
            logger.info(f"Resolved alias '{original_model_id}' -> '{model_id}' (backend={backend})")
        # Also get draft model from config if not provided (only for MLX)
        if backend == "mlx" and draft_model is None:
            draft_model = get_draft_model_for(original_model_id)
            if draft_model:
                from patches import resolve_alias
                draft_model = resolve_alias(draft_model)
    except ImportError:
        pass  # patches module not available

    # Resolve to local path if available (MLX Studio local model lookup)
    # This ensures the cache key matches what was used in /api/models/load
    try:
        from extensions.models import ModelManager
        _model_manager = ModelManager()
        local_models = _model_manager.list_local_models()
        local_model = next((m for m in local_models if m.id == model_id), None)
        if local_model and local_model.path:
            logger.info(f"Resolved to local path: {model_id} -> {local_model.path}")
            model_id = local_model.path
    except ImportError:
        pass  # extensions not available

    # =========================================================================
    # GGUF Backend (via llama-server)
    # =========================================================================
    if backend == "gguf":
        try:
            from extensions.gguf_backend import gguf_server, GGUFBackend, load_gguf_config
            from extensions.gguf_openai_adapter import GGUFOpenAIAdapter

            config = load_gguf_config()

            # Auto-start llama-server if needed
            if config.get("auto_start", True):
                if not gguf_server.is_running() or gguf_server.current_model != model_id:
                    logger.info(f"Auto-starting llama-server with model: {model_id}")
                    try:
                        gguf_server.start(model_id)
                    except Exception as e:
                        raise ModelNotLoadedError(
                            f"Failed to start llama-server for GGUF model '{model_id}': {e}"
                        )
            elif not gguf_server.is_running():
                raise ModelNotLoadedError(
                    f"GGUF model '{model_id}' requires llama-server to be running. "
                    f"Start it via POST /api/gguf/start?model_path={model_id}"
                )

            # Create GGUF backend and adapter
            gguf_backend = GGUFBackend(gguf_server.server_url)
            return GGUFOpenAIAdapter(gguf_backend)

        except ImportError as e:
            logger.error(f"GGUF backend not available: {e}")
            raise ModelNotLoadedError(
                f"GGUF backend not available. Install dependencies or use MLX model."
            )

    # =========================================================================
    # MLX Backend (default)
    # =========================================================================
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

    # Create OpenAIAdapter with the cached wrapper directly
    return OpenAIAdapter(wrapper=wrapper)


# Legacy caching variables removed - now using shared wrapper_cache
# This eliminates duplicate caching logic and enables sharing between endpoints
