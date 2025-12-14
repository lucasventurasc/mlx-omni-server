"""
Smart Prompt Cache Management Module

This module provides an improved caching strategy inspired by:
- vLLM's hash-based Automatic Prefix Caching (APC)
- MLX-Textgen's multi-slot disk-based caching
- Anthropic's prompt caching with cache_control breakpoints

Key improvements over the original PromptCache:
1. Hash-based lookup instead of prefix matching only
2. Multiple cache slots for different prompt patterns
3. LRU eviction when memory is constrained
4. Separate caching for static components (system prompt, tools)
5. Disk persistence for KV cache (saves ~70s on 21K token prompts)

This is designed for Claude Code workloads where:
- System prompt + tools are ~15,000 tokens and rarely change
- Conversation history grows but shares common prefixes
- Different request types (haiku vs sonnet) may have different prompts
"""

import hashlib
import json
import os
import struct
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
from mlx_lm.models.cache import (
    KVCache,
    RotatingKVCache,
    can_trim_prompt_cache,
    make_prompt_cache,
    trim_prompt_cache,
)

from mlx_omni_server.chat.mlx.model_types import MLXModel

from ...utils.logger import logger


# Configuration via environment variables
DEFAULT_BLOCK_SIZE = int(os.getenv("MLX_CACHE_BLOCK_SIZE", "256"))
DEFAULT_MAX_SLOTS = int(os.getenv("MLX_CACHE_MAX_SLOTS", "4"))
DEFAULT_MIN_REUSE = int(os.getenv("MLX_CACHE_MIN_REUSE", "512"))
MAX_CACHED_TOKENS = int(os.getenv("MLX_CACHE_MAX_TOKENS", "65536"))  # 64K max per slot

# Disk cache configuration
DISK_CACHE_DIR = Path(os.getenv("MLX_DISK_CACHE_DIR", os.path.expanduser("~/.cache/mlx-studio/kv")))
DISK_CACHE_ENABLED = os.getenv("MLX_DISK_CACHE_ENABLED", "true").lower() == "true"
DISK_CACHE_MIN_TOKENS = int(os.getenv("MLX_DISK_CACHE_MIN_TOKENS", "4096"))  # Only save large prompts
DISK_CACHE_MAX_AGE_DAYS = int(os.getenv("MLX_DISK_CACHE_MAX_AGE_DAYS", "7"))  # Auto-delete after N days
DISK_CACHE_MAX_SIZE_GB = float(os.getenv("MLX_DISK_CACHE_MAX_SIZE_GB", "50"))  # Max total cache size


def compute_token_hash(tokens: List[int], prefix_hash: str = "") -> str:
    """
    Compute a hash for a sequence of tokens, optionally chained with a prefix hash.

    This follows vLLM's approach where each block's hash includes:
    - The hash of the previous block (prefix_hash)
    - The tokens in the current block

    Optimized to use bytes directly instead of string conversion.

    Args:
        tokens: List of token IDs
        prefix_hash: Hash of the preceding tokens (for chaining)

    Returns:
        A hex string hash (16 chars = 64 bits)
    """
    # Convert tokens to bytes efficiently (4 bytes per int32)
    # This is 2-3x faster than string conversion for large blocks
    token_bytes = struct.pack(f'{len(tokens)}i', *tokens)
    prefix_bytes = prefix_hash.encode() if prefix_hash else b''
    combined = prefix_bytes + b':' + token_bytes
    return hashlib.sha256(combined).hexdigest()[:16]


def compute_block_hashes(tokens: List[int], block_size: int = 256) -> List[str]:
    """
    Compute hashes for token blocks, where each hash depends on all previous blocks.

    This enables finding the longest matching prefix by comparing block hashes.

    Args:
        tokens: Full list of tokens
        block_size: Number of tokens per block (smaller = finer granularity)

    Returns:
        List of cumulative block hashes
    """
    hashes = []
    prefix_hash = ""

    for i in range(0, len(tokens), block_size):
        block = tokens[i:i + block_size]
        block_hash = compute_token_hash(block, prefix_hash)
        hashes.append(block_hash)
        prefix_hash = block_hash

    return hashes


def get_disk_cache_path(model_id: str, prompt_hash: str) -> Path:
    """Get the path for a disk-cached KV cache file."""
    # Sanitize model_id for filesystem
    safe_model_id = model_id.replace("/", "_").replace("\\", "_")
    return DISK_CACHE_DIR / f"{safe_model_id}_{prompt_hash}.safetensors"


def save_cache_to_disk(
    cache: List[Any],
    tokens: List[int],
    model_id: str,
    block_hashes: List[str],
) -> Optional[Path]:
    """
    Save KV cache to disk for later reuse.

    Args:
        cache: List of KVCache objects (one per layer)
        tokens: The token sequence this cache represents
        model_id: Model identifier
        block_hashes: Pre-computed block hashes

    Returns:
        Path to saved file, or None if save failed
    """
    if not DISK_CACHE_ENABLED:
        return None

    if len(tokens) < DISK_CACHE_MIN_TOKENS:
        logger.debug(f"Skipping disk cache: {len(tokens)} tokens < {DISK_CACHE_MIN_TOKENS} min")
        return None

    # Skip RotatingKVCache - not supported for disk caching
    if cache and any(isinstance(c, RotatingKVCache) for c in cache):
        logger.debug("Skipping disk cache: RotatingKVCache not supported")
        return None

    # Skip non-standard KVCache types (quantized, MoE, etc.)
    if cache and not all(type(c).__name__ == 'KVCache' for c in cache):
        cache_types = set(type(c).__name__ for c in cache)
        logger.debug(f"Skipping disk cache: non-standard cache types {cache_types}")
        return None

    try:
        # Use the last block hash as the prompt identifier
        prompt_hash = block_hashes[-1] if block_hashes else compute_token_hash(tokens)
        cache_path = get_disk_cache_path(model_id, prompt_hash)

        # Create directory if needed
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Build the data dict for saving
        save_data = {}

        for i, layer_cache in enumerate(cache):
            if hasattr(layer_cache, 'state') and layer_cache.keys is not None:
                keys, values = layer_cache.state
                save_data[f"layer_{i}_keys"] = keys
                save_data[f"layer_{i}_values"] = values

        if not save_data:
            logger.warning("No cache data to save (empty cache)")
            return None

        # Save using MLX's safetensors format
        mx.save_safetensors(str(cache_path), save_data)

        # Save metadata separately (tokens, hashes, etc.)
        meta_path = cache_path.with_suffix(".json")
        metadata = {
            "model_id": model_id,
            "num_tokens": len(tokens),
            "num_layers": len(cache),
            "block_hashes": block_hashes,
            "tokens": tokens,  # Store tokens for validation
            "created_at": time.time(),
        }
        with open(meta_path, "w") as f:
            json.dump(metadata, f)

        # Get file size for logging
        file_size_mb = cache_path.stat().st_size / (1024 * 1024)
        logger.info(
            f"Saved KV cache to disk: {len(tokens)} tokens, "
            f"{len(cache)} layers, {file_size_mb:.1f}MB -> {cache_path.name}"
        )

        # Run cleanup after saving
        cleanup_disk_cache()

        return cache_path

    except Exception as e:
        # Log more details for debugging
        if cache:
            cache_types = [type(c).__name__ for c in cache[:3]]  # First 3 layers
            logger.warning(f"Failed to save cache to disk: {e} (cache types: {cache_types})")
        else:
            logger.warning(f"Failed to save cache to disk: {e}")
        return None


def load_cache_from_disk(
    model_id: str,
    prompt_hash: str,
    num_layers: int,
) -> Optional[Tuple[List[Any], List[int], List[str]]]:
    """
    Load KV cache from disk.

    Args:
        model_id: Model identifier
        prompt_hash: Hash of the prompt to load
        num_layers: Expected number of layers (must match)

    Returns:
        Tuple of (cache, tokens, block_hashes) or None if not found/invalid
    """
    if not DISK_CACHE_ENABLED:
        return None

    try:
        cache_path = get_disk_cache_path(model_id, prompt_hash)
        meta_path = cache_path.with_suffix(".json")

        if not cache_path.exists() or not meta_path.exists():
            return None

        # Load metadata first
        with open(meta_path, "r") as f:
            metadata = json.load(f)

        # Validate model and layers
        if metadata["model_id"] != model_id:
            logger.debug(f"Disk cache model mismatch: {metadata['model_id']} != {model_id}")
            return None

        if metadata["num_layers"] != num_layers:
            logger.debug(f"Disk cache layer count mismatch: {metadata['num_layers']} != {num_layers}")
            return None

        start_time = time.perf_counter()

        # Load the tensors
        data = mx.load(str(cache_path))

        # Reconstruct cache objects
        cache = []
        for i in range(num_layers):
            keys_key = f"layer_{i}_keys"
            values_key = f"layer_{i}_values"

            if keys_key not in data or values_key not in data:
                logger.warning(f"Missing layer {i} in disk cache")
                return None

            layer_cache = KVCache()
            layer_cache.state = (data[keys_key], data[values_key])
            cache.append(layer_cache)

        # Force evaluation to actually load into memory
        mx.eval([c.keys for c in cache] + [c.values for c in cache])

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        file_size_mb = cache_path.stat().st_size / (1024 * 1024)

        logger.info(
            f"Loaded KV cache from disk: {metadata['num_tokens']} tokens, "
            f"{num_layers} layers, {file_size_mb:.1f}MB in {elapsed_ms:.0f}ms"
        )

        return cache, metadata["tokens"], metadata["block_hashes"]

    except Exception as e:
        logger.warning(f"Failed to load cache from disk: {e}")
        return None


def find_disk_cache_by_prefix(
    model_id: str,
    prompt_hashes: List[str],
) -> Optional[Tuple[str, int]]:
    """
    Find a disk cache that matches a prefix of the given prompt.

    Args:
        model_id: Model identifier
        prompt_hashes: Block hashes for the new prompt

    Returns:
        Tuple of (matching_hash, matching_blocks) or None if no match
    """
    if not DISK_CACHE_ENABLED or not DISK_CACHE_DIR.exists():
        return None

    safe_model_id = model_id.replace("/", "_").replace("\\", "_")

    best_match_hash = None
    best_match_blocks = 0

    try:
        # Look for cache files for this model
        for meta_file in DISK_CACHE_DIR.glob(f"{safe_model_id}_*.json"):
            try:
                with open(meta_file, "r") as f:
                    metadata = json.load(f)

                cached_hashes = metadata.get("block_hashes", [])
                if not cached_hashes:
                    continue

                # Count matching blocks from the start
                match_blocks = 0
                for i, (ph, ch) in enumerate(zip(prompt_hashes, cached_hashes)):
                    if ph == ch:
                        match_blocks = i + 1
                    else:
                        break

                if match_blocks > best_match_blocks:
                    best_match_blocks = match_blocks
                    best_match_hash = cached_hashes[-1]  # The cache is indexed by last hash

            except Exception:
                continue

        if best_match_hash and best_match_blocks > 0:
            return best_match_hash, best_match_blocks

    except Exception as e:
        logger.debug(f"Error searching disk cache: {e}")

    return None


def clear_disk_cache(model_id: Optional[str] = None) -> int:
    """
    Clear disk cache files.

    Args:
        model_id: If provided, only clear cache for this model. Otherwise clear all.

    Returns:
        Number of files deleted
    """
    if not DISK_CACHE_DIR.exists():
        return 0

    deleted = 0
    try:
        if model_id:
            safe_model_id = model_id.replace("/", "_").replace("\\", "_")
            pattern = f"{safe_model_id}_*"
        else:
            pattern = "*"

        for cache_file in DISK_CACHE_DIR.glob(pattern):
            try:
                cache_file.unlink()
                deleted += 1
            except Exception:
                pass

        logger.info(f"Cleared {deleted} disk cache files")

    except Exception as e:
        logger.warning(f"Error clearing disk cache: {e}")

    return deleted


def cleanup_disk_cache() -> Dict[str, int]:
    """
    Clean up old or excess disk cache files.

    Removes:
    - Cache files older than DISK_CACHE_MAX_AGE_DAYS
    - Oldest files when total size exceeds DISK_CACHE_MAX_SIZE_GB

    Returns:
        Dict with cleanup statistics
    """
    if not DISK_CACHE_ENABLED or not DISK_CACHE_DIR.exists():
        return {"deleted_by_age": 0, "deleted_by_size": 0, "total_size_mb": 0}

    deleted_by_age = 0
    deleted_by_size = 0
    current_time = time.time()
    max_age_seconds = DISK_CACHE_MAX_AGE_DAYS * 24 * 60 * 60
    max_size_bytes = DISK_CACHE_MAX_SIZE_GB * 1024 * 1024 * 1024

    try:
        # Get all cache files with their metadata
        cache_files = []
        total_size = 0

        for cache_file in DISK_CACHE_DIR.glob("*.safetensors"):
            try:
                stat = cache_file.stat()
                age = current_time - stat.st_mtime
                size = stat.st_size

                # Delete old files immediately
                if age > max_age_seconds:
                    cache_file.unlink()
                    meta_file = cache_file.with_suffix(".json")
                    if meta_file.exists():
                        meta_file.unlink()
                    deleted_by_age += 1
                    logger.debug(f"Deleted old cache: {cache_file.name} (age: {age/86400:.1f} days)")
                else:
                    cache_files.append({
                        "path": cache_file,
                        "meta_path": cache_file.with_suffix(".json"),
                        "mtime": stat.st_mtime,
                        "size": size,
                    })
                    total_size += size

            except Exception:
                continue

        # If still over size limit, delete oldest files
        if total_size > max_size_bytes:
            # Sort by modification time (oldest first)
            cache_files.sort(key=lambda x: x["mtime"])

            for cf in cache_files:
                if total_size <= max_size_bytes:
                    break

                try:
                    cf["path"].unlink()
                    if cf["meta_path"].exists():
                        cf["meta_path"].unlink()
                    total_size -= cf["size"]
                    deleted_by_size += 1
                    logger.debug(f"Deleted cache for size limit: {cf['path'].name}")
                except Exception:
                    continue

        total_size_mb = total_size / (1024 * 1024)

        if deleted_by_age > 0 or deleted_by_size > 0:
            logger.info(
                f"Disk cache cleanup: deleted {deleted_by_age} old + {deleted_by_size} for size, "
                f"remaining: {total_size_mb:.1f}MB"
            )

        return {
            "deleted_by_age": deleted_by_age,
            "deleted_by_size": deleted_by_size,
            "total_size_mb": round(total_size_mb, 1),
        }

    except Exception as e:
        logger.warning(f"Error during disk cache cleanup: {e}")
        return {"deleted_by_age": 0, "deleted_by_size": 0, "total_size_mb": 0, "error": str(e)}


@dataclass
class CacheSlot:
    """
    A single cache slot storing KV cache for a specific token sequence.

    Attributes:
        tokens: The cached token sequence
        cache: The KV cache state (list of layer caches)
        block_hashes: Cumulative hashes for each block
        last_access: Timestamp of last access (for LRU)
        model_key: Model identifier
        hit_count: Number of times this cache was reused
        created_at: Timestamp of slot creation
        total_reuse_tokens: Total tokens reused from this slot
    """
    tokens: List[int] = field(default_factory=list)
    cache: List[Any] = field(default_factory=list)
    block_hashes: List[str] = field(default_factory=list)
    last_access: float = 0.0
    model_key: str = ""
    hit_count: int = 0
    created_at: float = field(default_factory=time.time)
    total_reuse_tokens: int = 0

    def update_access(self, reused_tokens: int = 0):
        """Update last access time and hit count."""
        self.last_access = time.time()
        self.hit_count += 1
        self.total_reuse_tokens += reused_tokens


class SmartPromptCache:
    """
    Smart prompt cache with hash-based lookup and multiple slots.

    This cache maintains multiple slots, each storing a different prompt's KV cache.
    When a new request comes in, it:
    1. Computes block hashes for the new prompt
    2. Looks for a slot with the longest matching prefix (by hash)
    3. If found, trims or extends that slot's cache
    4. If not found, creates a new slot (evicting LRU if needed)

    Configuration (via environment variables):
        MLX_CACHE_BLOCK_SIZE: Token block size for hashing (default 256)
        MLX_CACHE_MAX_SLOTS: Maximum number of cache slots (default 4)
        MLX_CACHE_MIN_REUSE: Minimum tokens to consider a cache hit (default 512)
        MLX_CACHE_MAX_TOKENS: Maximum tokens per slot (default 65536)
    """

    def __init__(
        self,
        max_slots: int = DEFAULT_MAX_SLOTS,
        block_size: int = DEFAULT_BLOCK_SIZE,
        min_reuse_tokens: int = DEFAULT_MIN_REUSE,
    ):
        """
        Initialize the smart cache.

        Args:
            max_slots: Maximum cache slots (more = more memory, better hit rate)
            block_size: Tokens per block for hashing
            min_reuse_tokens: Minimum tokens to reuse (below this, just recompute)
        """
        self.max_slots = max_slots
        self.block_size = block_size
        self.min_reuse_tokens = min_reuse_tokens

        # OrderedDict maintains insertion order, useful for LRU
        self.slots: OrderedDict[str, CacheSlot] = OrderedDict()

        # Statistics
        self.total_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.tokens_saved = 0
        self.total_evictions = 0
        self.disk_cache_hits = 0
        self.disk_cache_saves = 0

        # Pending disk save (set when a new large slot is created)
        self._pending_disk_save: Optional[Dict[str, Any]] = None

    def _find_best_slot(
        self,
        prompt_hashes: List[str],
        model_key: str,
    ) -> Tuple[Optional[str], int]:
        """
        Find the slot with the longest matching prefix.

        Args:
            prompt_hashes: Block hashes for the new prompt
            model_key: Model identifier

        Returns:
            Tuple of (slot_key, matching_blocks) or (None, 0) if no match
        """
        best_slot_key = None
        best_match_blocks = 0

        for slot_key, slot in self.slots.items():
            # Skip if different model
            if slot.model_key != model_key:
                continue

            # Count matching blocks from the start
            match_blocks = 0
            for i, (ph, sh) in enumerate(zip(prompt_hashes, slot.block_hashes)):
                if ph == sh:
                    match_blocks = i + 1
                else:
                    break

            if match_blocks > best_match_blocks:
                best_match_blocks = match_blocks
                best_slot_key = slot_key

        return best_slot_key, best_match_blocks

    def _release_cache_memory(self, cache: List[Any]):
        """
        Explicitly release memory held by KV cache arrays.

        MLX arrays hold unified memory that should be explicitly released
        to avoid memory pressure.
        """
        if not cache:
            return

        try:
            for cache_layer in cache:
                # MLX KV cache structure has keys and values attributes
                if hasattr(cache_layer, 'keys'):
                    del cache_layer.keys
                if hasattr(cache_layer, 'values'):
                    del cache_layer.values
            cache.clear()
        except Exception as e:
            logger.debug(f"Error releasing cache memory: {e}")

    def _evict_lru_slot(self):
        """Evict the least recently used slot and release its memory."""
        if not self.slots:
            return

        # Find slot with oldest last_access
        lru_key = min(self.slots.keys(), key=lambda k: self.slots[k].last_access)
        evicted = self.slots.pop(lru_key)

        # Explicitly release cache memory
        self._release_cache_memory(evicted.cache)

        self.total_evictions += 1
        logger.debug(
            f"Evicted cache slot (tokens: {len(evicted.tokens)}, "
            f"hits: {evicted.hit_count}, age: {time.time() - evicted.last_access:.1f}s, "
            f"total_reuse: {evicted.total_reuse_tokens})"
        )

    def _create_slot(
        self,
        model: MLXModel,
        prompt: List[int],
        block_hashes: List[str],
    ) -> CacheSlot:
        """Create a new cache slot for a prompt."""
        cache = make_prompt_cache(model.model)
        if model.draft_model is not None:
            cache += make_prompt_cache(model.draft_model)

        slot = CacheSlot(
            tokens=list(prompt),
            cache=cache,
            block_hashes=block_hashes,
            last_access=time.time(),
            model_key=model.model_id,
            hit_count=0,
            created_at=time.time(),
            total_reuse_tokens=0,
        )

        # Generate a unique key for this slot using first and last block hashes
        first_hash = block_hashes[0] if block_hashes else 'empty'
        last_hash = block_hashes[-1] if len(block_hashes) > 1 else ''
        slot_key = f"{model.model_id}:{first_hash}:{last_hash}"

        # Evict if at capacity
        if len(self.slots) >= self.max_slots:
            self._evict_lru_slot()

        self.slots[slot_key] = slot
        return slot

    def _trim_slot_if_needed(self, slot: CacheSlot):
        """
        Trim slot tokens if they exceed maximum to prevent unbounded memory growth.
        """
        if len(slot.tokens) > MAX_CACHED_TOKENS:
            trim_amount = len(slot.tokens) - MAX_CACHED_TOKENS
            if can_trim_prompt_cache(slot.cache):
                try:
                    trim_prompt_cache(slot.cache, trim_amount)
                    slot.tokens = slot.tokens[trim_amount:]
                    slot.block_hashes = compute_block_hashes(slot.tokens, self.block_size)
                    logger.debug(f"Auto-trimmed {trim_amount} tokens from slot (max: {MAX_CACHED_TOKENS})")
                except Exception as e:
                    logger.warning(f"Failed to auto-trim slot: {e}")

    def get_prompt_cache(
        self,
        model: MLXModel,
        prompt: List[int],
    ) -> Tuple[List[int], int, List[Any]]:
        """
        Get the portion of the prompt that needs processing, reusing cache if possible.

        This is the main entry point, compatible with the original PromptCache interface.

        Args:
            model: The MLX model
            prompt: Full tokenized prompt

        Returns:
            Tuple of:
            - tokens_to_process: Tokens that need to be processed
            - cached_tokens: Number of tokens reused from cache
            - cache: The KV cache to use
        """
        start_time = time.perf_counter()
        self.total_requests += 1

        # Handle empty or very short prompts
        if len(prompt) < self.min_reuse_tokens:
            logger.debug(f"Prompt too short ({len(prompt)} tokens), creating fresh cache")
            cache = make_prompt_cache(model.model)
            if model.draft_model is not None:
                cache += make_prompt_cache(model.draft_model)
            return prompt, 0, cache

        # Compute block hashes for the new prompt
        prompt_hashes = compute_block_hashes(prompt, self.block_size)

        # Find best matching slot
        best_slot_key, match_blocks = self._find_best_slot(prompt_hashes, model.model_id)

        # Calculate matched tokens
        matched_tokens = match_blocks * self.block_size
        # Don't exceed actual prompt length (leave at least 1 token)
        matched_tokens = min(matched_tokens, len(prompt) - 1)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # If good match found, reuse that slot
        if best_slot_key and matched_tokens >= self.min_reuse_tokens:
            slot = self.slots[best_slot_key]
            slot.update_access(matched_tokens)

            # Move to end (most recently used)
            self.slots.move_to_end(best_slot_key)

            slot_tokens = len(slot.tokens)

            # Case 1: Cache is prefix of new prompt (ideal - just append)
            if slot_tokens <= matched_tokens:
                # Trim cache if it has extra tokens beyond match
                if slot_tokens > matched_tokens:
                    if can_trim_prompt_cache(slot.cache):
                        try:
                            trim_amount = slot_tokens - matched_tokens
                            trim_prompt_cache(slot.cache, trim_amount)
                            slot.tokens = slot.tokens[:matched_tokens]
                            slot.block_hashes = slot.block_hashes[:match_blocks]
                        except Exception as e:
                            logger.warning(f"Cache trim failed: {e}. Creating new slot.")
                            self.cache_misses += 1
                            slot = self._create_slot(model, prompt, prompt_hashes)
                            return prompt, 0, slot.cache

                tokens_to_process = prompt[len(slot.tokens):]
                cached_tokens = len(slot.tokens)

                # Update slot with new tokens
                slot.tokens.extend(tokens_to_process)
                slot.block_hashes = prompt_hashes

                # Auto-trim if slot grows too large
                self._trim_slot_if_needed(slot)

                self.cache_hits += 1
                self.tokens_saved += cached_tokens

                reuse_pct = 100 * cached_tokens / len(prompt)
                logger.info(
                    f"Cache HIT: reusing {cached_tokens}/{len(prompt)} tokens "
                    f"({reuse_pct:.1f}% cached) in {elapsed_ms:.1f}ms "
                    f"[slot hits: {slot.hit_count}]"
                )

                return tokens_to_process, cached_tokens, slot.cache

            # Case 2: Cache has more tokens than match - need to trim
            else:
                if can_trim_prompt_cache(slot.cache):
                    try:
                        trim_amount = slot_tokens - matched_tokens
                        logger.debug(f"Trimming {trim_amount} tokens from cache")
                        trim_prompt_cache(slot.cache, trim_amount)
                        slot.tokens = slot.tokens[:matched_tokens]
                        slot.block_hashes = slot.block_hashes[:match_blocks]

                        tokens_to_process = prompt[matched_tokens:]

                        # Update slot
                        slot.tokens.extend(tokens_to_process)
                        slot.block_hashes = prompt_hashes

                        # Auto-trim if slot grows too large
                        self._trim_slot_if_needed(slot)

                        self.cache_hits += 1
                        self.tokens_saved += matched_tokens

                        reuse_pct = 100 * matched_tokens / len(prompt)
                        logger.info(
                            f"Cache HIT (trimmed): reusing {matched_tokens}/{len(prompt)} tokens "
                            f"({reuse_pct:.1f}% cached) in {elapsed_ms:.1f}ms "
                            f"[slot hits: {slot.hit_count}]"
                        )

                        return tokens_to_process, matched_tokens, slot.cache
                    except Exception as e:
                        logger.warning(f"Cache trim failed: {e}. Creating new slot.")
                else:
                    logger.debug("Cache trim not supported, creating new slot")

        # No good match in memory - check disk cache
        num_layers = len(model.model.layers)

        # Try to find a disk cache that matches a prefix
        disk_match = find_disk_cache_by_prefix(model.model_id, prompt_hashes)
        if disk_match:
            disk_hash, disk_match_blocks = disk_match
            disk_matched_tokens = disk_match_blocks * self.block_size
            disk_matched_tokens = min(disk_matched_tokens, len(prompt) - 1)

            if disk_matched_tokens >= self.min_reuse_tokens:
                # Try to load from disk
                disk_result = load_cache_from_disk(model.model_id, disk_hash, num_layers)
                if disk_result:
                    loaded_cache, loaded_tokens, loaded_hashes = disk_result

                    # IMPORTANT: Only use the matching portion of the cache!
                    # The disk cache might have more tokens than actually match the new prompt
                    actual_cached_tokens = min(disk_matched_tokens, len(loaded_tokens))

                    # If disk cache has more tokens than match, we need to trim
                    if len(loaded_tokens) > actual_cached_tokens:
                        if can_trim_prompt_cache(loaded_cache):
                            trim_amount = len(loaded_tokens) - actual_cached_tokens
                            try:
                                trim_prompt_cache(loaded_cache, trim_amount)
                                loaded_tokens = loaded_tokens[:actual_cached_tokens]
                                loaded_hashes = loaded_hashes[:disk_match_blocks]
                                logger.debug(f"Trimmed disk cache by {trim_amount} tokens to match prefix")
                            except Exception as e:
                                logger.warning(f"Failed to trim disk cache: {e}")
                                # Fall through to create new slot instead
                                loaded_cache = None
                        else:
                            logger.debug("Disk cache trim not supported, skipping")
                            loaded_cache = None

                    if loaded_cache:
                        # Create a slot from the loaded cache
                        slot = CacheSlot(
                            tokens=list(loaded_tokens),
                            cache=loaded_cache,
                            block_hashes=loaded_hashes,
                            last_access=time.time(),
                            model_key=model.model_id,
                            hit_count=1,
                            created_at=time.time(),
                            total_reuse_tokens=actual_cached_tokens,
                        )

                        # Generate slot key
                        first_hash = loaded_hashes[0] if loaded_hashes else 'empty'
                        last_hash = loaded_hashes[-1] if len(loaded_hashes) > 1 else ''
                        slot_key = f"{model.model_id}:{first_hash}:{last_hash}"

                        # Evict if at capacity
                        if len(self.slots) >= self.max_slots:
                            self._evict_lru_slot()

                        self.slots[slot_key] = slot

                        # Now use this slot like a memory hit
                        tokens_to_process = prompt[actual_cached_tokens:]

                        # Update slot with new tokens
                        slot.tokens = list(prompt)
                        slot.block_hashes = prompt_hashes

                        self.cache_hits += 1
                        self.disk_cache_hits += 1
                        self.tokens_saved += actual_cached_tokens

                        disk_elapsed_ms = (time.perf_counter() - start_time) * 1000
                        reuse_pct = 100 * actual_cached_tokens / len(prompt)
                        logger.info(
                            f"DISK Cache HIT: reusing {actual_cached_tokens}/{len(prompt)} tokens "
                            f"({reuse_pct:.1f}% cached) in {disk_elapsed_ms:.0f}ms"
                        )

                        return tokens_to_process, actual_cached_tokens, slot.cache

        # No disk cache either - create new slot
        self.cache_misses += 1
        logger.info(
            f"Cache MISS: creating new slot for {len(prompt)} tokens "
            f"(best match: {matched_tokens}, threshold: {self.min_reuse_tokens}) "
            f"slots: {len(self.slots)}/{self.max_slots} in {elapsed_ms:.1f}ms"
        )

        slot = self._create_slot(model, prompt, prompt_hashes)

        # Schedule disk save for large prompts (will happen after prefill completes)
        if len(prompt) >= DISK_CACHE_MIN_TOKENS:
            self._pending_disk_save = {
                "cache": slot.cache,
                "tokens": list(prompt),
                "model_id": model.model_id,
                "block_hashes": prompt_hashes,
            }

        return prompt, 0, slot.cache

    def extend_cache(self, completion_tokens: List[int]):
        """
        Extend the most recently used cache with completion tokens.

        Optimized to only recompute hashes for new blocks.

        Called after generation to include the generated tokens in the cache.
        """
        if not self.slots:
            return

        # Get most recently used slot
        slot_key = next(reversed(self.slots))
        slot = self.slots[slot_key]

        old_token_count = len(slot.tokens)
        slot.tokens.extend(completion_tokens)
        new_token_count = len(slot.tokens)

        # Only recompute hashes for affected blocks (incremental)
        old_block_count = (old_token_count + self.block_size - 1) // self.block_size
        new_block_count = (new_token_count + self.block_size - 1) // self.block_size

        if new_block_count > old_block_count:
            # New blocks created, recompute from the last complete block
            start_idx = max(0, (old_block_count - 1)) * self.block_size
            new_tokens = slot.tokens[start_idx:]

            # Get prefix hash from before the affected blocks
            prefix_hash = slot.block_hashes[old_block_count - 2] if old_block_count > 1 else ""

            # Compute new block hashes
            new_hashes = []
            for i in range(0, len(new_tokens), self.block_size):
                block = new_tokens[i:i + self.block_size]
                block_hash = compute_token_hash(block, prefix_hash)
                new_hashes.append(block_hash)
                prefix_hash = block_hash

            # Update only the changed portion
            keep_blocks = max(0, old_block_count - 1)
            slot.block_hashes = slot.block_hashes[:keep_blocks] + new_hashes

        # Auto-trim if slot grows too large
        self._trim_slot_if_needed(slot)

    def extend_completion_cache(self, completion_tokens: List[int]):
        """Alias for extend_cache for API compatibility."""
        self.extend_cache(completion_tokens)

    def flush_pending_disk_save(self):
        """
        Save pending KV cache to disk.

        This should be called after the prefill is complete and the KV cache
        has been populated with the actual computed values.
        """
        if self._pending_disk_save is None:
            return

        pending = self._pending_disk_save
        self._pending_disk_save = None

        try:
            result = save_cache_to_disk(
                cache=pending["cache"],
                tokens=pending["tokens"],
                model_id=pending["model_id"],
                block_hashes=pending["block_hashes"],
            )
            if result:
                self.disk_cache_saves += 1
        except Exception as e:
            logger.warning(f"Failed to save pending disk cache: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get detailed cache statistics."""
        active_slots_info = []
        for key, slot in self.slots.items():
            active_slots_info.append({
                "key": key[:40] + "..." if len(key) > 40 else key,
                "tokens": len(slot.tokens),
                "blocks": len(slot.block_hashes),
                "hits": slot.hit_count,
                "age_seconds": round(time.time() - slot.created_at, 1),
                "total_reuse": slot.total_reuse_tokens,
            })

        return {
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": round(self.cache_hits / max(1, self.total_requests), 3),
            "tokens_saved": self.tokens_saved,
            "avg_tokens_saved_per_hit": round(self.tokens_saved / max(1, self.cache_hits), 1),
            "total_evictions": self.total_evictions,
            "active_slots": len(self.slots),
            "max_slots": self.max_slots,
            "slots_detail": active_slots_info,
            "disk_cache": {
                "enabled": DISK_CACHE_ENABLED,
                "hits": self.disk_cache_hits,
                "saves": self.disk_cache_saves,
                "min_tokens": DISK_CACHE_MIN_TOKENS,
                "directory": str(DISK_CACHE_DIR),
            },
            "config": {
                "block_size": self.block_size,
                "min_reuse_tokens": self.min_reuse_tokens,
                "max_cached_tokens": MAX_CACHED_TOKENS,
            }
        }

    def get_health_report(self) -> str:
        """Generate human-readable health report."""
        stats = self.get_stats()

        report = [
            "=== SmartPromptCache Health Report ===",
            f"Hit Rate: {stats['hit_rate']*100:.1f}% ({stats['cache_hits']}/{stats['total_requests']})",
            f"Tokens Saved: {stats['tokens_saved']:,} ({stats['avg_tokens_saved_per_hit']:.0f} avg/hit)",
            f"Active Slots: {stats['active_slots']}/{stats['max_slots']}",
            f"Total Evictions: {stats['total_evictions']}",
            "",
            "Slot Details:",
        ]

        for slot in stats['slots_detail']:
            age_min = slot['age_seconds'] / 60
            report.append(
                f"  - {slot['tokens']:,} tokens, {slot['hits']} hits, "
                f"age: {age_min:.1f}min, reused: {slot['total_reuse']:,}"
            )

        return "\n".join(report)

    def log_health_report(self):
        """Log health report at INFO level."""
        logger.info(f"\n{self.get_health_report()}")

    def clear(self):
        """Clear all cache slots and release memory."""
        for slot in self.slots.values():
            self._release_cache_memory(slot.cache)
        self.slots.clear()
        logger.info("Cleared all cache slots")


# Backwards compatible alias
class PromptCache(SmartPromptCache):
    """
    Backwards compatible wrapper for SmartPromptCache.

    Provides the same interface as the original PromptCache class.
    """

    def __init__(self):
        super().__init__(max_slots=4, block_size=256, min_reuse_tokens=512)
        # For backwards compatibility
        self.tokens: List[int] = []
        self.cache: List[Any] = []
        self.model_key: str = ""

    def extend_completion_cache(self, completion_tokens: List[int]):
        """Backwards compatible method."""
        self.extend_cache(completion_tokens)
        self.tokens.extend(completion_tokens)

    def reset_prompt_cache(self, model: MLXModel, prompt: List[int]):
        """Backwards compatible method - creates new slot."""
        self.clear()
        prompt_hashes = compute_block_hashes(prompt, self.block_size)
        slot = self._create_slot(model, prompt, prompt_hashes)
        self.tokens = slot.tokens
        self.cache = slot.cache
        self.model_key = model.model_id

    def get_prompt_cache_legacy(
        self,
        model: MLXModel,
        prompt: List[int],
    ) -> Tuple[List[int], int]:
        """
        Legacy interface returning (tokens_to_process, cached_count).

        The cache object is stored in self.cache for backwards compatibility.
        """
        tokens_to_process, cached_count, cache = super().get_prompt_cache(model, prompt)
        self.cache = cache
        self.tokens = list(prompt)
        self.model_key = model.model_id
        return tokens_to_process, cached_count
