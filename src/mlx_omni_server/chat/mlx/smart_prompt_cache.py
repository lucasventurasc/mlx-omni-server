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

This is designed for Claude Code workloads where:
- System prompt + tools are ~15,000 tokens and rarely change
- Conversation history grows but shares common prefixes
- Different request types (haiku vs sonnet) may have different prompts
"""

import hashlib
import os
import struct
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from mlx_lm.models.cache import (
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

        # No good match - create new slot
        self.cache_misses += 1
        logger.info(
            f"Cache MISS: creating new slot for {len(prompt)} tokens "
            f"(best match: {matched_tokens}, threshold: {self.min_reuse_tokens}) "
            f"slots: {len(self.slots)}/{self.max_slots} in {elapsed_ms:.1f}ms"
        )

        slot = self._create_slot(model, prompt, prompt_hashes)
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
