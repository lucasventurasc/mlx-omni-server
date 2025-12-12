"""
Test suite for SmartPromptCache

Run with: python -m pytest tests/test_smart_prompt_cache.py -v
"""

import time
import pytest
from unittest.mock import MagicMock, patch

# Test the cache logic without needing actual MLX models
import sys
sys.path.insert(0, 'src')

from mlx_omni_server.chat.mlx.smart_prompt_cache import (
    compute_token_hash,
    compute_block_hashes,
    SmartPromptCache,
    CacheSlot,
)


class TestHashFunctions:
    """Test hash computation functions."""

    def test_compute_token_hash_deterministic(self):
        """Same tokens should produce same hash."""
        tokens = [1, 2, 3, 4, 5]
        hash1 = compute_token_hash(tokens)
        hash2 = compute_token_hash(tokens)
        assert hash1 == hash2

    def test_compute_token_hash_different_tokens(self):
        """Different tokens should produce different hashes."""
        tokens1 = [1, 2, 3, 4, 5]
        tokens2 = [1, 2, 3, 4, 6]
        hash1 = compute_token_hash(tokens1)
        hash2 = compute_token_hash(tokens2)
        assert hash1 != hash2

    def test_compute_token_hash_with_prefix(self):
        """Hash with prefix should be different from hash without."""
        tokens = [1, 2, 3]
        hash_no_prefix = compute_token_hash(tokens)
        hash_with_prefix = compute_token_hash(tokens, "some_prefix")
        assert hash_no_prefix != hash_with_prefix

    def test_compute_token_hash_uses_bytes(self):
        """Hash function should handle large token lists efficiently."""
        # This tests that the bytes-based implementation works
        large_tokens = list(range(10000))
        hash_result = compute_token_hash(large_tokens)
        assert len(hash_result) == 16  # 16 hex chars

    def test_compute_block_hashes_chaining(self):
        """Block hashes should chain correctly."""
        tokens = list(range(1000))  # 1000 tokens
        block_size = 256
        hashes = compute_block_hashes(tokens, block_size)

        # Should have ceil(1000/256) = 4 hashes
        assert len(hashes) == 4

        # All hashes should be unique (since blocks are different)
        assert len(set(hashes)) == len(hashes)

    def test_compute_block_hashes_prefix_match(self):
        """Two prompts with same prefix should have matching initial hashes."""
        prefix = list(range(512))  # 512 tokens = 2 blocks
        prompt1 = prefix + [9999, 9998, 9997]  # Different suffix
        prompt2 = prefix + [8888, 8887, 8886]  # Different suffix

        hashes1 = compute_block_hashes(prompt1, 256)
        hashes2 = compute_block_hashes(prompt2, 256)

        # First 2 block hashes should match (same prefix)
        assert hashes1[0] == hashes2[0]
        assert hashes1[1] == hashes2[1]
        # Third block should differ (different suffix)
        assert hashes1[2] != hashes2[2]


class TestSmartPromptCache:
    """Test SmartPromptCache class."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock MLX model."""
        model = MagicMock()
        model.model_id = "test-model"
        model.model = MagicMock()
        model.draft_model = None
        return model

    @pytest.fixture
    def cache(self):
        """Create a SmartPromptCache with test settings."""
        return SmartPromptCache(
            max_slots=4,
            block_size=256,
            min_reuse_tokens=100,  # Lower for testing
        )

    @patch('mlx_omni_server.chat.mlx.smart_prompt_cache.make_prompt_cache')
    def test_first_request_creates_slot(self, mock_make_cache, cache, mock_model):
        """First request should create a new cache slot."""
        mock_make_cache.return_value = [MagicMock()]  # Mock cache object

        prompt = list(range(1000))  # 1000 tokens
        tokens_to_process, cached_tokens, cache_obj = cache.get_prompt_cache(
            mock_model, prompt
        )

        # First request - nothing cached
        assert cached_tokens == 0
        assert len(tokens_to_process) == len(prompt)
        assert len(cache.slots) == 1
        assert cache.cache_misses == 1

    @patch('mlx_omni_server.chat.mlx.smart_prompt_cache.make_prompt_cache')
    @patch('mlx_omni_server.chat.mlx.smart_prompt_cache.can_trim_prompt_cache')
    @patch('mlx_omni_server.chat.mlx.smart_prompt_cache.trim_prompt_cache')
    def test_cache_hit_same_prefix(
        self, mock_trim, mock_can_trim, mock_make_cache, cache, mock_model
    ):
        """Second request with same prefix should get cache hit."""
        mock_make_cache.return_value = [MagicMock()]
        mock_can_trim.return_value = True

        # First request
        prompt1 = list(range(1000))
        cache.get_prompt_cache(mock_model, prompt1)

        # Second request with same prefix + some new tokens
        prompt2 = list(range(1000)) + [9999, 9998]
        tokens_to_process, cached_tokens, cache_obj = cache.get_prompt_cache(
            mock_model, prompt2
        )

        # Should have cache hit
        assert cached_tokens > 0
        assert len(tokens_to_process) < len(prompt2)
        assert cache.cache_hits >= 1

    @patch('mlx_omni_server.chat.mlx.smart_prompt_cache.make_prompt_cache')
    def test_different_models_different_slots(self, mock_make_cache, cache):
        """Different models should use different cache slots."""
        mock_make_cache.return_value = [MagicMock()]

        model1 = MagicMock()
        model1.model_id = "model-1"
        model1.model = MagicMock()
        model1.draft_model = None

        model2 = MagicMock()
        model2.model_id = "model-2"
        model2.model = MagicMock()
        model2.draft_model = None

        prompt = list(range(500))

        cache.get_prompt_cache(model1, prompt)
        cache.get_prompt_cache(model2, prompt)

        # Should have 2 slots (one per model)
        assert len(cache.slots) == 2

    @patch('mlx_omni_server.chat.mlx.smart_prompt_cache.make_prompt_cache')
    def test_lru_eviction(self, mock_make_cache, mock_model):
        """When max slots reached, LRU slot should be evicted."""
        mock_make_cache.return_value = [MagicMock()]

        cache = SmartPromptCache(max_slots=2, block_size=256, min_reuse_tokens=100)

        # Create prompts with different first blocks (so they don't match)
        prompt1 = [1] * 256 + list(range(256, 500))  # First block: all 1s
        prompt2 = [2] * 256 + list(range(256, 500))  # First block: all 2s
        prompt3 = [3] * 256 + list(range(256, 500))  # First block: all 3s

        cache.get_prompt_cache(mock_model, prompt1)
        cache.get_prompt_cache(mock_model, prompt2)

        assert len(cache.slots) == 2

        # Third prompt should evict the oldest (prompt1's slot)
        cache.get_prompt_cache(mock_model, prompt3)

        assert len(cache.slots) == 2  # Still max 2
        assert cache.total_evictions == 1

    def test_statistics(self, cache):
        """Cache should track statistics."""
        stats = cache.get_stats()

        assert "total_requests" in stats
        assert "cache_hits" in stats
        assert "cache_misses" in stats
        assert "hit_rate" in stats
        assert "tokens_saved" in stats
        assert "active_slots" in stats
        assert "max_slots" in stats
        assert "total_evictions" in stats
        assert "config" in stats

    @patch('mlx_omni_server.chat.mlx.smart_prompt_cache.make_prompt_cache')
    def test_extend_completion_cache(self, mock_make_cache, cache, mock_model):
        """extend_completion_cache should add tokens to most recent slot."""
        mock_make_cache.return_value = [MagicMock()]

        prompt = list(range(500))
        cache.get_prompt_cache(mock_model, prompt)

        # Get the slot
        slot_key = next(iter(cache.slots))
        initial_tokens = len(cache.slots[slot_key].tokens)

        # Extend with completion tokens
        completion_tokens = [9999, 9998, 9997]
        cache.extend_completion_cache(completion_tokens)

        # Slot should have more tokens now
        assert len(cache.slots[slot_key].tokens) == initial_tokens + len(completion_tokens)

    def test_clear(self, cache):
        """clear() should remove all slots."""
        # Add some fake slots
        cache.slots["slot1"] = CacheSlot()
        cache.slots["slot2"] = CacheSlot()

        assert len(cache.slots) == 2

        cache.clear()

        assert len(cache.slots) == 0

    def test_health_report(self, cache):
        """Health report should be a readable string."""
        report = cache.get_health_report()
        assert isinstance(report, str)
        assert "SmartPromptCache Health Report" in report
        assert "Hit Rate" in report

    @patch('mlx_omni_server.chat.mlx.smart_prompt_cache.make_prompt_cache')
    def test_short_prompt_handling(self, mock_make_cache, mock_model):
        """Short prompts should not use cache."""
        mock_make_cache.return_value = [MagicMock()]

        cache = SmartPromptCache(max_slots=4, block_size=256, min_reuse_tokens=512)

        # Very short prompt (below min_reuse_tokens)
        short_prompt = list(range(100))
        tokens_to_process, cached_tokens, _ = cache.get_prompt_cache(mock_model, short_prompt)

        assert cached_tokens == 0
        assert len(tokens_to_process) == len(short_prompt)
        # Should not create a slot for short prompts
        assert len(cache.slots) == 0


class TestClaudeCodeScenario:
    """Test scenarios specific to Claude Code usage patterns."""

    @patch('mlx_omni_server.chat.mlx.smart_prompt_cache.make_prompt_cache')
    @patch('mlx_omni_server.chat.mlx.smart_prompt_cache.can_trim_prompt_cache')
    @patch('mlx_omni_server.chat.mlx.smart_prompt_cache.trim_prompt_cache')
    def test_system_prompt_reuse(
        self, mock_trim, mock_can_trim, mock_make_cache
    ):
        """
        Claude Code scenario: System prompt (~15k tokens) stays the same,
        only conversation history changes.
        """
        mock_make_cache.return_value = [MagicMock()]
        mock_can_trim.return_value = True

        cache = SmartPromptCache(
            max_slots=4,
            block_size=256,
            min_reuse_tokens=512,
        )

        mock_model = MagicMock()
        mock_model.model_id = "qwen3-coder"
        mock_model.model = MagicMock()
        mock_model.draft_model = None

        # Simulate system prompt + tools (15000 tokens)
        system_tokens = list(range(15000))

        # First request: system + "hello"
        prompt1 = system_tokens + [99001, 99002]  # User says "hello"
        _, cached1, _ = cache.get_prompt_cache(mock_model, prompt1)

        # Second request: system + "hello" + response + "how are you"
        prompt2 = system_tokens + [99001, 99002, 99003, 99004, 99005]
        _, cached2, _ = cache.get_prompt_cache(mock_model, prompt2)

        # Should have significant cache hit
        # At least the system prompt blocks should be reused
        assert cached2 >= 14000, f"Expected >14000 cached tokens, got {cached2}"

        stats = cache.get_stats()
        assert stats["cache_hits"] >= 1
        assert stats["tokens_saved"] > 0

    @patch('mlx_omni_server.chat.mlx.smart_prompt_cache.make_prompt_cache')
    def test_different_agent_requests(self, mock_make_cache):
        """
        Claude Code scenario: Different agents (haiku vs sonnet) may have
        slightly different prompts but share most of the system prompt.
        """
        mock_make_cache.return_value = [MagicMock()]

        cache = SmartPromptCache(max_slots=4, block_size=256, min_reuse_tokens=512)

        mock_model = MagicMock()
        mock_model.model_id = "qwen3-coder"
        mock_model.model = MagicMock()
        mock_model.draft_model = None

        # Shared system prompt base
        shared_base = list(range(10000))

        # Agent 1: haiku prompt variant
        haiku_suffix = [90001, 90002, 90003]
        prompt_haiku = shared_base + haiku_suffix

        # Agent 2: sonnet prompt variant
        sonnet_suffix = [91001, 91002, 91003]
        prompt_sonnet = shared_base + sonnet_suffix

        cache.get_prompt_cache(mock_model, prompt_haiku)
        cache.get_prompt_cache(mock_model, prompt_sonnet)

        # Both should create slots since suffixes differ
        # But they can still benefit from hash-based lookup
        assert len(cache.slots) <= 2


class TestMemoryManagement:
    """Test memory-related functionality."""

    def test_release_cache_memory(self):
        """Test that cache memory is properly released."""
        cache = SmartPromptCache(max_slots=2, block_size=256, min_reuse_tokens=100)

        # Create mock cache with keys and values
        mock_cache_layer = MagicMock()
        mock_cache_layer.keys = MagicMock()
        mock_cache_layer.values = MagicMock()

        mock_cache = [mock_cache_layer]

        # Should not raise
        cache._release_cache_memory(mock_cache)

        # Cache should be cleared
        assert len(mock_cache) == 0

    def test_release_cache_memory_handles_errors(self):
        """Test that release handles errors gracefully."""
        cache = SmartPromptCache(max_slots=2, block_size=256, min_reuse_tokens=100)

        # Create mock that raises on access
        mock_cache_layer = MagicMock()
        mock_cache_layer.keys = property(lambda self: (_ for _ in ()).throw(RuntimeError("test")))

        # Should not raise even with error
        cache._release_cache_memory([mock_cache_layer])


class TestEnvironmentConfig:
    """Test environment variable configuration."""

    def test_default_values(self):
        """Test default configuration values."""
        cache = SmartPromptCache()
        # These should use env vars or defaults
        assert cache.max_slots >= 1
        assert cache.block_size >= 64
        assert cache.min_reuse_tokens >= 1


class TestIncrementalHashing:
    """Test incremental hash computation in extend_cache."""

    @patch('mlx_omni_server.chat.mlx.smart_prompt_cache.make_prompt_cache')
    def test_extend_cache_incremental(self, mock_make_cache):
        """Test that extend_cache only recomputes necessary hashes."""
        mock_make_cache.return_value = [MagicMock()]

        cache = SmartPromptCache(max_slots=4, block_size=256, min_reuse_tokens=100)

        mock_model = MagicMock()
        mock_model.model_id = "test"
        mock_model.model = MagicMock()
        mock_model.draft_model = None

        # Create initial prompt
        prompt = list(range(1000))
        cache.get_prompt_cache(mock_model, prompt)

        slot_key = next(iter(cache.slots))
        initial_hashes = len(cache.slots[slot_key].block_hashes)

        # Extend with more tokens (enough to create new blocks)
        completion_tokens = list(range(1000, 1500))
        cache.extend_completion_cache(completion_tokens)

        new_hashes = len(cache.slots[slot_key].block_hashes)
        assert new_hashes > initial_hashes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
