"""
Shared GPU lock for MLX operations.

Metal GPU operations crash when run concurrently, so all MLX generation
operations must be serialized using this lock.
"""
import threading

# Global lock shared between all routers (OpenAI, Anthropic, etc.)
# This prevents concurrent GPU access which causes Metal command buffer crashes
mlx_generation_lock = threading.Lock()
