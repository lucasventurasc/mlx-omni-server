"""Anthropic Messages API Schema Definitions

This module defines Pydantic models for the Anthropic Messages API,
following the official API specification.
"""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator


# Basic Enums and Types
class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"  # For compatibility with Claude Code router


class StopReason(str, Enum):
    END_TURN = "end_turn"
    MAX_TOKENS = "max_tokens"
    STOP_SEQUENCE = "stop_sequence"
    TOOL_USE = "tool_use"
    PAUSE_TURN = "pause_turn"
    REFUSAL = "refusal"


class ServiceTier(str, Enum):
    AUTO = "auto"
    STANDARD_ONLY = "standard_only"


class ToolChoiceType(str, Enum):
    AUTO = "auto"
    ANY = "any"
    NONE = "none"
    TOOL = "tool"


# Cache Control (for prompt caching)
class CacheControl(BaseModel):
    """Cache control for prompt caching."""

    type: Literal["ephemeral"] = "ephemeral"


# Content Blocks
class TextBlock(BaseModel):
    """Text content block."""

    type: Literal["text"] = "text"
    text: str


class ThinkingBlock(BaseModel):
    """Thinking content block for extended reasoning."""

    type: Literal["thinking"] = "thinking"
    thinking: str


class ToolUseBlock(BaseModel):
    """Tool use content block."""

    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: Dict[str, Any]


class ImageBlock(BaseModel):
    """Image content block (placeholder for future implementation)."""

    type: Literal["image"] = "image"
    source: Dict[str, Any]


class ToolResultBlock(BaseModel):
    """Tool result content block."""

    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: Union[str, List[Union[TextBlock, ImageBlock]]]
    is_error: Optional[bool] = False


# Content block union type
ContentBlock = Union[
    TextBlock, ThinkingBlock, ToolUseBlock, ToolResultBlock, ImageBlock
]


# Tool Definitions
class ToolInputSchema(BaseModel):
    """JSON schema for tool input."""

    type: str = "object"
    properties: Optional[Dict[str, Any]] = None
    required: Optional[List[str]] = None

    class Config:
        extra = "allow"  # Allow extra fields like $schema, additionalProperties


class OpenAIFunctionDef(BaseModel):
    """OpenAI function definition (nested inside tool)."""

    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class OpenAITool(BaseModel):
    """OpenAI tool format: {type: 'function', function: {...}}"""

    type: Literal["function"]
    function: OpenAIFunctionDef


class AnthropicToolDirect(BaseModel):
    """Anthropic tool format: {name: '...', input_schema: {...}}"""

    name: str = Field(..., max_length=200, pattern=r"^[a-zA-Z0-9_-]+$")
    description: Optional[str] = None
    input_schema: ToolInputSchema


class AnthropicTool(BaseModel):
    """Tool definition that accepts both Anthropic and OpenAI formats.

    Anthropic format: {"name": "...", "input_schema": {...}}
    OpenAI format: {"type": "function", "function": {"name": "...", "parameters": {...}}}
    """

    name: str = Field(default="", max_length=200)
    description: Optional[str] = None
    input_schema: Optional[ToolInputSchema] = None
    # OpenAI format fields
    type: Optional[str] = None
    function: Optional[OpenAIFunctionDef] = None

    @classmethod
    def model_validate(cls, obj: Any, *args, **kwargs):
        """Custom validation to handle both formats."""
        if isinstance(obj, dict):
            # Check if this is OpenAI format
            if obj.get("type") == "function" and "function" in obj:
                func = obj["function"]
                # Convert to Anthropic format
                return cls(
                    name=func.get("name", ""),
                    description=func.get("description"),
                    input_schema=ToolInputSchema(
                        type="object",
                        properties=func.get("parameters", {}).get("properties", {}),
                        required=func.get("parameters", {}).get("required", []),
                    ) if func.get("parameters") else ToolInputSchema(),
                )
        return super().model_validate(obj, *args, **kwargs)

    def __init__(self, **data):
        # Handle OpenAI format in constructor
        if data.get("type") == "function" and "function" in data:
            func = data["function"]
            if isinstance(func, dict):
                data["name"] = func.get("name", "")
                data["description"] = func.get("description")
                params = func.get("parameters", {})
                data["input_schema"] = ToolInputSchema(
                    type="object",
                    properties=params.get("properties", {}),
                    required=params.get("required", []),
                )
        super().__init__(**data)


class ToolChoiceAuto(BaseModel):
    """Automatic tool choice."""

    type: Literal["auto"] = "auto"
    disable_parallel_tool_use: Optional[bool] = False


class ToolChoiceAny(BaseModel):
    """Use any available tool."""

    type: Literal["any"] = "any"
    disable_parallel_tool_use: Optional[bool] = False


class ToolChoiceNone(BaseModel):
    """Don't use any tools."""

    type: Literal["none"] = "none"


class ToolChoiceTool(BaseModel):
    """Use specific tool."""

    type: Literal["tool"] = "tool"
    name: str
    disable_parallel_tool_use: Optional[bool] = False


ToolChoice = Union[ToolChoiceAuto, ToolChoiceAny, ToolChoiceNone, ToolChoiceTool]


# Thinking Configuration
class ThinkingConfigEnabled(BaseModel):
    """Enabled thinking configuration."""

    type: Literal["enabled"] = "enabled"
    budget_tokens: int = Field(..., ge=1024)


class ThinkingConfigDisabled(BaseModel):
    """Disabled thinking configuration."""

    type: Literal["disabled"] = "disabled"


ThinkingConfig = Union[ThinkingConfigEnabled, ThinkingConfigDisabled]


# Request Messages
class RequestTextBlock(BaseModel):
    """Text block in request."""

    type: Literal["text"] = "text"
    text: str
    cache_control: Optional[CacheControl] = None


class RequestImageBlock(BaseModel):
    """Image block in request (base64 format)."""

    type: Literal["image"] = "image"
    source: Dict[str, Any]  # Simplified for now
    cache_control: Optional[CacheControl] = None


class RequestToolUseBlock(BaseModel):
    """Tool use block in request."""

    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: Dict[str, Any]


class RequestToolResultBlock(BaseModel):
    """Tool result block in request."""

    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: Union[str, List[Union[RequestTextBlock, RequestImageBlock]]]
    is_error: Optional[bool] = False


class RequestThinkingBlock(BaseModel):
    """Thinking block in request (from previous assistant response)."""

    type: Literal["thinking"] = "thinking"
    thinking: str
    signature: Optional[str] = None


class RequestRedactedThinkingBlock(BaseModel):
    """Redacted thinking block in request."""

    type: Literal["redacted_thinking"] = "redacted_thinking"
    data: str


RequestContentBlock = Union[
    RequestTextBlock,
    RequestImageBlock,
    RequestToolUseBlock,
    RequestToolResultBlock,
    RequestThinkingBlock,
    RequestRedactedThinkingBlock,
]


class InputMessage(BaseModel):
    """Input message for Messages API."""

    role: MessageRole
    content: Union[str, List[RequestContentBlock]]


# System Prompt
class SystemTextBlock(BaseModel):
    """System text block."""

    type: Literal["text"] = "text"
    text: str
    cache_control: Optional[CacheControl] = None


SystemPrompt = Union[str, List[SystemTextBlock]]


# Metadata
class Metadata(BaseModel):
    """Request metadata."""

    user_id: Optional[str] = Field(None, max_length=256)


# Usage Statistics
class Usage(BaseModel):
    """Usage statistics."""

    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: Optional[int] = None
    cache_read_input_tokens: Optional[int] = None


# Main Request Model
class MessagesRequest(BaseModel):
    """Anthropic Messages API request."""

    # Required fields
    model: str = Field(..., max_length=256, min_length=1)
    messages: List[InputMessage]
    max_tokens: int = Field(..., ge=1)

    # Optional fields
    system: Optional[SystemPrompt] = None
    temperature: Optional[float] = Field(None, ge=0, le=1)
    top_p: Optional[float] = Field(None, ge=0, le=1)
    top_k: Optional[int] = Field(None, ge=0)
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    tools: Optional[List[AnthropicTool]] = None
    tool_choice: Optional[ToolChoice] = None
    thinking: Optional[ThinkingConfig] = None
    metadata: Optional[Metadata] = None
    service_tier: Optional[ServiceTier] = None

    # Allow extra fields for compatibility
    class Config:
        extra = "allow"

    @field_validator("temperature")
    def validate_temperature(cls, v):
        if v is not None and (v < 0 or v > 1):
            raise ValueError("Temperature must be between 0 and 1")
        return v

    @field_validator("top_p")
    def validate_top_p(cls, v):
        if v is not None and (v < 0 or v > 1):
            raise ValueError("Top_p must be between 0 and 1")
        return v


# Main Response Model
class MessagesResponse(BaseModel):
    """Anthropic Messages API response."""

    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: List[ContentBlock]
    model: str
    stop_reason: Optional[StopReason] = None
    stop_sequence: Optional[str] = None
    usage: Usage
    container: Optional[Dict[str, Any]] = None


# Streaming Models
class StreamEventType(str, Enum):
    MESSAGE_START = "message_start"
    MESSAGE_DELTA = "message_delta"
    MESSAGE_STOP = "message_stop"
    CONTENT_BLOCK_START = "content_block_start"
    CONTENT_BLOCK_DELTA = "content_block_delta"
    CONTENT_BLOCK_STOP = "content_block_stop"
    PING = "ping"


class StreamDelta(BaseModel):
    """Delta object for streaming."""

    type: Optional[str] = None
    text: Optional[str] = None
    thinking: Optional[str] = None
    partial_json: Optional[str] = None
    signature: Optional[str] = None
    stop_reason: Optional[StopReason] = None
    stop_sequence: Optional[str] = None
    usage: Optional[Usage] = None


class MessageStreamEvent(BaseModel):
    """Base streaming event."""

    type: StreamEventType

    # Event-specific data
    message: Optional[MessagesResponse] = None
    delta: Optional[StreamDelta] = None
    content_block: Optional[ContentBlock] = None
    index: Optional[int] = None
    usage: Optional[Usage] = None


# Error Models (for compatibility)
class AnthropicError(BaseModel):
    """Anthropic API error."""

    type: str
    message: str


class ErrorResponse(BaseModel):
    """Error response."""

    type: Literal["error"] = "error"
    error: AnthropicError
