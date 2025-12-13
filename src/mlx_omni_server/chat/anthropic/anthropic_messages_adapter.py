"""Anthropic Messages API Adapter

This module provides an adapter to convert between Anthropic Messages API format
and the internal MLX generation interface.
"""

import uuid
from typing import Any, Dict, Generator, List, Optional

from mlx_omni_server.chat.anthropic.anthropic_schema import (
    AnthropicTool,
    ContentBlock,
    InputMessage,
    MessagesRequest,
    MessagesResponse,
    MessageStreamEvent,
    RequestRedactedThinkingBlock,
    RequestTextBlock,
    RequestThinkingBlock,
    RequestToolResultBlock,
    RequestToolUseBlock,
    StopReason,
    StreamDelta,
    StreamEventType,
    SystemPrompt,
    TextBlock,
    ThinkingBlock,
    ThinkingConfigEnabled,
    ToolUseBlock,
    Usage,
)
from mlx_omni_server.chat.mlx.chat_generator import ChatGenerator
from mlx_omni_server.utils.logger import logger


class AnthropicMessagesAdapter:
    """Anthropic Messages API adapter with internal parameter management."""

    def __init__(self, wrapper: ChatGenerator):
        """Initialize adapter with wrapper object.

        Args:
            wrapper: ChatGenerator instance (cached and ready to use)
        """
        self._default_max_tokens = 2048
        self._generate_wrapper = wrapper

    def _convert_system_to_messages(
        self, system: Optional[SystemPrompt], messages: List[InputMessage]
    ) -> List[Dict[str, Any]]:
        """Convert system prompt and messages to MLX format.

        Args:
            system: System prompt (string or list of text blocks)
            messages: Input messages

        Returns:
            List of messages in MLX format
        """
        mlx_messages = []

        # Tool use guidance for better agentic behavior
        # This helps smaller models use tools more effectively
        tool_guidance = """
IMPORTANT TOOL USE GUIDELINES:
1. ALWAYS use Read tool BEFORE Edit - never edit a file you haven't read
2. Use Edit with SMALL, SURGICAL changes - only the exact lines that need to change
3. NEVER output entire file contents in your response - use Write or Edit tools instead
4. Break complex tasks into steps using TodoWrite
5. One tool call at a time, verify each works before proceeding
6. For Edit: old_string must match EXACTLY (including whitespace)
"""

        # Convert system prompt to system message if present
        if system:
            system_content = ""
            if isinstance(system, str):
                system_content = system
            else:
                # List of SystemTextBlock
                system_content = "\n".join(block.text for block in system)

            # Prepend tool guidance to system prompt
            system_content = tool_guidance + "\n\n" + system_content

            mlx_messages.append(
                {
                    "role": "system",
                    "content": system_content,
                }
            )

        # Convert input messages
        for msg in messages:
            # Handle system messages in the messages array (Claude Code compatibility)
            if msg.role.value == "system":
                # Extract content and add as system message
                if isinstance(msg.content, str):
                    system_text = msg.content
                else:
                    # List of content blocks
                    system_text = "\n".join(
                        block.text for block in msg.content
                        if isinstance(block, RequestTextBlock)
                    )
                mlx_messages.append({"role": "system", "content": system_text})
                continue
            mlx_msg = {
                "role": msg.role.value,
            }

            # Handle content
            if isinstance(msg.content, str):
                mlx_msg["content"] = msg.content
            else:
                # List of content blocks - convert to appropriate format
                content_parts = []
                for block in msg.content:
                    if isinstance(block, RequestTextBlock):
                        content_parts.append(block.text)
                    elif isinstance(block, RequestThinkingBlock):
                        # Include thinking content as part of assistant's message
                        # (This is from previous turn's thinking, kept for context)
                        pass  # Skip thinking blocks - they're internal reasoning
                    elif isinstance(block, RequestRedactedThinkingBlock):
                        # Skip redacted thinking blocks
                        pass
                    elif isinstance(block, RequestToolUseBlock):
                        # Handle tool use blocks
                        mlx_msg["tool_calls"] = [
                            {
                                "id": block.id,
                                "type": "function",
                                "function": {
                                    "name": block.name,
                                    "arguments": block.input,
                                },
                            }
                        ]
                    elif isinstance(block, RequestToolResultBlock):
                        # Handle tool result blocks
                        tool_content = block.content
                        if isinstance(tool_content, str):
                            content_parts.append(tool_content)
                        else:
                            # List of blocks
                            for sub_block in tool_content:
                                if isinstance(sub_block, RequestTextBlock):
                                    content_parts.append(sub_block.text)

                        mlx_msg["tool_call_id"] = block.tool_use_id
                        if block.is_error:
                            mlx_msg["name"] = "error"
                    # Note: Image blocks would be handled here too

                if content_parts:
                    mlx_msg["content"] = "\n".join(content_parts)
                else:
                    # Ensure content is always present (required by Jinja template)
                    mlx_msg["content"] = ""

            mlx_messages.append(mlx_msg)

        return mlx_messages

    # Tools that require state tracking - see PlanModeTracker
    # ExitPlanMode can only be called if EnterPlanMode was called first
    PLAN_MODE_TOOLS = {'ExitPlanMode', 'EnterPlanMode'}

    def _convert_tools_to_mlx(
        self, tools: Optional[List[AnthropicTool]]
    ) -> Optional[List[Dict[str, Any]]]:
        """Convert Anthropic tools to MLX format.

        Args:
            tools: List of Anthropic tools

        Returns:
            List of tools in MLX format
        """
        if not tools:
            return None

        mlx_tools = []
        for tool in tools:
            mlx_tool = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": {
                        "type": tool.input_schema.type,
                        "properties": tool.input_schema.properties or {},
                        "required": tool.input_schema.required or [],
                    },
                },
            }
            mlx_tools.append(mlx_tool)

        return mlx_tools

    def _prepare_generation_params(self, request: MessagesRequest, temp_boost: float = 0.0) -> Dict[str, Any]:
        """Prepare parameters for MLX generation.

        Args:
            request: Anthropic Messages API request
            temp_boost: Additional temperature to add (for breaking out of loops)

        Returns:
            Parameters for ChatGenerator
        """
        # Convert messages
        messages = self._convert_system_to_messages(request.system, request.messages)

        # Convert tools
        tools = self._convert_tools_to_mlx(request.tools)

        # Template parameters
        template_kwargs = {}

        # Handle thinking configuration
        # ALWAYS disable thinking for local models - they don't support it properly
        # Claude Code may request thinking but local models can't handle it
        # TODO: Re-enable when local model thinking is properly supported
        template_kwargs["enable_thinking"] = False

        # Prepare sampler configuration
        # Use global settings from UI as defaults, with request values taking precedence
        try:
            from extensions.global_settings import get_global_settings
            sampler_config = get_global_settings().get_sampler_config(
                request_temp=request.temperature,
                request_top_p=request.top_p,
                request_top_k=request.top_k
            )
        except ImportError:
            # Fallback if global_settings not available
            # Conservative defaults for reasoning (low hallucination)
            sampler_config = {
                "temp": request.temperature if request.temperature is not None else 0.3,
                "top_p": request.top_p if request.top_p is not None else 0.85,
                "top_k": request.top_k if request.top_k is not None else 30,
            }

        # Apply temperature boost if provided (to break out of repetition loops)
        if temp_boost > 0:
            original_temp = sampler_config['temp']
            sampler_config['temp'] = min(1.0, sampler_config['temp'] + temp_boost)
            logger.info(f"Applied temp boost: {original_temp} -> {sampler_config['temp']} (boost={temp_boost})")

        logger.debug(f"Anthropic messages count: {len(messages)}")
        logger.debug(f"Anthropic tools count: {len(tools) if tools else 0}")
        logger.debug(f"Sampler config: temp={sampler_config['temp']}, top_p={sampler_config['top_p']}, top_k={sampler_config['top_k']}")
        if tools:
            logger.debug(f"First tool: {tools[0].get('function', {}).get('name', tools[0].get('name', 'unknown'))}")

        params = {
            "messages": messages,
            "tools": tools,
            "max_tokens": request.max_tokens,
            "sampler": sampler_config,
            "template_kwargs": template_kwargs,
            "enable_prompt_cache": True,
        }

        # Note: ChatGenerator doesn't currently support stop_sequences
        # This is a known limitation that will be addressed in the future
        # if request.stop_sequences:
        #     params["stop_sequences"] = request.stop_sequences

        return params

    def _create_content_blocks(
        self,
        text_content: Optional[str],
        reasoning_content: Optional[str],
        tool_calls: Optional[List[Any]] = None,
    ) -> List[ContentBlock]:
        """Create content blocks from generation result.

        Args:
            text_content: Main text content
            reasoning_content: Thinking/reasoning content
            tool_calls: Tool calls from generation

        Returns:
            List of content blocks
        """
        blocks = []

        # Add thinking block first if present
        if reasoning_content:
            blocks.append(ThinkingBlock(thinking=reasoning_content))

        # Add text block if present
        if text_content:
            blocks.append(TextBlock(text=text_content))

        # Add tool use blocks
        if tool_calls:
            for tool_call in tool_calls:
                blocks.append(
                    ToolUseBlock(
                        id=tool_call.id,
                        name=tool_call.name,
                        input=tool_call.arguments,
                    )
                )

        # Ensure we always have at least one content block
        if not blocks:
            blocks.append(TextBlock(text=""))

        return blocks

    def _map_finish_reason(
        self, finish_reason: Optional[str], has_tool_calls: bool
    ) -> StopReason:
        """Map internal finish reason to Anthropic format.

        Args:
            finish_reason: Internal finish reason
            has_tool_calls: Whether response has tool calls

        Returns:
            Anthropic stop reason
        """
        if has_tool_calls:
            return StopReason.TOOL_USE

        if finish_reason == "stop":
            return StopReason.END_TURN
        elif finish_reason == "length":
            return StopReason.MAX_TOKENS
        elif finish_reason == "stop_sequence":
            return StopReason.STOP_SEQUENCE
        else:
            return StopReason.END_TURN

    def generate(self, request: MessagesRequest) -> MessagesResponse:
        """Generate complete response using the wrapper.

        Args:
            request: Anthropic Messages API request

        Returns:
            Anthropic Messages API response
        """
        try:
            # Prepare parameters
            params = self._prepare_generation_params(request)

            # Generate using wrapper
            result = self._generate_wrapper.generate(**params)

            # Debug logging
            logger.info(f"Generation result: text='{result.content.text[:100] if result.content.text else 'None'}...', "
                       f"finish_reason={result.finish_reason}, "
                       f"output_tokens={result.stats.completion_tokens if result.stats else 0}")

            # Create content blocks
            content_blocks = self._create_content_blocks(
                text_content=result.content.text,
                reasoning_content=result.content.reasoning,
                tool_calls=result.content.tool_calls,
            )

            # Map stop reason
            stop_reason = self._map_finish_reason(
                result.finish_reason, bool(result.content.tool_calls)
            )

            # Create usage statistics
            cached_tokens = result.stats.cache_hit_tokens
            usage = Usage(
                input_tokens=result.stats.prompt_tokens + cached_tokens,
                output_tokens=result.stats.completion_tokens,
                cache_read_input_tokens=cached_tokens if cached_tokens > 0 else None,
            )

            return MessagesResponse(
                id=f"msg_{uuid.uuid4().hex[:24]}",
                content=content_blocks,
                model=request.model,
                stop_reason=stop_reason,
                usage=usage,
            )

        except Exception as e:
            logger.error(
                f"Failed to generate Anthropic completion: {str(e)}", exc_info=True
            )
            raise RuntimeError(f"Failed to generate completion: {str(e)}")

    def generate_stream(
        self, request: MessagesRequest, temp_boost: float = 0.0
    ) -> Generator[MessageStreamEvent, None, None]:
        """Generate streaming response.

        Args:
            request: Anthropic Messages API request
            temp_boost: Additional temperature to add (for breaking out of loops)

        Yields:
            Anthropic streaming events
        """
        try:
            message_id = f"msg_{uuid.uuid4().hex[:24]}"

            # Prepare parameters with optional temperature boost
            params = self._prepare_generation_params(request, temp_boost=temp_boost)

            # Start message event
            yield MessageStreamEvent(
                type=StreamEventType.MESSAGE_START,
                message=MessagesResponse(
                    id=message_id,
                    content=[],
                    model=request.model,
                    stop_reason=None,
                    usage=Usage(input_tokens=0, output_tokens=0),
                ),
            )

            # Track content for final message
            accumulated_text = ""
            accumulated_reasoning = ""
            final_result = None
            current_block_index = 0
            in_thinking = False
            chunk_count = 0

            logger.info(f"Starting stream generation...")
            for chunk in self._generate_wrapper.generate_stream(**params):
                chunk_count += 1
                if chunk_count <= 3 or chunk_count % 50 == 0:
                    logger.debug(f"Stream chunk {chunk_count}: text_delta='{chunk.content.text_delta[:50] if chunk.content.text_delta else None}'")
                # Determine content type and send appropriate events
                if chunk.content.reasoning_delta:
                    # Thinking content
                    if not in_thinking:
                        # Start thinking block
                        yield MessageStreamEvent(
                            type=StreamEventType.CONTENT_BLOCK_START,
                            index=current_block_index,
                            content_block=ThinkingBlock(thinking=""),
                        )
                        in_thinking = True

                    # Thinking delta
                    yield MessageStreamEvent(
                        type=StreamEventType.CONTENT_BLOCK_DELTA,
                        index=current_block_index,
                        delta=StreamDelta(
                            type="thinking_delta",
                            thinking=chunk.content.reasoning_delta,
                        ),
                    )
                    accumulated_reasoning += chunk.content.reasoning_delta

                elif chunk.content.text_delta:
                    # Text content
                    if in_thinking:
                        # End thinking block
                        yield MessageStreamEvent(
                            type=StreamEventType.CONTENT_BLOCK_STOP,
                            index=current_block_index,
                        )
                        current_block_index += 1
                        in_thinking = False

                        # Start text block
                        yield MessageStreamEvent(
                            type=StreamEventType.CONTENT_BLOCK_START,
                            index=current_block_index,
                            content_block=TextBlock(text=""),
                        )
                    elif not accumulated_text:
                        # First text chunk - start text block
                        yield MessageStreamEvent(
                            type=StreamEventType.CONTENT_BLOCK_START,
                            index=current_block_index,
                            content_block=TextBlock(text=""),
                        )

                    # Check if this delta contains tool call XML that should be hidden
                    # Tool calls use <function=name> or <tool_call> format
                    text_to_send = chunk.content.text_delta

                    # If we detect start of tool call XML, don't stream it
                    # Instead, accumulate it silently for parsing at end
                    if '<function=' in accumulated_text or '<tool_call>' in accumulated_text:
                        # Already in tool call mode - don't stream anything
                        accumulated_text += chunk.content.text_delta
                        continue
                    elif '<function=' in text_to_send or '<tool_call>' in text_to_send:
                        # Starting tool call - send text before the marker, hide the rest
                        marker = '<function=' if '<function=' in text_to_send else '<tool_call>'
                        marker_pos = text_to_send.find(marker)
                        text_before = text_to_send[:marker_pos]

                        if text_before.strip():
                            yield MessageStreamEvent(
                                type=StreamEventType.CONTENT_BLOCK_DELTA,
                                index=current_block_index,
                                delta=StreamDelta(
                                    type="text_delta", text=text_before
                                ),
                            )
                        accumulated_text += chunk.content.text_delta
                        continue

                    # Normal text delta - stream it
                    yield MessageStreamEvent(
                        type=StreamEventType.CONTENT_BLOCK_DELTA,
                        index=current_block_index,
                        delta=StreamDelta(
                            type="text_delta", text=chunk.content.text_delta
                        ),
                    )
                    accumulated_text += chunk.content.text_delta

                final_result = chunk

            # Add signature delta for thinking blocks if we had thinking content
            if in_thinking and accumulated_reasoning:
                # Add a placeholder signature for thinking block integrity
                yield MessageStreamEvent(
                    type=StreamEventType.CONTENT_BLOCK_DELTA,
                    index=current_block_index,
                    delta=StreamDelta(
                        type="signature_delta", signature="placeholder_signature_hash"
                    ),
                )

            # End final content block (text or thinking)
            if accumulated_text or accumulated_reasoning:
                yield MessageStreamEvent(
                    type=StreamEventType.CONTENT_BLOCK_STOP, index=current_block_index
                )
                current_block_index += 1

            # Parse tool calls from accumulated text at end of stream
            tool_calls = None
            if accumulated_text:
                chat_result = self._generate_wrapper.chat_template.parse_chat_response(
                    accumulated_text
                )
                tool_calls = chat_result.tool_calls

                # Filter out tool calls with empty or missing required inputs
                # This prevents loops where the model generates tool calls without parameters
                if tool_calls:
                    valid_tool_calls = []
                    for tc in tool_calls:
                        if tc.arguments and len(tc.arguments) > 0:
                            valid_tool_calls.append(tc)
                        else:
                            logger.warning(f"Filtered out tool call '{tc.name}' with empty arguments")
                    tool_calls = valid_tool_calls if valid_tool_calls else None

                # If we found tool calls, emit ToolUseBlock events for each
                if tool_calls:
                    import json as json_module
                    for tool_call in tool_calls:
                        # Start tool_use block with empty input (per Anthropic protocol)
                        yield MessageStreamEvent(
                            type=StreamEventType.CONTENT_BLOCK_START,
                            index=current_block_index,
                            content_block=ToolUseBlock(
                                id=tool_call.id,
                                name=tool_call.name,
                                input={},
                            ),
                        )
                        # Send input_json_delta with the full JSON
                        # (Anthropic protocol allows chunked partial_json, but full JSON works too)
                        input_json = json_module.dumps(tool_call.arguments)
                        yield MessageStreamEvent(
                            type=StreamEventType.CONTENT_BLOCK_DELTA,
                            index=current_block_index,
                            delta=StreamDelta(
                                type="input_json_delta",
                                partial_json=input_json,
                            ),
                        )
                        # End tool_use block
                        yield MessageStreamEvent(
                            type=StreamEventType.CONTENT_BLOCK_STOP,
                            index=current_block_index,
                        )
                        current_block_index += 1

            # Map stop reason and usage
            has_tool_calls = tool_calls is not None and len(tool_calls) > 0
            if final_result:
                cached_tokens = final_result.stats.cache_hit_tokens
                usage = Usage(
                    input_tokens=final_result.stats.prompt_tokens + cached_tokens,
                    output_tokens=final_result.stats.completion_tokens,
                    cache_read_input_tokens=(
                        cached_tokens if cached_tokens > 0 else None
                    ),
                )

                stop_reason = self._map_finish_reason(
                    final_result.finish_reason,
                    has_tool_calls,
                )
            else:
                usage = Usage(input_tokens=0, output_tokens=0)
                stop_reason = StopReason.TOOL_USE if has_tool_calls else StopReason.END_TURN

            logger.info(f"Stream complete: {chunk_count} chunks, tool_calls={len(tool_calls) if tool_calls else 0}, accumulated_text='{accumulated_text[:100] if accumulated_text else 'EMPTY'}...'")

            # Message delta event with stop reason and usage
            yield MessageStreamEvent(
                type=StreamEventType.MESSAGE_DELTA,
                delta=StreamDelta(stop_reason=stop_reason),
                usage=usage,
            )

            # Message stop event (no delta)
            yield MessageStreamEvent(type=StreamEventType.MESSAGE_STOP)

        except Exception as e:
            logger.error(
                f"Error during Anthropic stream generation: {str(e)}", exc_info=True
            )
            raise
