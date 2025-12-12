import time
import uuid
from typing import Generator

from mlx_omni_server.chat.mlx.chat_generator import DEFAULT_MAX_TOKENS, ChatGenerator
from mlx_omni_server.chat.openai.schema import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatMessage,
    Role,
)
from mlx_omni_server.utils.logger import logger


class OpenAIAdapter:
    """MLX Chat Model wrapper with internal parameter management"""

    def __init__(
        self,
        wrapper: ChatGenerator,
    ):
        """Initialize MLXModel with wrapper object.

        Args:
            wrapper: ChatGenerator instance (cached and ready to use)
        """
        self._default_max_tokens = DEFAULT_MAX_TOKENS
        self._generate_wrapper = wrapper

    def _prepare_generation_params(self, request: ChatCompletionRequest) -> dict:
        """Prepare common parameters for both generate and stream_generate."""
        max_tokens = (
            request.max_completion_tokens
            or request.max_tokens
            or self._default_max_tokens
        )

        # Extract parameters from request and extra params
        extra_params = request.get_extra_params()
        extra_body = extra_params.get("extra_body", {})

        # Prepare sampler configuration
        sampler_config = {
            "temp": 1.0 if request.temperature is None else request.temperature,
            "top_p": 1.0 if request.top_p is None else request.top_p,
            "top_k": extra_body.get("top_k", 0),
        }

        # Add additional sampler parameters from extra_body
        if extra_body.get("min_p") is not None:
            sampler_config["min_p"] = extra_body.get("min_p")
        if extra_body.get("min_tokens_to_keep") is not None:
            sampler_config["min_tokens_to_keep"] = extra_body.get("min_tokens_to_keep")
        if extra_body.get("xtc_probability") is not None:
            sampler_config["xtc_probability"] = extra_body.get("xtc_probability")
        if extra_body.get("xtc_threshold") is not None:
            sampler_config["xtc_threshold"] = extra_body.get("xtc_threshold")

        # Prepare template parameters - include both extra_body and direct extra params
        template_kwargs = dict(extra_body)

        # Handle direct extra parameters (for backward compatibility)
        for key in ["enable_thinking"]:
            if key in extra_params:
                template_kwargs[key] = extra_params[key]

        # Convert messages to dict format
        messages = []
        for msg in request.messages:
            role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
            msg_dict = {
                "role": role,
                "content": msg.content,
            }
            if msg.name:
                msg_dict["name"] = msg.name
            if msg.tool_calls:
                # Convert tool_calls to pure dicts with JSON serialization (for enum conversion)
                # Also parse arguments from JSON string to dict for templates that expect dict
                import json as json_module
                tool_calls_list = []
                for tc in msg.tool_calls:
                    tc_dict = tc.model_dump(mode='json') if hasattr(tc, "model_dump") else (dict(tc) if hasattr(tc, "__dict__") else tc)
                    # Parse arguments JSON string to dict if needed
                    if 'function' in tc_dict and isinstance(tc_dict['function'].get('arguments'), str):
                        try:
                            tc_dict['function']['arguments'] = json_module.loads(tc_dict['function']['arguments'])
                        except (json_module.JSONDecodeError, TypeError):
                            pass  # Keep as string if parsing fails
                    tool_calls_list.append(tc_dict)
                msg_dict["tool_calls"] = tool_calls_list
            if msg.tool_call_id:
                msg_dict["tool_call_id"] = msg.tool_call_id
            messages.append(msg_dict)

        # Convert tools to dict format with JSON serialization (for enum conversion)
        tools = None
        if request.tools:
            tools = [
                tool.model_dump(mode='json') if hasattr(tool, "model_dump") else dict(tool)
                for tool in request.tools
            ]

        # logger.debug(f"messages: {messages}")
        # logger.debug(f"template_kwargs: {template_kwargs}")

        json_schema = None
        if request.response_format and request.response_format.json_schema:
            json_schema = request.response_format.json_schema.schema_def

        return {
            "messages": messages,
            "tools": tools,
            "max_tokens": max_tokens,
            "sampler": sampler_config,
            "top_logprobs": request.top_logprobs if request.logprobs else None,
            "template_kwargs": template_kwargs,
            "enable_prompt_cache": True,
            "repetition_penalty": request.presence_penalty,
            "json_schema": json_schema,
        }

    def generate(
        self,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        """Generate complete response using the wrapper."""
        try:
            # Prepare parameters
            params = self._prepare_generation_params(request)

            # Directly use wrapper's generate method for complete response
            result = self._generate_wrapper.generate(**params)

            # logger.debug(f"Model Response:\n{result.content.text}")

            # Use reasoning from the wrapper's result
            final_content = result.content.text
            reasoning_content = result.content.reasoning

            # Include tool_calls in response if present (from request.tools OR parsed from content)
            # Convert from internal ToolCall format to OpenAI schema format
            schema_tool_calls = None
            if result.content.tool_calls:
                import json as json_module
                from .schema import ToolCall as SchemaToolCall, FunctionCall, ToolType

                schema_tool_calls = [
                    SchemaToolCall(
                        id=tc.id,
                        type=ToolType.FUNCTION,
                        function=FunctionCall(
                            name=tc.name,
                            arguments=json_module.dumps(tc.arguments) if tc.arguments else "{}"
                        )
                    )
                    for tc in result.content.tool_calls
                ]

            message = ChatMessage(
                role=Role.ASSISTANT,
                content=final_content,
                tool_calls=schema_tool_calls,
                reasoning=reasoning_content,
            )

            # Use cached tokens from wrapper stats
            cached_tokens = result.stats.cache_hit_tokens
            logger.debug(f"Generate response with {cached_tokens} cached tokens")

            prompt_tokens_details = None
            if cached_tokens > 0:
                from .schema import PromptTokensDetails

                prompt_tokens_details = PromptTokensDetails(cached_tokens=cached_tokens)

            return ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex[:10]}",
                created=int(time.time()),
                model=request.model,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=message,
                        finish_reason=(
                            "tool_calls"
                            if message.tool_calls
                            else (result.finish_reason or "stop")
                        ),
                        logprobs=result.logprobs,
                    )
                ],
                usage=ChatCompletionUsage(
                    prompt_tokens=result.stats.prompt_tokens + cached_tokens,
                    completion_tokens=result.stats.completion_tokens,
                    total_tokens=result.stats.prompt_tokens
                    + result.stats.completion_tokens
                    + cached_tokens,
                    prompt_tokens_details=prompt_tokens_details,
                ),
            )
        except Exception as e:
            logger.error(f"Failed to generate completion: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to generate completion: {str(e)}")

    def generate_stream(
        self,
        request: ChatCompletionRequest,
    ) -> Generator[ChatCompletionChunk, None, None]:
        """Stream generate OpenAI-compatible chunks."""
        try:
            chat_id = f"chatcmpl-{uuid.uuid4().hex[:10]}"

            # Prepare parameters
            params = self._prepare_generation_params(request)

            result = None
            accumulated_text = ""  # Accumulate all text for tool parsing at end
            buffer = ""  # Buffer for detecting tool call start
            in_tool_call = False  # Flag to stop streaming once tool call detected

            # Tool call markers to detect
            TOOL_MARKERS = ['<tool_call>', '<function=']
            MAX_MARKER_LEN = max(len(m) for m in TOOL_MARKERS)

            for chunk in self._generate_wrapper.generate_stream(**params):
                created = int(time.time())

                # For streaming, get the delta content
                if chunk.content.text_delta:
                    content = chunk.content.text_delta
                    accumulated_text += content
                elif chunk.content.reasoning_delta:
                    content = chunk.content.reasoning_delta
                else:
                    content = ""

                # Smart buffering to hide tool call XML
                if content and not in_tool_call:
                    buffer += content

                    # Check if we've started a tool call
                    for marker in TOOL_MARKERS:
                        if marker in buffer:
                            in_tool_call = True
                            # Send any content before the marker
                            marker_pos = buffer.find(marker)
                            if marker_pos > 0:
                                pre_marker = buffer[:marker_pos]
                                message = ChatMessage(role=Role.ASSISTANT, content=pre_marker)
                                yield ChatCompletionChunk(
                                    id=chat_id,
                                    created=created,
                                    model=request.model,
                                    choices=[
                                        ChatCompletionChunkChoice(
                                            index=0,
                                            delta=message,
                                            finish_reason=None,
                                            logprobs=chunk.logprobs,
                                        )
                                    ],
                                )
                            buffer = ""
                            break

                    # If not in tool call and buffer is long enough, flush safe content
                    if not in_tool_call and len(buffer) > MAX_MARKER_LEN:
                        # Keep the last MAX_MARKER_LEN chars in buffer for marker detection
                        safe_content = buffer[:-MAX_MARKER_LEN]
                        buffer = buffer[-MAX_MARKER_LEN:]

                        # Check if safe_content has '<' that could be start of tool marker
                        # Move everything from last '<' onward back to buffer
                        last_angle = safe_content.rfind('<')
                        if last_angle >= 0:
                            buffer = safe_content[last_angle:] + buffer
                            safe_content = safe_content[:last_angle]

                        if safe_content:
                            message = ChatMessage(role=Role.ASSISTANT, content=safe_content)
                            yield ChatCompletionChunk(
                                id=chat_id,
                                created=created,
                                model=request.model,
                                choices=[
                                    ChatCompletionChunkChoice(
                                        index=0,
                                        delta=message,
                                        finish_reason=None,
                                        logprobs=chunk.logprobs,
                                    )
                                ],
                            )

                result = chunk

            # Flush remaining buffer if no tool call was detected
            # But don't flush if buffer looks like start of tool call (starts with '<')
            if buffer and not in_tool_call:
                # Check if buffer might be incomplete tool call
                is_potential_tool_call = buffer.strip().startswith('<') and any(
                    buffer.strip().startswith(m[:len(buffer.strip())])
                    for m in TOOL_MARKERS
                )
                if not is_potential_tool_call:
                    message = ChatMessage(role=Role.ASSISTANT, content=buffer)
                    yield ChatCompletionChunk(
                        id=chat_id,
                        created=int(time.time()),
                        model=request.model,
                        choices=[
                            ChatCompletionChunkChoice(
                                index=0,
                                delta=message,
                                finish_reason=None,
                                logprobs=None,
                            )
                        ],
                    )

            # After streaming completes, emit final chunk with finish_reason
            # Check for tool calls in accumulated text
            tool_calls = None
            final_finish_reason = "stop"

            if accumulated_text and ('<function=' in accumulated_text or '<tool_call>' in accumulated_text):
                from mlx_omni_server.chat.mlx.tools.qwen3_moe_tools_parser import Qwen3MoeToolParser
                import json as json_module
                from .schema import ToolCall as SchemaToolCall, FunctionCall, ToolType

                parser = Qwen3MoeToolParser()
                parsed_tools = parser.parse_tools(accumulated_text)
                if parsed_tools:
                    tool_calls = [
                        SchemaToolCall(
                            id=tc.id,
                            type=ToolType.FUNCTION,
                            function=FunctionCall(
                                name=tc.name,
                                arguments=json_module.dumps(tc.arguments) if tc.arguments else "{}"
                            )
                        )
                        for tc in parsed_tools
                    ]
                    final_finish_reason = "tool_calls"

            # Always emit a final chunk with finish_reason
            yield ChatCompletionChunk(
                id=chat_id,
                created=int(time.time()),
                model=request.model,
                choices=[
                    ChatCompletionChunkChoice(
                        index=0,
                        delta=ChatMessage(
                            role=Role.ASSISTANT,
                            content="",
                            tool_calls=tool_calls
                        ),
                        finish_reason=final_finish_reason,
                        logprobs=None,
                    )
                ],
            )

            if (
                request.stream_options
                and request.stream_options.include_usage
                and result is not None
            ):
                created = int(time.time())
                cached_tokens = result.stats.cache_hit_tokens
                logger.debug(f"Stream response with {cached_tokens} cached tokens")

                prompt_tokens_details = None
                if cached_tokens > 0:
                    from .schema import PromptTokensDetails

                    prompt_tokens_details = PromptTokensDetails(
                        cached_tokens=cached_tokens
                    )

                yield ChatCompletionChunk(
                    id=chat_id,
                    created=created,
                    model=request.model,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0,
                            delta=ChatMessage(role=Role.ASSISTANT),
                            finish_reason="stop",
                            logprobs=None,
                        )
                    ],
                    usage=ChatCompletionUsage(
                        prompt_tokens=result.stats.prompt_tokens + cached_tokens,
                        completion_tokens=result.stats.completion_tokens,
                        total_tokens=result.stats.prompt_tokens
                        + result.stats.completion_tokens
                        + cached_tokens,
                        prompt_tokens_details=prompt_tokens_details,
                    ),
                )

        except Exception as e:
            logger.error(f"Error during stream generation: {str(e)}", exc_info=True)
            raise
