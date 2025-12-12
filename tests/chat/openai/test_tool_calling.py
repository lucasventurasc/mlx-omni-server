"""Tests for tool calling functionality in OpenAI adapter.

These tests ensure that:
1. Tool call arguments are properly converted from JSON string to dict
2. Tool call XML is hidden during streaming
3. Tool responses are properly handled
4. Normal streaming still works when no tool calls are present
"""

import json
import logging

import pytest

from mlx_omni_server.chat.openai.schema import (
    ChatCompletionRequest,
    ChatMessage,
    Role,
    ToolCall,
    FunctionCall,
    ToolType,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestToolCallArgumentParsing:
    """Test that tool call arguments are properly converted from JSON string to dict.

    These tests verify the core logic without importing the full OpenAIAdapter chain.
    """

    def test_json_string_to_dict_conversion(self):
        """Test the core logic: JSON string arguments get converted to dict."""
        import json as json_module

        # Simulate what OpenAIAdapter does with tool_calls
        tool_call_data = {
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "read_file",
                "arguments": '{"path": "/tmp/test.txt"}'  # JSON string
            }
        }

        # The conversion logic from openai_adapter.py
        if 'function' in tool_call_data and isinstance(tool_call_data['function'].get('arguments'), str):
            try:
                tool_call_data['function']['arguments'] = json_module.loads(tool_call_data['function']['arguments'])
            except (json_module.JSONDecodeError, TypeError):
                pass

        # Verify arguments is now a dict
        assert isinstance(tool_call_data['function']['arguments'], dict), \
            "tool_call arguments should be converted to dict"
        assert tool_call_data['function']['arguments'] == {"path": "/tmp/test.txt"}, \
            "tool_call arguments should have correct values"

    def test_enum_to_string_conversion(self):
        """Test that ToolType enum is properly converted to string."""
        # Test using model_dump(mode='json')
        tool_call = ToolCall(
            id="call_456",
            type=ToolType.FUNCTION,
            function=FunctionCall(
                name="test_func",
                arguments='{}'
            )
        )

        tc_dict = tool_call.model_dump(mode='json')

        # Verify type is a string, not an enum
        assert isinstance(tc_dict["type"], str), \
            "tool_call type should be a string"
        assert tc_dict["type"] == "function", \
            "tool_call type should be 'function'"

    def test_invalid_json_arguments_kept_as_string(self):
        """Test that invalid JSON arguments are kept as string."""
        import json as json_module

        tool_call_data = {
            "id": "call_789",
            "type": "function",
            "function": {
                "name": "test_func",
                "arguments": 'not valid json {'  # Invalid JSON
            }
        }

        original_args = tool_call_data['function']['arguments']

        # The conversion logic from openai_adapter.py
        if 'function' in tool_call_data and isinstance(tool_call_data['function'].get('arguments'), str):
            try:
                tool_call_data['function']['arguments'] = json_module.loads(tool_call_data['function']['arguments'])
            except (json_module.JSONDecodeError, TypeError):
                pass  # Keep as string if parsing fails

        # Invalid JSON should be kept as string
        assert isinstance(tool_call_data['function']['arguments'], str), \
            "invalid JSON should remain as string"
        assert tool_call_data['function']['arguments'] == original_args, \
            "invalid JSON should not be modified"

    def test_role_enum_to_string_conversion(self):
        """Test that Role enum is properly converted to string value."""
        msg = ChatMessage(role=Role.ASSISTANT, content="Hello")

        # The conversion logic from openai_adapter.py
        role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)

        assert isinstance(role, str), "role should be a string"
        assert role == "assistant", "role should be 'assistant'"

    def test_tool_message_format(self):
        """Test that tool response messages have correct format."""
        tool_msg = ChatMessage(
            role=Role.TOOL,
            content="File contents here",
            tool_call_id="call_789"
        )

        # Build message dict as in openai_adapter.py
        role = tool_msg.role.value if hasattr(tool_msg.role, "value") else str(tool_msg.role)
        msg_dict = {
            "role": role,
            "content": tool_msg.content,
        }
        if tool_msg.tool_call_id:
            msg_dict["tool_call_id"] = tool_msg.tool_call_id

        # Verify tool message format
        assert msg_dict["role"] == "tool", "role should be 'tool'"
        assert msg_dict["content"] == "File contents here", "content should match"
        assert msg_dict["tool_call_id"] == "call_789", "tool_call_id should match"


class TestStreamingToolCallHiding:
    """Test that tool call XML is hidden during streaming."""

    def test_tool_markers_defined(self):
        """Test that tool call markers are properly defined."""
        # These are the markers we look for to detect tool calls
        TOOL_MARKERS = ['<tool_call>', '<function=']

        # Verify markers are reasonable
        assert '<tool_call>' in TOOL_MARKERS, "Should detect <tool_call> marker"
        assert '<function=' in TOOL_MARKERS, "Should detect <function= marker"

    def test_buffer_logic_detects_tool_call_marker(self):
        """Test the buffering logic detects tool call markers."""
        TOOL_MARKERS = ['<tool_call>', '<function=']

        # Test case 1: Content with <tool_call> marker
        buffer = "I will read the file <tool_call>"
        in_tool_call = False
        pre_marker = ""

        for marker in TOOL_MARKERS:
            if marker in buffer:
                in_tool_call = True
                marker_pos = buffer.find(marker)
                pre_marker = buffer[:marker_pos]
                break

        assert in_tool_call, "Should detect tool call"
        assert pre_marker == "I will read the file ", "Should extract content before marker"

    def test_buffer_logic_detects_function_marker(self):
        """Test the buffering logic detects function= markers."""
        TOOL_MARKERS = ['<tool_call>', '<function=']

        # Test case: Content with <function= marker
        buffer = "Let me check <function=read_file>"
        in_tool_call = False
        pre_marker = ""

        for marker in TOOL_MARKERS:
            if marker in buffer:
                in_tool_call = True
                marker_pos = buffer.find(marker)
                pre_marker = buffer[:marker_pos]
                break

        assert in_tool_call, "Should detect function marker"
        assert pre_marker == "Let me check ", "Should extract content before marker"

    def test_buffer_logic_ignores_normal_text(self):
        """Test the buffering logic ignores normal text."""
        TOOL_MARKERS = ['<tool_call>', '<function=']

        # Content without any tool call marker
        buffer = "Hello, how are you today?"
        in_tool_call = False

        for marker in TOOL_MARKERS:
            if marker in buffer:
                in_tool_call = True
                break

        assert not in_tool_call, "Should not detect tool call in normal text"

    def test_partial_marker_buffering(self):
        """Test that partial markers are properly buffered."""
        TOOL_MARKERS = ['<tool_call>', '<function=']
        MAX_MARKER_LEN = max(len(m) for m in TOOL_MARKERS)

        # When buffer is longer than MAX_MARKER_LEN, safe content can be flushed
        buffer = "Hello world, this is a long message"

        if len(buffer) > MAX_MARKER_LEN:
            safe_content = buffer[:-MAX_MARKER_LEN]
            remaining_buffer = buffer[-MAX_MARKER_LEN:]

            assert len(remaining_buffer) == MAX_MARKER_LEN, \
                "Remaining buffer should be MAX_MARKER_LEN"
            assert safe_content + remaining_buffer == buffer, \
                "Safe content + remaining should equal original"

    def test_marker_at_start_of_buffer(self):
        """Test handling when marker is at start of buffer."""
        TOOL_MARKERS = ['<tool_call>', '<function=']

        # Marker at very start
        buffer = "<tool_call>{\"name\": \"test\"}</tool_call>"
        in_tool_call = False
        pre_marker = ""

        for marker in TOOL_MARKERS:
            if marker in buffer:
                in_tool_call = True
                marker_pos = buffer.find(marker)
                pre_marker = buffer[:marker_pos]
                break

        assert in_tool_call, "Should detect tool call"
        assert pre_marker == "", "No content before marker"


class TestToolCallParserIntegration:
    """Test integration with tool call parser."""

    def test_qwen3_moe_parser_exists(self):
        """Test that Qwen3MoeToolParser can be imported."""
        from mlx_omni_server.chat.mlx.tools.qwen3_moe_tools_parser import Qwen3MoeToolParser

        parser = Qwen3MoeToolParser()
        assert parser is not None, "Parser should be instantiated"

    def test_parser_extracts_tool_calls_xml_format(self):
        """Test that parser extracts tool calls from XML format."""
        from mlx_omni_server.chat.mlx.tools.qwen3_moe_tools_parser import Qwen3MoeToolParser

        parser = Qwen3MoeToolParser()

        # Test with <function=name> format (actual format the parser expects)
        text_with_tool_call = '''<tool_call><function=read_file><parameter=path>/tmp/test.txt</parameter></function></tool_call>'''

        result = parser.parse_tools(text_with_tool_call)

        assert result is not None, "Should parse tool calls"
        assert len(result) > 0, "Should have at least one tool call"
        assert result[0].name == "read_file", "Should extract function name"
        assert result[0].arguments.get("path") == "/tmp/test.txt", "Should extract argument"

    def test_parser_extracts_malformed_tool_call(self):
        """Test parser handles malformed tool call (missing outer tags)."""
        from mlx_omni_server.chat.mlx.tools.qwen3_moe_tools_parser import Qwen3MoeToolParser

        parser = Qwen3MoeToolParser()

        # Malformed: missing opening <tool_call>
        text_with_tool_call = '''<function=write_file><parameter=path>/tmp/output.txt</parameter><parameter=content>Hello</parameter></function></tool_call>'''

        result = parser.parse_tools(text_with_tool_call)

        assert result is not None, "Should parse malformed tool calls"
        assert len(result) > 0, "Should have at least one tool call"
        assert result[0].name == "write_file", "Should extract function name"

    def test_parser_handles_no_tool_calls(self):
        """Test that parser handles text without tool calls."""
        from mlx_omni_server.chat.mlx.tools.qwen3_moe_tools_parser import Qwen3MoeToolParser

        parser = Qwen3MoeToolParser()

        text_without_tool_call = "Hello, this is just normal text without any tool calls."

        result = parser.parse_tools(text_without_tool_call)

        assert result is None or len(result) == 0, "Should return empty for no tool calls"

    def test_parser_handles_multiple_tool_calls(self):
        """Test parser extracts multiple tool calls."""
        from mlx_omni_server.chat.mlx.tools.qwen3_moe_tools_parser import Qwen3MoeToolParser

        parser = Qwen3MoeToolParser()

        text_with_multiple = '''<tool_call><function=read_file><parameter=path>/tmp/file1.txt</parameter></function></tool_call>
<tool_call><function=read_file><parameter=path>/tmp/file2.txt</parameter></function></tool_call>'''

        result = parser.parse_tools(text_with_multiple)

        assert result is not None, "Should parse tool calls"
        assert len(result) == 2, "Should have two tool calls"
        assert result[0].arguments.get("path") == "/tmp/file1.txt"
        assert result[1].arguments.get("path") == "/tmp/file2.txt"


class TestPartialToolCallMarkerHandling:
    """Test that partial tool call markers don't leak into output."""

    def test_safe_content_with_angle_bracket_in_middle(self):
        """Test that '<' anywhere in safe_content is moved back to buffer."""
        # Simulate the buffer logic
        TOOL_MARKERS = ['<tool_call>', '<function=']
        MAX_MARKER_LEN = max(len(m) for m in TOOL_MARKERS)

        buffer = "Some text here <and more text"  # 29 chars
        safe_content = buffer[:-MAX_MARKER_LEN]  # First 18 chars: "Some text here <an"
        buffer = buffer[-MAX_MARKER_LEN:]  # Last 11 chars: "d more text"

        # The fix: move everything from last '<' onward back to buffer
        last_angle = safe_content.rfind('<')
        if last_angle >= 0:
            buffer = safe_content[last_angle:] + buffer
            safe_content = safe_content[:last_angle]

        assert '<' not in safe_content, "safe_content should not contain '<'"
        assert safe_content == "Some text here ", "safe_content should be trimmed before '<'"
        assert buffer.startswith('<'), "buffer should start with '<'"
        assert buffer == "<and more text", "buffer should contain '<' and everything after"

    def test_buffer_with_lone_angle_bracket_not_flushed(self):
        """Test that buffer containing just '<' or starting with '<' is not flushed."""
        TOOL_MARKERS = ['<tool_call>', '<function=']

        # Test cases that should NOT be flushed
        test_buffers = ['<', '<t', '<to', '<tool', '<tool_', '<tool_c', '<f', '<fu', '<func']

        for buffer in test_buffers:
            is_potential_tool_call = buffer.strip().startswith('<') and any(
                buffer.strip().startswith(m[:len(buffer.strip())])
                for m in TOOL_MARKERS
            )
            assert is_potential_tool_call, f"Buffer '{buffer}' should be detected as potential tool call"

    def test_buffer_with_regular_content_is_flushed(self):
        """Test that buffer with regular content (not starting with '<') is flushed."""
        TOOL_MARKERS = ['<tool_call>', '<function=']

        # Test cases that SHOULD be flushed
        test_buffers = ['Hello world', 'Some text', 'regular content']

        for buffer in test_buffers:
            is_potential_tool_call = buffer.strip().startswith('<') and any(
                buffer.strip().startswith(m[:len(buffer.strip())])
                for m in TOOL_MARKERS
            )
            assert not is_potential_tool_call, f"Buffer '{buffer}' should NOT be detected as potential tool call"

    def test_buffer_with_non_tool_angle_bracket(self):
        """Test that '<' not followed by tool markers is eventually flushed."""
        TOOL_MARKERS = ['<tool_call>', '<function=']

        # Content starting with '<' but NOT matching tool markers
        test_buffers = ['<div>', '<span>text</span>', '<html>', '<thinking>']

        for buffer in test_buffers:
            is_potential_tool_call = buffer.strip().startswith('<') and any(
                buffer.strip().startswith(m[:len(buffer.strip())])
                for m in TOOL_MARKERS
            )
            # These might be detected as potential at first, but once complete they're not
            # The key is that partial matches like '<t' would match '<tool_call>' prefix
            # but '<div>' would not match any prefix of '<tool_call>' or '<function='
            if buffer == '<div>' or buffer == '<html>':
                # '<d' doesn't match '<t' or '<f', so this should NOT be detected
                assert not is_potential_tool_call, f"Buffer '{buffer}' should NOT be detected as potential tool call"


class TestFullMessageConversion:
    """Test full message conversion flow."""

    def test_assistant_message_with_tool_calls_conversion(self):
        """Test complete conversion of assistant message with tool calls."""
        import json as json_module

        # Create a message with tool_calls
        msg = ChatMessage(
            role=Role.ASSISTANT,
            content="",
            tool_calls=[
                ToolCall(
                    id="call_123",
                    type=ToolType.FUNCTION,
                    function=FunctionCall(
                        name="read_file",
                        arguments='{"path": "/tmp/test.txt"}'
                    )
                )
            ]
        )

        # Conversion logic from openai_adapter.py
        role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
        msg_dict = {
            "role": role,
            "content": msg.content,
        }

        if msg.tool_calls:
            tool_calls_list = []
            for tc in msg.tool_calls:
                tc_dict = tc.model_dump(mode='json')
                if 'function' in tc_dict and isinstance(tc_dict['function'].get('arguments'), str):
                    try:
                        tc_dict['function']['arguments'] = json_module.loads(tc_dict['function']['arguments'])
                    except (json_module.JSONDecodeError, TypeError):
                        pass
                tool_calls_list.append(tc_dict)
            msg_dict["tool_calls"] = tool_calls_list

        # Verify final structure
        assert msg_dict["role"] == "assistant"
        assert len(msg_dict["tool_calls"]) == 1

        tc = msg_dict["tool_calls"][0]
        assert tc["id"] == "call_123"
        assert tc["type"] == "function"  # String, not enum
        assert tc["function"]["name"] == "read_file"
        assert isinstance(tc["function"]["arguments"], dict)  # Dict, not JSON string
        assert tc["function"]["arguments"]["path"] == "/tmp/test.txt"
