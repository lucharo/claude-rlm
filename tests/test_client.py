"""Tests for ClaudeCodeClient."""

import pytest
from pathlib import Path

from claude_rlm.client import ClaudeCodeClient


class TestPreparePrompt:
    """Tests for _prepare_prompt method."""

    def test_string_prompt(self):
        """String prompt should be returned as-is."""
        client = ClaudeCodeClient(model_name="test-model")
        prompt, system = client._prepare_prompt("Hello, world!")

        assert prompt == "Hello, world!"
        assert system is None

    def test_string_prompt_with_system(self):
        """String prompt with pre-set system prompt."""
        client = ClaudeCodeClient(model_name="test-model")
        client._system_prompt = "You are helpful"
        prompt, system = client._prepare_prompt("Hello")

        assert prompt == "Hello"
        assert system == "You are helpful"

    def test_message_list_simple(self):
        """Simple message list with user message only."""
        client = ClaudeCodeClient(model_name="test-model")
        messages = [{"role": "user", "content": "Hello"}]
        prompt, system = client._prepare_prompt(messages)

        assert "User: Hello" in prompt
        assert system is None

    def test_message_list_with_system(self):
        """Message list with system message."""
        client = ClaudeCodeClient(model_name="test-model")
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]
        prompt, system = client._prepare_prompt(messages)

        assert "User: Hello" in prompt
        assert system == "You are helpful"

    def test_message_list_with_assistant(self):
        """Message list with assistant message."""
        client = ClaudeCodeClient(model_name="test-model")
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]
        prompt, system = client._prepare_prompt(messages)

        assert "User: Hello" in prompt
        assert "Assistant: Hi there!" in prompt
        assert "User: How are you?" in prompt

    def test_invalid_prompt_type(self):
        """Invalid prompt type should raise ValueError."""
        client = ClaudeCodeClient(model_name="test-model")

        with pytest.raises(ValueError, match="Invalid prompt type"):
            client._prepare_prompt(12345)


class TestBuildOptions:
    """Tests for _build_options method."""

    def test_default_options(self):
        """Default options for no tools."""
        client = ClaudeCodeClient(model_name="test-model")
        options = client._build_options()

        assert options.model == "test-model"
        assert options.max_turns == 1
        assert options.permission_mode == "bypassPermissions"
        assert options.allowed_tools is None  # None when no tools

    def test_with_tools(self):
        """Options with specific tools."""
        client = ClaudeCodeClient(model_name="test-model", allowed_tools=["Read", "Bash"])
        options = client._build_options()

        assert options.max_turns == 50
        assert "Read" in options.allowed_tools
        assert "Bash" in options.allowed_tools

    def test_all_tools(self):
        """Options with ALL_TOOLS."""
        from claude_rlm.client import ALL_TOOLS
        client = ClaudeCodeClient(model_name="test-model", allowed_tools=ALL_TOOLS)
        options = client._build_options()

        assert options.max_turns == 50
        assert len(options.allowed_tools) == len(ALL_TOOLS)

    def test_custom_permission_mode(self):
        """Custom permission mode."""
        client = ClaudeCodeClient(
            model_name="test-model",
            permission_mode="acceptEdits"
        )
        options = client._build_options()

        assert options.permission_mode == "acceptEdits"

    def test_system_prompt_passed(self):
        """System prompt passed to options."""
        client = ClaudeCodeClient(model_name="test-model")
        options = client._build_options(system_prompt="Be helpful")

        assert options.system_prompt == "Be helpful"


class TestUsageTracking:
    """Tests for usage tracking methods."""

    def test_initial_usage_summary(self):
        """Initial usage should be zero."""
        client = ClaudeCodeClient(model_name="test-model")
        usage = client.get_usage_summary()

        assert "test-model" in usage.model_usage_summaries
        assert usage.model_usage_summaries["test-model"].total_calls == 0

    def test_initial_last_usage(self):
        """Initial last usage should be zero."""
        client = ClaudeCodeClient(model_name="test-model")
        usage = client.get_last_usage()

        assert usage.total_calls == 0
        assert usage.total_input_tokens == 0


class TestClientConfiguration:
    """Tests for client configuration."""

    def test_default_model(self):
        """Default model should be set."""
        client = ClaudeCodeClient()
        assert client.model_name == "claude-sonnet-4-20250514"

    def test_default_cwd(self):
        """Default cwd should be current directory."""
        client = ClaudeCodeClient(model_name="test-model")
        assert client.cwd == Path.cwd().resolve()

    def test_custom_cwd(self):
        """Custom cwd should be resolved."""
        client = ClaudeCodeClient(model_name="test-model", cwd="/tmp")
        assert client.cwd == Path("/tmp").resolve()

    def test_permission_modes(self):
        """All permission modes should be valid."""
        for mode in ["default", "acceptEdits", "plan", "bypassPermissions"]:
            client = ClaudeCodeClient(model_name="test", permission_mode=mode)
            assert client.permission_mode == mode
