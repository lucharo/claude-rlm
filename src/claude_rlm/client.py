"""ClaudeCodeClient - RLM BaseLM implementation using claude-agent-sdk."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from typing import Literal

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    query,
)

PermissionMode = Literal["default", "acceptEdits", "plan", "bypassPermissions"]
from rlm.clients.base_lm import BaseLM
from rlm.core.types import ModelUsageSummary, UsageSummary


@dataclass
class ClaudeCodeClient(BaseLM):
    """RLM client using Claude Code via claude-agent-sdk.

    This client implements RLM's BaseLM interface, allowing Claude Code
    to be used as the backend for recursive language model operations.

    Attributes:
        model_name: The Claude model to use (default: claude-sonnet-4-20250514)
        agentic: Whether to enable Claude's tools (Read, Write, Bash, etc.)
        cwd: Working directory for agentic operations
        max_tokens: Maximum tokens for response
        verbose: Enable verbose output
        permission_mode: How to handle tool permissions:
            - "default": Normal permission prompts
            - "acceptEdits": Auto-accept file edits
            - "plan": Planning mode
            - "bypassPermissions": Skip all permission checks (use with caution)
    """

    model_name: str = "claude-sonnet-4-20250514"
    agentic: bool = False
    cwd: Path | None = None
    max_tokens: int = 16384
    verbose: bool = False
    permission_mode: PermissionMode = "bypassPermissions"
    _system_prompt: str | None = field(default=None, repr=False)

    # Usage tracking
    _total_calls: int = field(default=0, repr=False)
    _total_input_tokens: int = field(default=0, repr=False)
    _total_output_tokens: int = field(default=0, repr=False)
    _last_input_tokens: int = field(default=0, repr=False)
    _last_output_tokens: int = field(default=0, repr=False)

    def __post_init__(self):
        """Initialize the client."""
        if self.cwd is None:
            self.cwd = Path.cwd()
        self.cwd = Path(self.cwd).resolve()

    @property
    def system_prompt(self) -> str | None:
        """Get the system prompt."""
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, value: str | None) -> None:
        """Set the system prompt."""
        self._system_prompt = value

    def _prepare_prompt(
        self, prompt: str | list[dict[str, Any]]
    ) -> tuple[str, str | None]:
        """Convert prompt to string format and extract system prompt.

        Args:
            prompt: Either a string or a list of message dicts (OpenAI chat format)

        Returns:
            Tuple of (user_prompt, system_prompt)
        """
        if isinstance(prompt, str):
            return prompt, self._system_prompt

        if isinstance(prompt, list) and all(isinstance(item, dict) for item in prompt):
            system_prompt = None
            user_messages: list[str] = []

            for msg in prompt:
                role = msg.get("role", "")
                content = msg.get("content", "")

                if role == "system":
                    system_prompt = content
                elif role == "user":
                    user_messages.append(f"User: {content}")
                elif role == "assistant":
                    user_messages.append(f"Assistant: {content}")

            # Combine system and user prompts
            combined_prompt = "\n\n".join(user_messages) if user_messages else ""
            return combined_prompt, system_prompt or self._system_prompt

        raise ValueError(f"Invalid prompt type: {type(prompt)}")

    def _build_options(self, system_prompt: str | None = None) -> ClaudeAgentOptions:
        """Build ClaudeAgentOptions for the query."""
        # Determine allowed tools based on agentic mode
        if self.agentic:
            allowed_tools = [
                "Read",
                "Write",
                "Edit",
                "Bash",
                "Glob",
                "Grep",
                "WebFetch",
                "WebSearch",
            ]
        else:
            # Non-agentic: explicitly disable all tools
            allowed_tools = []

        if self.verbose and self.permission_mode == "bypassPermissions":
            print("[ClaudeCodeClient] WARNING: Running with bypassPermissions - all tool calls auto-approved")

        return ClaudeAgentOptions(
            model=self.model_name,
            max_turns=50 if self.agentic else 1,
            permission_mode=self.permission_mode,
            cwd=str(self.cwd),
            allowed_tools=allowed_tools,
            system_prompt=system_prompt or self._system_prompt,
        )

    async def _acompletion_impl(
        self,
        prompt: str | list[dict[str, Any]],
        model: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Implement async completion using claude-agent-sdk.

        Args:
            prompt: The prompt to send to Claude (string or message list)
            model: Override model (optional, uses self.model_name if not provided)
            **kwargs: Additional arguments (ignored for compatibility)

        Returns:
            The model's response text
        """
        # Convert prompt to string and extract system prompt
        user_prompt, system_prompt = self._prepare_prompt(prompt)
        options = self._build_options(system_prompt)

        # Override model if specified
        if model:
            options.model = model

        if self.verbose:
            print(f"[ClaudeCodeClient] Sending prompt ({len(user_prompt)} chars)")
            print(f"[ClaudeCodeClient] Model: {options.model}")
            print(f"[ClaudeCodeClient] Agentic: {self.agentic}")
            if system_prompt:
                print(f"[ClaudeCodeClient] System prompt: {len(system_prompt)} chars")

        response_parts: list[str] = []
        total_input_tokens = 0
        total_output_tokens = 0

        async for message in query(prompt=user_prompt, options=options):
            if self.verbose:
                print(f"[ClaudeCodeClient] Received {type(message).__name__}")

            if isinstance(message, AssistantMessage):
                # Extract text from content blocks
                if message.content:
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            response_parts.append(block.text)
                        elif isinstance(block, ThinkingBlock) and self.verbose:
                            thinking_preview = block.thinking[:100] if block.thinking else ""
                            print(f"[ClaudeCodeClient] Thinking: {thinking_preview}...")
                        elif isinstance(block, ToolUseBlock) and self.verbose:
                            print(f"[ClaudeCodeClient] Tool: {block.name}")
                        elif isinstance(block, ToolResultBlock) and self.verbose:
                            print(f"[ClaudeCodeClient] Tool result: {block.tool_use_id}")
            elif isinstance(message, ResultMessage):
                # Final result - extract usage info if available
                if message.usage:
                    total_input_tokens = getattr(message.usage, "input_tokens", 0)
                    total_output_tokens = getattr(message.usage, "output_tokens", 0)
                if message.is_error:
                    raise RuntimeError(f"Claude agent error: {message.result}")

        response = "\n".join(response_parts) if response_parts else ""

        # Update usage tracking
        self._total_calls += 1
        # Use actual tokens if available, otherwise estimate
        if total_input_tokens == 0:
            total_input_tokens = len(user_prompt) // 4
        if total_output_tokens == 0:
            total_output_tokens = len(response) // 4
        self._last_input_tokens = total_input_tokens
        self._last_output_tokens = total_output_tokens
        self._total_input_tokens += total_input_tokens
        self._total_output_tokens += total_output_tokens

        if self.verbose:
            print(f"[ClaudeCodeClient] Response ({len(response)} chars)")

        return response

    def _completion_impl(
        self,
        prompt: str | list[dict[str, Any]],
        model: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Implement sync completion by wrapping async version.

        Args:
            prompt: The prompt to send to Claude (string or message list)
            model: Override model (optional)
            **kwargs: Additional arguments

        Returns:
            The model's response text
        """
        return asyncio.run(self._acompletion_impl(prompt, model, **kwargs))

    def completion(
        self,
        prompt: str | list[dict[str, Any]],
        model: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Get a completion from Claude.

        Args:
            prompt: The prompt to send (string or message list)
            model: Override model (optional)
            **kwargs: Additional arguments

        Returns:
            The model's response
        """
        return self._completion_impl(prompt, model, **kwargs)

    async def acompletion(
        self,
        prompt: str | list[dict[str, Any]],
        model: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Get an async completion from Claude.

        Args:
            prompt: The prompt to send (string or message list)
            model: Override model (optional)
            **kwargs: Additional arguments

        Returns:
            The model's response
        """
        return await self._acompletion_impl(prompt, model, **kwargs)

    def get_last_usage(self) -> ModelUsageSummary:
        """Get usage statistics for the last API call.

        Returns:
            ModelUsageSummary with token counts for the last call
        """
        return ModelUsageSummary(
            total_calls=1 if self._last_input_tokens > 0 else 0,
            total_input_tokens=self._last_input_tokens,
            total_output_tokens=self._last_output_tokens,
        )

    def get_usage_summary(self) -> UsageSummary:
        """Get aggregated usage statistics across all API calls.

        Returns:
            UsageSummary with total token counts by model
        """
        return UsageSummary(
            model_usage_summaries={
                self.model_name: ModelUsageSummary(
                    total_calls=self._total_calls,
                    total_input_tokens=self._total_input_tokens,
                    total_output_tokens=self._total_output_tokens,
                )
            }
        )
