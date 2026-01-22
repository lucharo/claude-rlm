"""ClaudeCodeClient - RLM BaseLM implementation using claude-agent-sdk."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

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
from rlm.clients.base_lm import BaseLM
from rlm.core.types import ModelUsageSummary, UsageSummary

PermissionMode = Literal["default", "acceptEdits", "plan", "bypassPermissions"]

# Available tools that can be enabled
ALL_TOOLS = [
    "Read",
    "Write",
    "Edit",
    "Bash",
    "Glob",
    "Grep",
    "WebFetch",
    "WebSearch",
    "Task",
    "TodoWrite",
    "NotebookEdit",
]

# Commonly used tool bundles
TOOL_BUNDLES = {
    "read-only": ["Read", "Glob", "Grep"],
    "file-ops": ["Read", "Write", "Edit", "Glob", "Grep"],
    "web": ["WebFetch", "WebSearch"],
    "all": ALL_TOOLS,
}


@dataclass
class CallMetrics:
    """Metrics for a single API call."""

    input_tokens: int = 0
    output_tokens: int = 0
    input_chars: int = 0
    output_chars: int = 0
    duration_ms: float = 0
    tool_uses: int = 0
    tool_names: list[str] = field(default_factory=list)
    thinking_blocks: int = 0
    text_blocks: int = 0
    model: str = ""


@dataclass
class SessionMetrics:
    """Cumulative metrics for an RLM session."""

    total_api_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_input_chars: int = 0
    total_output_chars: int = 0
    total_duration_ms: float = 0
    total_tool_uses: int = 0
    total_thinking_blocks: int = 0
    total_text_blocks: int = 0
    tool_usage_counts: dict[str, int] = field(default_factory=dict)
    calls: list[CallMetrics] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)

    def add_call(self, call: CallMetrics) -> None:
        """Add a call's metrics to the session totals."""
        self.total_api_calls += 1
        self.total_input_tokens += call.input_tokens
        self.total_output_tokens += call.output_tokens
        self.total_input_chars += call.input_chars
        self.total_output_chars += call.output_chars
        self.total_duration_ms += call.duration_ms
        self.total_tool_uses += call.tool_uses
        self.total_thinking_blocks += call.thinking_blocks
        self.total_text_blocks += call.text_blocks
        for tool in call.tool_names:
            self.tool_usage_counts[tool] = self.tool_usage_counts.get(tool, 0) + 1
        self.calls.append(call)

    def summary(self) -> str:
        """Return a formatted summary of session metrics."""
        elapsed = time.time() - self.start_time
        lines = [
            "═" * 50,
            "SESSION METRICS",
            "═" * 50,
            f"API Calls:        {self.total_api_calls}",
            f"Input Tokens:     {self.total_input_tokens:,}",
            f"Output Tokens:    {self.total_output_tokens:,}",
            f"Total Tokens:     {self.total_input_tokens + self.total_output_tokens:,}",
            f"Input Chars:      {self.total_input_chars:,}",
            f"Output Chars:     {self.total_output_chars:,}",
            f"API Time:         {self.total_duration_ms / 1000:.2f}s",
            f"Wall Time:        {elapsed:.2f}s",
            f"Tool Uses:        {self.total_tool_uses}",
            f"Thinking Blocks:  {self.total_thinking_blocks}",
            f"Text Blocks:      {self.total_text_blocks}",
        ]
        if self.tool_usage_counts:
            lines.append("─" * 50)
            lines.append("Tool Breakdown:")
            for tool, count in sorted(self.tool_usage_counts.items(), key=lambda x: -x[1]):
                lines.append(f"  {tool}: {count}")
        lines.append("═" * 50)
        return "\n".join(lines)


@dataclass
class ClaudeCodeClient(BaseLM):
    """RLM client using Claude Code via claude-agent-sdk.

    This client implements RLM's BaseLM interface, allowing Claude Code
    to be used as the backend for recursive language model operations.

    Attributes:
        model_name: The Claude model to use (default: claude-sonnet-4-20250514)
        allowed_tools: List of tools to enable, or None for no tools.
            Available: Read, Write, Edit, Bash, Glob, Grep, WebFetch, WebSearch, Task, etc.
            Use ALL_TOOLS constant for all tools, or TOOL_BUNDLES for presets.
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
    allowed_tools: list[str] | None = None  # None = no tools, list = specific tools
    cwd: Path | None = None
    max_tokens: int = 16384
    verbose: bool = False
    permission_mode: PermissionMode = "bypassPermissions"
    _system_prompt: str | None = field(default=None, repr=False)

    # Usage tracking (legacy)
    _total_calls: int = field(default=0, repr=False)
    _total_input_tokens: int = field(default=0, repr=False)
    _total_output_tokens: int = field(default=0, repr=False)
    _last_input_tokens: int = field(default=0, repr=False)
    _last_output_tokens: int = field(default=0, repr=False)

    # Detailed metrics tracking
    _metrics: SessionMetrics = field(default_factory=SessionMetrics, repr=False)
    _last_call_metrics: CallMetrics | None = field(default=None, repr=False)

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
        # Use provided tools or empty list
        tools = list(self.allowed_tools) if self.allowed_tools else []
        has_tools = len(tools) > 0

        if self.verbose:
            if tools:
                print(f"[ClaudeCodeClient] Tools enabled: {', '.join(tools)}")
            else:
                print("[ClaudeCodeClient] No tools enabled")
            if self.permission_mode == "bypassPermissions" and has_tools:
                print("[ClaudeCodeClient] WARNING: bypassPermissions - all tool calls auto-approved")

        return ClaudeAgentOptions(
            model=self.model_name,
            max_turns=50 if has_tools else 1,
            permission_mode=self.permission_mode,
            cwd=str(self.cwd),
            allowed_tools=tools if tools else None,
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
            if system_prompt:
                print(f"[ClaudeCodeClient] System prompt: {len(system_prompt)} chars")

        # Initialize call metrics
        call_start = time.time()
        call_metrics = CallMetrics(
            input_chars=len(user_prompt) + (len(system_prompt) if system_prompt else 0),
            model=options.model or self.model_name,
        )

        response_parts: list[str] = []
        total_input_tokens = 0
        total_output_tokens = 0
        duration_ms = 0

        async for message in query(prompt=user_prompt, options=options):
            if self.verbose:
                print(f"[ClaudeCodeClient] Received {type(message).__name__}")

            if isinstance(message, AssistantMessage):
                # Extract text from content blocks
                if message.content:
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            response_parts.append(block.text)
                            call_metrics.text_blocks += 1
                        elif isinstance(block, ThinkingBlock):
                            call_metrics.thinking_blocks += 1
                            if self.verbose:
                                thinking_preview = block.thinking[:100] if block.thinking else ""
                                print(f"[ClaudeCodeClient] Thinking: {thinking_preview}...")
                        elif isinstance(block, ToolUseBlock):
                            call_metrics.tool_uses += 1
                            call_metrics.tool_names.append(block.name)
                            if self.verbose:
                                print(f"[ClaudeCodeClient] Tool: {block.name}")
                        elif isinstance(block, ToolResultBlock) and self.verbose:
                            print(f"[ClaudeCodeClient] Tool result: {block.tool_use_id}")
            elif isinstance(message, ResultMessage):
                # Final result - extract usage info if available
                if message.usage:
                    total_input_tokens = getattr(message.usage, "input_tokens", 0)
                    total_output_tokens = getattr(message.usage, "output_tokens", 0)
                duration_ms = getattr(message, "duration_ms", 0) or 0
                if message.is_error:
                    raise RuntimeError(f"Claude agent error: {message.result}")

        response = "\n".join(response_parts) if response_parts else ""

        # Finalize call metrics
        call_metrics.output_chars = len(response)
        call_metrics.duration_ms = duration_ms or ((time.time() - call_start) * 1000)
        # Use actual tokens if available, otherwise estimate
        call_metrics.input_tokens = total_input_tokens or (call_metrics.input_chars // 4)
        call_metrics.output_tokens = total_output_tokens or (call_metrics.output_chars // 4)

        # Update session metrics
        self._metrics.add_call(call_metrics)
        self._last_call_metrics = call_metrics

        # Update legacy usage tracking
        self._total_calls += 1
        self._last_input_tokens = call_metrics.input_tokens
        self._last_output_tokens = call_metrics.output_tokens
        self._total_input_tokens += call_metrics.input_tokens
        self._total_output_tokens += call_metrics.output_tokens

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

    def get_metrics(self) -> SessionMetrics:
        """Get detailed session metrics.

        Returns:
            SessionMetrics with comprehensive usage data
        """
        return self._metrics

    def get_last_call_metrics(self) -> CallMetrics | None:
        """Get metrics for the last API call.

        Returns:
            CallMetrics for the last call, or None if no calls made
        """
        return self._last_call_metrics

    def reset_metrics(self) -> None:
        """Reset all metrics to start fresh."""
        self._metrics = SessionMetrics()
        self._last_call_metrics = None
        self._total_calls = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._last_input_tokens = 0
        self._last_output_tokens = 0

    def print_metrics(self) -> None:
        """Print a formatted summary of session metrics."""
        print(self._metrics.summary())
