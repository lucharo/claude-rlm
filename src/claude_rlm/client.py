"""ClaudeCodeClient - RLM BaseLM implementation using claude-agent-sdk."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions, query
from rlm.clients.base_lm import BaseLM


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
    """

    model_name: str = "claude-sonnet-4-20250514"
    agentic: bool = False
    cwd: Path | None = None
    max_tokens: int = 16384
    verbose: bool = False
    _system_prompt: str | None = field(default=None, repr=False)

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

    def _build_options(self) -> ClaudeAgentOptions:
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
            # Non-agentic: only allow basic response
            allowed_tools = []

        return ClaudeAgentOptions(
            model=self.model_name,
            max_turns=50 if self.agentic else 1,
            max_tokens=self.max_tokens,
            permission_mode="bypassPermissions",  # Run without permission prompts
            cwd=str(self.cwd),
            allowed_tools=allowed_tools if allowed_tools else None,
            system_prompt=self._system_prompt,
        )

    async def _acompletion_impl(
        self,
        prompt: str,
        model: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Implement async completion using claude-agent-sdk.

        Args:
            prompt: The prompt to send to Claude
            model: Override model (optional, uses self.model_name if not provided)
            **kwargs: Additional arguments (ignored for compatibility)

        Returns:
            The model's response text
        """
        options = self._build_options()

        # Override model if specified
        if model:
            options.model = model

        if self.verbose:
            print(f"[ClaudeCodeClient] Sending prompt ({len(prompt)} chars)")
            print(f"[ClaudeCodeClient] Model: {options.model}")
            print(f"[ClaudeCodeClient] Agentic: {self.agentic}")

        response_parts: list[str] = []

        async for message in query(prompt=prompt, options=options):
            # Handle different message types from claude-agent-sdk
            if hasattr(message, "type"):
                if message.type == "text":
                    response_parts.append(message.content)
                elif message.type == "result":
                    # Final result message
                    if hasattr(message, "text"):
                        response_parts.append(message.text)
                elif message.type == "error":
                    raise RuntimeError(f"Claude agent error: {message.content}")

        response = "".join(response_parts)

        if self.verbose:
            print(f"[ClaudeCodeClient] Received response ({len(response)} chars)")

        return response

    def _completion_impl(
        self,
        prompt: str,
        model: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Implement sync completion by wrapping async version.

        Args:
            prompt: The prompt to send to Claude
            model: Override model (optional)
            **kwargs: Additional arguments

        Returns:
            The model's response text
        """
        return asyncio.run(self._acompletion_impl(prompt, model, **kwargs))

    def completion(
        self,
        prompt: str,
        model: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Get a completion from Claude.

        Args:
            prompt: The prompt to send
            model: Override model (optional)
            **kwargs: Additional arguments

        Returns:
            The model's response
        """
        return self._completion_impl(prompt, model, **kwargs)

    async def acompletion(
        self,
        prompt: str,
        model: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Get an async completion from Claude.

        Args:
            prompt: The prompt to send
            model: Override model (optional)
            **kwargs: Additional arguments

        Returns:
            The model's response
        """
        return await self._acompletion_impl(prompt, model, **kwargs)
