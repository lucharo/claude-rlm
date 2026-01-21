"""Claude Code backend for RLM (Recursive Language Models).

This package provides:
- ClaudeCodeClient: RLM BaseLM implementation using claude-agent-sdk
- CLI/REPL interface for interactive recursive reasoning
"""

from claude_rlm.client import ClaudeCodeClient

__version__ = "0.1.0"
__all__ = ["ClaudeCodeClient"]
