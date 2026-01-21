"""CLI and REPL interface for claude-rlm."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from claude_rlm.client import ClaudeCodeClient, SessionMetrics

app = typer.Typer(
    name="claude-rlm",
    help="Claude Code backend for RLM (Recursive Language Models)",
    add_completion=False,
)
console = Console()


_rlm_patched = False
_active_clients: list[ClaudeCodeClient] = []  # Track created clients for metrics


def _patch_rlm_clients():
    """Monkey-patch RLM to support claude-code backend."""
    global _rlm_patched
    if _rlm_patched:
        return
    _rlm_patched = True

    # Patch at the source module level
    import rlm.clients

    original_get_client = rlm.clients.get_client

    def patched_get_client(backend, backend_kwargs):
        if backend == "claude-code":
            client = ClaudeCodeClient(**backend_kwargs)
            _active_clients.append(client)
            return client
        return original_get_client(backend, backend_kwargs)

    # Patch the function in the module
    rlm.clients.get_client = patched_get_client

    # Also patch in rlm.core.rlm if it has already imported get_client
    try:
        import rlm.core.rlm as rlm_core
        if hasattr(rlm_core, "get_client"):
            rlm_core.get_client = patched_get_client
    except (ImportError, AttributeError):
        pass


def get_combined_metrics() -> SessionMetrics:
    """Combine metrics from all active clients."""
    combined = SessionMetrics()
    for client in _active_clients:
        client_metrics = client.get_metrics()
        for call in client_metrics.calls:
            combined.add_call(call)
    return combined


def clear_clients():
    """Clear active clients list."""
    _active_clients.clear()


def print_metrics_panel():
    """Print a rich-formatted metrics panel."""
    metrics = get_combined_metrics()
    if metrics.total_api_calls == 0:
        return

    # Build metrics table
    lines = [
        f"[cyan]API Calls:[/cyan]        {metrics.total_api_calls}",
        f"[cyan]Input Tokens:[/cyan]     {metrics.total_input_tokens:,}",
        f"[cyan]Output Tokens:[/cyan]    {metrics.total_output_tokens:,}",
        f"[cyan]Total Tokens:[/cyan]     {metrics.total_input_tokens + metrics.total_output_tokens:,}",
        f"[cyan]API Time:[/cyan]         {metrics.total_duration_ms / 1000:.2f}s",
        f"[cyan]Tool Uses:[/cyan]        {metrics.total_tool_uses}",
    ]

    if metrics.tool_usage_counts:
        tool_str = ", ".join(f"{t}:{c}" for t, c in sorted(metrics.tool_usage_counts.items(), key=lambda x: -x[1]))
        lines.append(f"[cyan]Tools:[/cyan]            {tool_str}")

    console.print(Panel("\n".join(lines), title="Usage Metrics", border_style="yellow"))


# Apply patch immediately at import time
_patch_rlm_clients()


def get_rlm(
    model: str,
    max_depth: int,
    max_iterations: int,
    agentic: bool,
    cwd: Path,
    verbose: bool,
):
    """Create an RLM instance with ClaudeCodeClient backend.

    Args:
        model: Claude model to use
        max_depth: Maximum recursion depth
        max_iterations: Maximum REPL iterations
        agentic: Whether to enable Claude's tools
        cwd: Working directory
        verbose: Enable verbose output

    Returns:
        Configured RLM instance
    """
    from rlm import RLM

    return RLM(
        backend="claude-code",
        backend_kwargs={
            "model_name": model,
            "agentic": agentic,
            "cwd": cwd,
            "verbose": verbose,
        },
        max_depth=max_depth,
        max_iterations=max_iterations,
    )


def run_query(
    prompt: str,
    model: str,
    max_depth: int,
    max_iterations: int,
    agentic: bool,
    cwd: Path,
    verbose: bool,
    show_metrics: bool = True,
) -> str:
    """Execute a single query through RLM.

    Args:
        prompt: The prompt to process
        model: Claude model to use
        max_depth: Maximum recursion depth
        max_iterations: Maximum REPL iterations
        agentic: Whether to enable Claude's tools
        cwd: Working directory
        verbose: Enable verbose output
        show_metrics: Whether to print metrics after completion

    Returns:
        The RLM response
    """
    # Clear previous clients to get fresh metrics
    clear_clients()

    rlm = get_rlm(model, max_depth, max_iterations, agentic, cwd, verbose)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Processing with RLM...", total=None)
        result = rlm.completion(prompt)

    # Print metrics
    if show_metrics:
        print_metrics_panel()

    return result.response


@app.command("query")
def query_cmd(
    prompt: Annotated[str, typer.Argument(help="The prompt to process")],
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Claude model to use"),
    ] = "claude-sonnet-4-20250514",
    max_depth: Annotated[
        int,
        typer.Option("--max-depth", "-d", help="Maximum recursion depth"),
    ] = 5,
    max_iterations: Annotated[
        int,
        typer.Option("--max-iterations", "-i", help="Maximum REPL iterations"),
    ] = 100,
    agentic: Annotated[
        bool,
        typer.Option("--agentic", "-a", help="Enable Claude tools (Read, Write, Bash, etc.)"),
    ] = False,
    cwd: Annotated[
        Path,
        typer.Option("--cwd", "-C", help="Working directory"),
    ] = Path("."),
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output"),
    ] = False,
    no_metrics: Annotated[
        bool,
        typer.Option("--no-metrics", help="Suppress usage metrics output"),
    ] = False,
) -> None:
    """Execute a single query through RLM."""
    try:
        result = run_query(
            prompt, model, max_depth, max_iterations, agentic, cwd.resolve(), verbose,
            show_metrics=not no_metrics
        )
        console.print(Panel(Markdown(result), title="Result", border_style="green"))
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from e


@app.command("repl")
def repl_cmd(
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Claude model to use"),
    ] = "claude-sonnet-4-20250514",
    max_depth: Annotated[
        int,
        typer.Option("--max-depth", "-d", help="Maximum recursion depth"),
    ] = 5,
    max_iterations: Annotated[
        int,
        typer.Option("--max-iterations", "-i", help="Maximum REPL iterations"),
    ] = 100,
    agentic: Annotated[
        bool,
        typer.Option("--agentic", "-a", help="Enable Claude tools (Read, Write, Bash, etc.)"),
    ] = False,
    cwd: Annotated[
        Path,
        typer.Option("--cwd", "-C", help="Working directory"),
    ] = Path("."),
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output"),
    ] = False,
) -> None:
    """Start an interactive REPL session."""
    console.print(
        Panel(
            "[bold blue]claude-rlm[/bold blue] - Recursive Language Model with Claude Code\n\n"
            f"Model: [cyan]{model}[/cyan]\n"
            f"Max depth: [cyan]{max_depth}[/cyan]\n"
            f"Max iterations: [cyan]{max_iterations}[/cyan]\n"
            f"Agentic: [cyan]{agentic}[/cyan]\n"
            f"Working directory: [cyan]{cwd.resolve()}[/cyan]\n\n"
            "Type [bold]exit[/bold] or [bold]quit[/bold] to exit.\n"
            "Type [bold]help[/bold] for more information.",
            title="Welcome",
            border_style="blue",
        )
    )

    # Set up history file
    history_file = Path.home() / ".claude-rlm_history"
    session: PromptSession = PromptSession(history=FileHistory(str(history_file)))

    cwd_resolved = cwd.resolve()

    while True:
        try:
            prompt = session.prompt("\n[claude-rlm] > ")
            prompt = prompt.strip()

            if not prompt:
                continue

            if prompt.lower() in ("exit", "quit"):
                console.print("[yellow]Goodbye![/yellow]")
                break

            if prompt.lower() == "help":
                console.print(
                    Panel(
                        "Commands:\n"
                        "  [bold]exit[/bold], [bold]quit[/bold] - Exit the REPL\n"
                        "  [bold]help[/bold] - Show this help message\n\n"
                        "Enter any prompt to process it through RLM.\n"
                        "RLM will recursively decompose complex tasks.",
                        title="Help",
                        border_style="cyan",
                    )
                )
                continue

            result = run_query(
                prompt, model, max_depth, max_iterations, agentic, cwd_resolved, verbose,
                show_metrics=True
            )
            console.print(Panel(Markdown(result), title="Result", border_style="green"))

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Type 'exit' to quit.[/yellow]")
        except EOFError:
            console.print("\n[yellow]Goodbye![/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: Annotated[
        bool,
        typer.Option("--version", "-V", help="Show version and exit"),
    ] = False,
) -> None:
    """Claude Code backend for RLM (Recursive Language Models).

    Run without arguments to start the interactive REPL.
    Use 'claude-rlm query "prompt"' for one-shot queries.
    """
    if version:
        from claude_rlm import __version__

        console.print(f"claude-rlm version {__version__}")
        raise typer.Exit()

    # If no subcommand provided, run the REPL
    if ctx.invoked_subcommand is None:
        ctx.invoke(repl_cmd)


if __name__ == "__main__":
    app()
