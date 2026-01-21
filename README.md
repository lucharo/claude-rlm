# claude-rlm

Claude Code backend for [RLM](https://github.com/alexzhang13/rlm) (Recursive Language Models).

## Overview

`claude-rlm` provides:
- **ClaudeCodeClient**: RLM `BaseLM` implementation using `claude-agent-sdk`
- **CLI/REPL interface**: Interactive recursive reasoning sessions

```
┌─────────────────────────────────────────────────────┐
│                    claude-rlm CLI                    │
│         (prompt-toolkit REPL + rich output)         │
├─────────────────────────────────────────────────────┤
│                   claude_rlm.client                  │
│    (ClaudeCodeClient using claude-agent-sdk)        │
├─────────────────────────────────────────────────────┤
│                    RLM (upstream)                    │
│  (recursive reasoning, REPL env, sub-LM calls)      │
└─────────────────────────────────────────────────────┘
```

## Installation

```bash
# Using uv (recommended)
uv add "claude-rlm @ git+https://github.com/lucharo/claude-rlm.git"

# Or install from source
git clone https://github.com/lucharo/claude-rlm.git
cd claude-rlm
uv sync
```

## Usage

### REPL Mode (default)

```bash
claude-rlm
```

Starts an interactive session with history and rich output.

### One-shot Query

```bash
claude-rlm query "Calculate the first 20 prime numbers"
```

### CLI Options

```
Options:
  -m, --model TEXT          Claude model to use [default: claude-sonnet-4-20250514]
  -d, --max-depth INT       Maximum recursion depth [default: 5]
  -i, --max-iterations INT  Maximum REPL iterations [default: 100]
  -a, --agentic             Enable Claude tools (Read, Write, Bash, etc.)
  -C, --cwd PATH            Working directory [default: .]
  -v, --verbose             Enable verbose output
  -V, --version             Show version and exit
```

### Agentic Mode

Enable Claude's tools for file system access:

```bash
# Analyze a codebase
claude-rlm --agentic query "Find all error handling patterns in ./src"

# Summarize documents
claude-rlm --agentic query "Summarize all markdown files in ./docs"
```

## Why RLM?

RLM excels at tasks that:

1. **Require chunking**: Large documents/codebases that don't fit in context
2. **Benefit from decomposition**: Complex problems that can be broken into sub-problems
3. **Need aggregation**: Results from multiple sub-queries must be combined

### Example: Codebase Analysis with Haiku

Using Haiku (fast/cheap) makes RLM economical for large-scale analysis:

```bash
claude-rlm --model claude-3-5-haiku-latest --agentic query \
  "Analyze src/ directory: find all error handling patterns,
   categorize them, and identify inconsistencies"
```

**How RLM processes this:**
1. **Chunking**: RLM breaks the codebase into manageable pieces
2. **Sub-LM calls**: Each chunk analyzed with `llm_query()`
3. **Aggregation**: Results combined programmatically in REPL
4. **Cost-effective**: Haiku processes each chunk cheaply

## Evaluation Tasks

See `evals/` for tasks designed to showcase RLM's strengths:

- **book_summary**: Summarize 500k+ word books
- **linux_kernel_docs**: Document kernel subsystems
- **codebase_security**: Security audit large codebases
- **paper_synthesis**: Synthesize 20+ research papers
- **config_audit**: Audit 100+ config files
- **changelog_gen**: Generate changelogs from 1000+ commits

```bash
# List available tasks
python -m evals.tasks
```

## Development

```bash
# Clone and install
git clone https://github.com/lucharo/claude-rlm.git
cd claude-rlm
uv sync

# Run tests
uv run pytest

# Lint
uv run ruff check .
```

## License

MIT
