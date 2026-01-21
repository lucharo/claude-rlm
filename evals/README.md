# Evaluation Tasks

Tasks designed to showcase RLM's recursive decomposition strengths. These tasks involve large contexts that benefit from chunking and aggregation.

## Task Categories

### Summarization
- **book_summary**: Summarize a 500k+ word book into key themes and plot arcs

### Synthesis
- **paper_synthesis**: Synthesize findings from 20+ research papers into coherent themes

### Code Analysis
- **linux_kernel_docs**: Generate documentation for kernel subsystem
- **codebase_security**: Find security issues in a large codebase
- **error_patterns**: Find and categorize all error handling patterns

### Infrastructure
- **config_audit**: Audit 100+ config files for inconsistencies

### Documentation
- **changelog_gen**: Generate changelog from 1000+ commits
- **commit_analysis**: Analyze commit patterns for code quality insights

### Analysis
- **log_analysis**: Analyze large log files for patterns and anomalies

### Math (Simple, non-agentic)
- **prime_numbers**: Calculate first N prime numbers
- **fibonacci_analysis**: Analyze properties of Fibonacci sequence

## Why RLM?

RLM (Recursive Language Models) excel at tasks that:

1. **Require chunking**: Large documents/codebases that don't fit in context
2. **Benefit from decomposition**: Complex problems that can be broken into sub-problems
3. **Need aggregation**: Results from multiple sub-queries must be combined

### Example: Codebase Security Audit

Without RLM:
- Must fit entire codebase in context (impossible for large projects)
- Or manually split and lose cross-file analysis

With RLM:
1. RLM chunks codebase into manageable pieces
2. Each chunk analyzed via `llm_query()` sub-call
3. Results aggregated programmatically in REPL environment
4. Cross-references maintained through RLM's state

## Running Tasks

```bash
# List available tasks
python -m evals.tasks

# Run a specific task (example)
claude-rlm query "$(python -c "from evals.tasks import get_task; t=get_task('error_patterns'); print(t.prompt.format(codebase_path='/path/to/code'))")"

# Run with agentic mode for file access
claude-rlm --agentic query "Analyze error handling in /path/to/project"
```

## Cost Considerations

Using Haiku (fast/cheap) makes RLM economical for large tasks:

```bash
# Use Haiku for cost-effective analysis
claude-rlm --model claude-3-5-haiku-latest --agentic query \
  "Find all error handling patterns in ./src"
```

Estimated costs (rough):
- **book_summary**: ~$0.50-1.00 with Haiku
- **codebase_security**: ~$1-5 depending on size
- **changelog_gen**: ~$0.25-0.50 for 1000 commits
