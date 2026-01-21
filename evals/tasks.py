"""Evaluation tasks for claude-rlm.

These tasks are designed to showcase RLM's recursive decomposition strengths.
They involve large contexts that benefit from chunking and aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class EvalTask:
    """An evaluation task for RLM."""

    id: str
    name: str
    description: str
    prompt: str
    input_size: str
    expected_chunks: str
    category: str
    difficulty: str  # easy, medium, hard
    requires_agentic: bool = False
    sample_input: str | None = None


EVAL_TASKS: list[EvalTask] = [
    # Document summarization tasks
    EvalTask(
        id="book_summary",
        name="Long Book Summarization",
        description="Summarize a 500k+ word book into key themes, characters, and plot arcs",
        prompt=(
            "Read and summarize the book at {book_path}. Identify:\n"
            "1. Main themes and motifs\n"
            "2. Key characters and their development\n"
            "3. Major plot arcs and turning points\n"
            "4. Writing style and narrative techniques\n"
            "Provide a comprehensive summary suitable for study purposes."
        ),
        input_size="~500k words",
        expected_chunks="50-100 chapters",
        category="summarization",
        difficulty="hard",
        requires_agentic=True,
        sample_input="/path/to/war_and_peace.txt",
    ),
    EvalTask(
        id="paper_synthesis",
        name="Research Paper Synthesis",
        description="Synthesize findings from 20+ research papers into coherent themes",
        prompt=(
            "Analyze the research papers in {papers_dir}. For each paper:\n"
            "1. Extract key findings and methodology\n"
            "2. Identify the main contributions\n"
            "3. Note limitations and future work suggestions\n\n"
            "Then synthesize across all papers:\n"
            "- Common themes and consensus findings\n"
            "- Areas of disagreement or contradiction\n"
            "- Research gaps and opportunities\n"
            "- Recommended next steps for the field"
        ),
        input_size="20+ papers, ~200 pages total",
        expected_chunks="20-30 documents",
        category="synthesis",
        difficulty="hard",
        requires_agentic=True,
        sample_input="/path/to/papers/",
    ),
    # Code analysis tasks
    EvalTask(
        id="linux_kernel_docs",
        name="Linux Kernel Documentation",
        description="Generate documentation/commentary for a kernel subsystem",
        prompt=(
            "Analyze the Linux kernel source code at {kernel_path}/mm/ "
            "(memory management subsystem). For each file:\n"
            "1. Identify the main purpose and functionality\n"
            "2. Document key functions and their roles\n"
            "3. Explain data structures used\n"
            "4. Note interactions with other subsystems\n\n"
            "Produce comprehensive documentation suitable for kernel developers."
        ),
        input_size="~100 files, ~50k LOC",
        expected_chunks="100+ files",
        category="code_analysis",
        difficulty="hard",
        requires_agentic=True,
        sample_input="/usr/src/linux/",
    ),
    EvalTask(
        id="codebase_security",
        name="Codebase Security Audit",
        description="Find security issues in a large codebase",
        prompt=(
            "Perform a security audit of the codebase at {codebase_path}. "
            "Scan for:\n"
            "1. Input validation vulnerabilities (SQL injection, XSS, etc.)\n"
            "2. Authentication and authorization issues\n"
            "3. Sensitive data exposure\n"
            "4. Insecure cryptography usage\n"
            "5. Dependency vulnerabilities\n"
            "6. Configuration security issues\n\n"
            "For each finding, provide:\n"
            "- Severity (Critical/High/Medium/Low)\n"
            "- Location (file and line)\n"
            "- Description of the vulnerability\n"
            "- Recommended fix"
        ),
        input_size="Variable, 10k-100k LOC",
        expected_chunks="Varies by codebase",
        category="security",
        difficulty="medium",
        requires_agentic=True,
        sample_input="/path/to/codebase/",
    ),
    EvalTask(
        id="error_patterns",
        name="Error Handling Pattern Analysis",
        description="Find and categorize all error handling patterns in a codebase",
        prompt=(
            "Analyze the codebase at {codebase_path} for error handling patterns. "
            "Identify:\n"
            "1. All error handling approaches used (try/catch, Result types, etc.)\n"
            "2. Consistency issues between different parts of the code\n"
            "3. Missing error handling (uncaught exceptions, ignored errors)\n"
            "4. Error message quality and helpfulness\n"
            "5. Logging practices for errors\n\n"
            "Provide recommendations for standardizing error handling."
        ),
        input_size="Variable",
        expected_chunks="Varies by codebase",
        category="code_analysis",
        difficulty="medium",
        requires_agentic=True,
        sample_input="/path/to/project/src/",
    ),
    # Configuration and infrastructure tasks
    EvalTask(
        id="config_audit",
        name="Configuration File Audit",
        description="Audit 100+ config files for inconsistencies and best practices",
        prompt=(
            "Audit all configuration files in {config_dir}. Check for:\n"
            "1. Inconsistencies between environments (dev/staging/prod)\n"
            "2. Hardcoded secrets or credentials\n"
            "3. Missing required settings\n"
            "4. Deprecated or unused settings\n"
            "5. Security misconfigurations\n"
            "6. Performance-related settings\n\n"
            "Compare each file against baseline best practices "
            "and provide a prioritized list of issues to fix."
        ),
        input_size="100+ config files",
        expected_chunks="100+ files",
        category="infrastructure",
        difficulty="medium",
        requires_agentic=True,
        sample_input="/path/to/configs/",
    ),
    # Git/history analysis tasks
    EvalTask(
        id="changelog_gen",
        name="Changelog Generation",
        description="Generate changelog from 1000+ commits",
        prompt=(
            "Analyze the git history of the repository at {repo_path}. "
            "Generate a changelog that:\n"
            "1. Groups commits by category (features, fixes, refactoring, etc.)\n"
            "2. Identifies breaking changes\n"
            "3. Highlights significant improvements\n"
            "4. Credits contributors appropriately\n"
            "5. Notes deprecated features\n\n"
            "Format the output as a markdown changelog suitable for release notes."
        ),
        input_size="1000+ commits",
        expected_chunks="Batches of 50-100 commits",
        category="documentation",
        difficulty="medium",
        requires_agentic=True,
        sample_input="/path/to/repo/",
    ),
    EvalTask(
        id="commit_analysis",
        name="Commit Pattern Analysis",
        description="Analyze commit patterns for code quality insights",
        prompt=(
            "Analyze the commit history of {repo_path} to identify:\n"
            "1. Hotspot files (frequently modified, likely bugs)\n"
            "2. Coupling between files (often changed together)\n"
            "3. Commit quality trends over time\n"
            "4. Areas with high churn (potential refactoring candidates)\n"
            "5. Bus factor (who knows what code)\n\n"
            "Provide actionable recommendations for improving code health."
        ),
        input_size="Full git history",
        expected_chunks="Time periods or commit batches",
        category="code_analysis",
        difficulty="medium",
        requires_agentic=True,
        sample_input="/path/to/repo/",
    ),
    # Data processing tasks
    EvalTask(
        id="log_analysis",
        name="Log File Analysis",
        description="Analyze large log files for patterns and anomalies",
        prompt=(
            "Analyze the log files in {logs_dir}. Identify:\n"
            "1. Error patterns and their frequency\n"
            "2. Performance anomalies (slow requests, timeouts)\n"
            "3. Security-related events (failed logins, suspicious activity)\n"
            "4. Resource usage patterns\n"
            "5. Correlation between different types of events\n\n"
            "Provide a summary report with key findings and recommendations."
        ),
        input_size="GB of logs",
        expected_chunks="Time windows or log types",
        category="analysis",
        difficulty="hard",
        requires_agentic=True,
        sample_input="/var/log/myapp/",
    ),
    # Simple test tasks (non-agentic)
    EvalTask(
        id="prime_numbers",
        name="Prime Number Calculation",
        description="Calculate first N prime numbers using recursive decomposition",
        prompt=(
            "Calculate the first {n} prime numbers. "
            "Show your work by explaining the algorithm and listing each prime."
        ),
        input_size="Single number",
        expected_chunks="1 (no chunking needed)",
        category="math",
        difficulty="easy",
        requires_agentic=False,
        sample_input="100",
    ),
    EvalTask(
        id="fibonacci_analysis",
        name="Fibonacci Sequence Analysis",
        description="Analyze properties of Fibonacci sequence",
        prompt=(
            "Analyze the Fibonacci sequence up to the {n}th term. Identify:\n"
            "1. The golden ratio approximation at various points\n"
            "2. Divisibility patterns\n"
            "3. Digit sum patterns\n"
            "4. Any other interesting mathematical properties"
        ),
        input_size="Single number",
        expected_chunks="1",
        category="math",
        difficulty="easy",
        requires_agentic=False,
        sample_input="50",
    ),
]


def get_task(task_id: str) -> EvalTask | None:
    """Get a task by ID."""
    for task in EVAL_TASKS:
        if task.id == task_id:
            return task
    return None


def list_tasks(category: str | None = None, agentic_only: bool = False) -> list[EvalTask]:
    """List tasks, optionally filtered by category or agentic requirement."""
    tasks = EVAL_TASKS
    if category:
        tasks = [t for t in tasks if t.category == category]
    if agentic_only:
        tasks = [t for t in tasks if t.requires_agentic]
    return tasks


def get_categories() -> list[str]:
    """Get all unique task categories."""
    return sorted(set(t.category for t in EVAL_TASKS))


if __name__ == "__main__":
    # Print all tasks for reference
    print("Available Evaluation Tasks:\n")
    for task in EVAL_TASKS:
        agentic_badge = " [AGENTIC]" if task.requires_agentic else ""
        print(f"  {task.id}: {task.name}{agentic_badge}")
        print(f"    {task.description}")
        print(f"    Category: {task.category} | Difficulty: {task.difficulty}")
        print(f"    Input size: {task.input_size} | Expected chunks: {task.expected_chunks}")
        print()
