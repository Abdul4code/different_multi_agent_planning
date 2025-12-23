Centralized LLM Multi-Agent System (LangGraph compatible)

This repository contains a minimal, runnable Python implementation of a centralized multi-agent system using a LangGraph-style StateGraph. It uses a local OpenAI-compatible endpoint (Qwen2.5:7b) and strictly performs all environment interactions via real side-effect tools.

Setup

1. Create a virtualenv (optional):

   python3 -m venv env
   source env/bin/activate

2. Install requirements:

   pip install -r requirements.txt

3. Ensure your local OpenAI-compatible endpoint is running at:

   http://localhost:11434/v1

   The system expects model: qwen2.5:7b. Set environment variable OPENAI_API_BASE if needed.

Run

   python3 run_system.py --user "Fix failing tests"

Target a different repository

You can make the system operate on a different repository by passing `--repo` (or `-r`). The runner will chdir into the provided directory before executing agents, so all file and test operations will happen inside that repo.

Example:

```bash
python3 run_system.py --repo /path/to/target/repo --user "Fix failing tests"
```

Control maximum planner iterations

You can control how many planner iterations (analyze/write/test/debug cycles) the system will perform before stopping using `--max-iterations` (or `-m`). Default is 8.

Example:

```bash
python3 run_system.py --repo /path/to/target/repo --user "Fix failing tests" --max-iterations 20
```

Notes

- PlannerAgent is the only agent that decides control flow.
- Worker agents call only the provided tools which perform real filesystem and subprocess operations.
- The system writes a `CHANGELOG_AGENT.md` file describing the mode and modifications.
