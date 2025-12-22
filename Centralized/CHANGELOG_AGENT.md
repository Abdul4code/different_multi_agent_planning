CHANGELOG_AGENT
===============

Description
-----------
Added a centralized multi-agent orchestration package `agent/` that implements a small
LangGraph-like runtime and the following nodes: PlannerNode, CodeAnalyzerNode,
CodeWriterNode, TestRunnerNode, and DebuggerNode. The system is designed to use a
local OpenAI-compatible LLM endpoint at http://localhost:11434/v1 with model
name `qwen2.5:7b`.

Files modified/added
--------------------
- agent/__init__.py — package initializer and exports
- agent/graph.py — minimal StateGraph and Node base classes
- agent/llm_client.py — local LLM client for OpenAI-compatible endpoint
- agent/nodes.py — implementations: PlannerNode, CodeAnalyzerNode, CodeWriterNode, TestRunnerNode, DebuggerNode
- run_agent.py — entrypoint to wire and execute the graph
- CHANGELOG_AGENT.md — this file

Key changes and notes
---------------------
- The runtime is intentionally self-contained and does not require an installed
  `langgraph` package. It provides small `StateGraph` and `Node` primitives that
  mirror the behavior required by the task.
- The PlannerNode asks the LLM to produce a JSON plan. The DebuggerNode asks the
  LLM to propose fixes in JSON mapping file paths to full new file contents.
- The CodeWriterNode writes full file contents to disk. Be careful: the LLM
  must propose safe, minimal changes. Always review changes produced by the
  agent before committing to source control.
- The TestRunnerNode runs pytest using the repository Python executable and
  collects structured results. The system terminates when tests pass or the
  iteration limit is reached.

How to run
----------
Ensure your local LLM is running and listening at:

http://localhost:11434/v1

Then run:

python run_agent.py --prompt "Fix failing tests and make tests pass"

Limitations and next steps
--------------------------
- The LLM client uses a plain HTTP POST to `/chat/completions`. If your local
  server uses a different path or expects headers, adjust `agent/llm_client.py`.
- The current DebuggerNode expects the LLM to return a JSON object with a
  top-level `files` mapping. If the LLM returns non-JSON text, that output is
  recorded to the state and the run terminates.
- Optional: integrate with git to produce diffs, and add safety checks before
  writing files. Add unit tests for the agent nodes themselves.
