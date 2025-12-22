"""Centralized multi-agent runtime using LangGraph-like primitives.

This package provides a small orchestration runtime and nodes that implement
the Planner, Analyzer, Writer, Tester and Debugger nodes. It uses a local
OpenAI-compatible LLM endpoint at http://localhost:11434/v1 and the model
name qwen2.5:7b by default.
"""

from .graph import StateGraph
from .nodes import (
    PlannerNode,
    CodeAnalyzerNode,
    CodeWriterNode,
    TestRunnerNode,
    DebuggerNode,
)

__all__ = [
    "StateGraph",
    "PlannerNode",
    "CodeAnalyzerNode",
    "CodeWriterNode",
    "TestRunnerNode",
    "DebuggerNode",
]
