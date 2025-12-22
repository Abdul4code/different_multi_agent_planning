"""A minimal LangGraph-like orchestrator.

This file implements a tiny StateGraph and Node primitives to orchestrate the
centralized workflow described in the task. If the real `langgraph` package
is available, this code can be swapped out; however the implementation here
is self-contained so the agent can run in the user's environment.
"""
from typing import Any, Callable, Dict, List, Optional
import traceback


class Node:
    """Base node: operate on shared state and return updated state."""

    def __init__(self, name: str):
        self.name = name

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError()


class StateGraph:
    """Simple directed graph runner with explicit edges.

    Nodes are added and edges list the next node(s) to call. This orchestrator
    runs synchronously and keeps an iteration count in the shared state.
    Termination is controlled by nodes updating state['terminate'] = True.
    """

    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[str, List[str]] = {}

    def add_node(self, node: Node) -> None:
        self.nodes[node.name] = node

    def add_edge(self, src: str, dst: str) -> None:
        self.edges.setdefault(src, []).append(dst)

    def run(self, initial_state: Optional[Dict[str, Any]] = None, start: Optional[str] = None, max_iterations: int = 20) -> Dict[str, Any]:
        state = initial_state or {}
        state.setdefault("iteration", 0)
        state.setdefault("terminate", False)

        current = start
        if current is None:
            # Pick a default start node if available
            current = next(iter(self.nodes)) if self.nodes else None

        while not state.get("terminate", False) and state["iteration"] < max_iterations and current:
            node = self.nodes.get(current)
            if node is None:
                break
            try:
                state = node.run(state) or state
            except Exception:
                state.setdefault("errors", []).append(traceback.format_exc())
            state["iteration"] = state.get("iteration", 0) + 1
            # Move to next node(s) according to edges; we only follow the first
            next_nodes = self.edges.get(current, [])
            current = next_nodes[0] if next_nodes else None

        return state
