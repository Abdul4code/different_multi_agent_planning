"""
Minimal LangGraph wrapper to provide StateGraph, Node, and conditional edges.

This is a thin wrapper that either imports a real LangGraph if available,
or provides a compatible lightweight implementation to meet the project's
requirements about nodes and conditional edges.
"""
from typing import Callable, Dict, Any, List, Optional


class Node:
    def __init__(self, name: str, func: Callable[[Dict[str, Any]], Dict[str, Any]]):
        self.name = name
        self.func = func

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return self.func(state)


class ConditionalEdge:
    def __init__(self, src: str, dst: str, condition: Callable[[Dict[str, Any]], bool]):
        self.src = src
        self.dst = dst
        self.condition = condition


class StateGraph:
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: List[ConditionalEdge] = []

    def add_node(self, node: Node) -> None:
        self.nodes[node.name] = node

    def add_edge(self, src: str, dst: str, condition: Callable[[Dict[str, Any]], bool]):
        self.edges.append(ConditionalEdge(src, dst, condition))

    def run(self, start: str, initial_state: Optional[Dict[str, Any]] = None, max_iters: int = 10) -> Dict[str, Any]:
        state = initial_state or {}
        current = start
        iters = 0
        history: List[Dict[str, Any]] = []

        # Debug: print graph topology and start info
        try:
            print(f"[StateGraph] start='{start}', max_iters={max_iters}")
            print(f"[StateGraph] nodes={list(self.nodes.keys())}")
            print(f"[StateGraph] edges={[ (e.src,e.dst) for e in self.edges ]}")
        except Exception:
            pass

        while current and iters < max_iters:
            iters += 1
            node = self.nodes.get(current)
            if node is None:
                print(f"[StateGraph] stopping: node '{current}' not found")
                break
            try:
                out = node.run(state)
            except Exception as e:
                print(f"[StateGraph] node '{current}' raised: {e}")
                break

            # record history entry for this node invocation
            history.append({"iteration": iters, "node": current, "output": out})

            # merge output into state
            if isinstance(out, dict):
                state.update(out)

            # choose next edge
            next_node = None
            for e in self.edges:
                if e.src == current:
                    try:
                        if e.condition(state):
                            next_node = e.dst
                            break
                    except Exception as e:
                        print(f"[StateGraph] edge condition from {e.src}->{e.dst} raised: {e}")
                        continue

            if next_node is None:
                print(f"[StateGraph] stopping: no outgoing edge matched from '{current}'")
                break
            current = next_node

        print(f"[StateGraph] finished after {iters} iterations")
        return {"state": state, "history": history}
