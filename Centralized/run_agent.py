"""Entrypoint to run the centralized LangGraph multi-agent system.

Usage: python run_agent.py --prompt "Fix failing tests: ..."

This script wires the small StateGraph and nodes and runs the loop until
tests pass or max iterations are reached.
"""
import argparse
import os
import json
from agent.graph import StateGraph
from agent.llm_client import LocalLLM
from agent.nodes import PlannerNode, CodeAnalyzerNode, CodeWriterNode, TestRunnerNode, DebuggerNode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=False, default="Run tests and fix failures until all pass.")
    parser.add_argument("--repo", type=str, required=False, default=os.getcwd())
    parser.add_argument("--max-iter", type=int, default=10)
    args = parser.parse_args()

    llm = LocalLLM()

    planner = PlannerNode(llm)
    analyzer = CodeAnalyzerNode(args.repo)
    writer = CodeWriterNode(args.repo)
    tester = TestRunnerNode(args.repo)
    debugger = DebuggerNode(llm)

    g = StateGraph()
    g.add_node(planner)
    g.add_node(analyzer)
    g.add_node(writer)
    g.add_node(tester)
    g.add_node(debugger)

    # Edges per spec: Planner -> Analyzer -> Writer -> Tester -> Planner
    g.add_edge("planner", "analyzer")
    g.add_edge("analyzer", "writer")
    g.add_edge("writer", "tester")
    g.add_edge("tester", "planner")
    # Planner -> Debugger -> Writer
    g.add_edge("planner", "debugger")
    g.add_edge("debugger", "writer")

    initial_state = {
        "user_prompt": args.prompt,
        "iteration": 0,
        "terminate": False,
    }

    final = g.run(initial_state=initial_state, start="planner", max_iterations=args.max_iter)
    print("Final state:\n", json.dumps({k: (v if k != 'test_results' else {'passed': v.get('passed'), 'returncode': v.get('returncode')} ) for k,v in final.items()}, indent=2, default=str))


if __name__ == "__main__":
    main()
