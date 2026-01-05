"""Entry point for the decentralized LangGraph multi-agent system.

Usage: python3 run_system.py --user "Fix failing tests" --repo ./path/to/repo

This script creates a truly decentralized multi-agent system where:
- NO agent acts as a central planner
- ALL agents are autonomous peers
- Coordination emerges through shared state
- Each agent decides when to act based on state observation
- Control flow is determined by agent decisions, not hardcoded logic
"""
import argparse
import json
import os
import time
from typing import Dict
from langgraph_agents.langgraph_wrapper import StateGraph, Node
from langgraph_agents import agents
from langgraph_agents import tools


def build_decentralized_graph(
    analyzer: agents.CodeAnalyzerAgent,
    testrunner: agents.TestRunnerAgent,
    debugger: agents.DebuggerAgent,
    writer: agents.CodeWriterAgent,
    perf: agents.PerformanceAnalyzerAgent
) -> StateGraph:
    """Build a decentralized graph where agents coordinate without a central planner.
    
    Key principles:
    1. Each node wraps exactly one autonomous agent
    2. No planner node exists
    3. Agents decide whether to act via should_act()
    4. Edges are conditional based on shared state, not planner decisions
    5. Termination emerges from state convergence (all tests passing)
    """
    g = StateGraph()

    # Add agent nodes (each agent is autonomous)
    g.add_node(Node('analyzer', lambda s: analyzer.run(s)))
    g.add_node(Node('testrunner', lambda s: testrunner.run(s)))
    g.add_node(Node('debugger', lambda s: debugger.run(s)))
    g.add_node(Node('writer', lambda s: writer.run(s)))
    g.add_node(Node('perf', lambda s: perf.run(s)))
    
    # Helper to check if all tests pass (for early termination)
    def all_tests_pass(st):
        tr = st.get('test_results') or {}
        total = tr.get('total_tests', 0)
        failed = tr.get('failed', 0)
        return total > 0 and failed == 0
    
    # Decentralized wiring (no central coordinator)
    # Agents are connected directly. Each agent decides whether to act via should_act().
    # Ordering here drives the execution order, but agents may skip if should_act() is False.

    # After analysis, run tests
    g.add_edge('analyzer', 'testrunner', lambda st: True)

    # After tests, check if we should continue or stop (early termination if all pass)
    # If tests fail, go to debugger; if all pass, go to perf (which will likely skip) then stop
    g.add_edge('testrunner', 'debugger', lambda st: not all_tests_pass(st))
    g.add_edge('testrunner', 'perf', lambda st: all_tests_pass(st))  # Only go to perf if tests pass

    # Debugger suggests fixes if failures exist
    g.add_edge('debugger', 'writer', lambda st: bool(st.get('pending_fixes')))
    # If debugger has nothing to do, stop (no edge matches = termination)

    # Writer applies fixes then trigger tests again
    g.add_edge('writer', 'testrunner', lambda st: True)

    # Perf analyzer runs after all tests pass, then stops (no outgoing edge = termination)

    return g


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--user', '-u', type=str, default='Fix failing tests', help='User request')
    parser.add_argument('--repo', '-r', type=str, default=None, help='Path to target repository')
    parser.add_argument('--max-iterations', '-m', type=int, default=20, help='Maximum iterations')
    args = parser.parse_args()

    # Change to target repository
    if args.repo:
        repo_path = os.path.abspath(os.path.expanduser(args.repo))
        if not os.path.isdir(repo_path):
            raise SystemExit(f"Provided repo path does not exist: {repo_path}")
        os.chdir(repo_path)
        print(f"Changed working directory to: {repo_path}")

    base_url = os.environ.get('OPENAI_API_BASE', 'http://localhost:11434/v1')

    # Create autonomous peer agents
    analyzer = agents.CodeAnalyzerAgent(base_url=base_url)
    testrunner = agents.TestRunnerAgent(base_url=base_url)
    debugger = agents.DebuggerAgent(base_url=base_url)
    writer = agents.CodeWriterAgent(base_url=base_url)
    perf = agents.PerformanceAnalyzerAgent(base_url=base_url)

    # Build decentralized graph
    graph = build_decentralized_graph(analyzer, testrunner, debugger, writer, perf)

    # Initial state
    initial_state = {
        'user_request': args.user,
        'iteration': 0,
        'max_iterations': args.max_iterations,
        'modified_files': [],
        'agent_messages': [],
        'should_continue': True,
    }

    print(f"\n{'='*60}")
    print(f"DECENTRALIZED MULTI-AGENT SYSTEM")
    print(f"{'='*60}")
    print(f"User Request: {args.user}")
    print(f"Max Iterations: {args.max_iterations}")
    print(f"{'='*60}\n")

    # Track start time
    start_time = time.time()

    # Execute graph (starts at analyzer)
    # max_iters = exact number of node invocations allowed
    result = graph.run('analyzer', initial_state=initial_state, max_iters=args.max_iterations)

    # Track end time
    end_time = time.time()
    elapsed_seconds = end_time - start_time

    # Extract final state and history
    final_state = result.get('state', {}) or {}
    history = result.get('history', []) or []

    # Always run tests one final time to capture actual end state
    # This ensures the changelog reflects reality even if the graph stopped mid-cycle
    final_test_results = tools.run_tests("pytest -q")
    # Append to history so changelog writer finds it
    history.append({
        'iteration': len(history) + 1,
        'node': 'testrunner (final)',
        'output': {'test_results': final_test_results}
    })

    # Use history length as authoritative iteration count (node invocations)
    iterations = len(history)

    total_tests = final_test_results.get('total_tests', 0)
    passed_tests = final_test_results.get('passed', 0)
    failed_tests = final_test_results.get('failed', 0)
    failed_test_names = final_test_results.get('failed_test_names', [])

    # Aggregate unique failing test names seen during the whole run
    aggregated_failed = set()
    for entry in history:
        output = entry.get('output', {})
        if isinstance(output, dict):
            # testrunner may return data under 'last_test_run' or 'test_results'
            tr = output.get('last_test_run') or output.get('test_results') or {}
            if isinstance(tr, dict):
                names = tr.get('failed_test_names') or tr.get('failed_tests') or []
                try:
                    for n in names:
                        aggregated_failed.add(n)
                except Exception:
                    pass

    aggregated_failed_list = sorted(aggregated_failed)
    aggregated_failed_count = len(aggregated_failed_list)

    # Determine termination reason
    if total_tests > 0 and failed_tests == 0:
        termination_reason = "All tests passing"
    elif iterations >= args.max_iterations:
        termination_reason = "Max iterations reached"
    else:
        termination_reason = "No matching edge (graph converged)"

    # Print results (use history-derived iterations)
    print(f"\n{'='*60}")
    print(f"EXECUTION COMPLETE")
    print(f"{'='*60}")
    print(f"Execution Time: {elapsed_seconds:.2f} seconds")
    print(f"Iterations: {iterations}")
    print(f"Termination Reason: {termination_reason}")
    print(f"\nTest Results:")
    print(f"  Total Tests: {total_tests}")
    print(f"  Passed: {passed_tests}")
    print(f"  Failed: {failed_tests}")
    
    if failed_tests > 0:
        print(f"\nFailing Tests:")
        for test_name in failed_test_names:
            print(f"  - {test_name}")
    
    print(f"\nModified Files:")
    for filepath in final_state.get('modified_files', []):
        print(f"  - {filepath}")
    
    print(f"{'='*60}\n")

    # Write CHANGELOG_AGENT.md
    lines = []
    lines.append("# Decentralized Multi-Agent System Execution Log\n")
    lines.append(f"**User Request:** {args.user}\n")
    lines.append(f"**Execution Time:** {elapsed_seconds:.2f} seconds")
    lines.append(f"**Iterations:** {iterations}")
    lines.append(f"**Termination Reason:** {termination_reason}\n")
    
    lines.append("## Final Test Results\n")
    lines.append(f"- **Total Tests:** {total_tests}")
    lines.append(f"- **Passed:** {passed_tests}")
    lines.append(f"- **Failed:** {failed_tests}\n")

    # If tests failed, include the final run's failing tests and also aggregated unique failures
    if failed_tests > 0:
        if failed_test_names:
            lines.append("### Failing Tests (final run)\n")
            for test_name in failed_test_names:
                lines.append(f"- `{test_name}`")
            lines.append("")

    if aggregated_failed_count > 0:
        lines.append("### Detected failing tests (unique across run)\n")
        for test_name in aggregated_failed_list:
            lines.append(f"- `{test_name}`")
        lines.append("")
    
    lines.append("## Modified Files\n")
    for filepath in final_state.get('modified_files', []):
        lines.append(f"- `{filepath}`")
    lines.append("")
    
    lines.append("## Agent Actions\n")
    agent_messages = final_state.get('agent_messages', [])
    for msg in agent_messages:
        agent = msg.get('agent', 'Unknown')
        action = msg.get('action', 'unknown')
        summary = msg.get('summary', msg.get('reason', 'No details'))
        lines.append(f"- **{agent}** ({action}): {summary}")
    lines.append("")
    
    lines.append("## Execution History\n")
    for entry in history:
        iteration = entry.get('iteration')
        node = entry.get('node')
        output = entry.get('output', {})
        
        lines.append(f"### Iteration {iteration} - Node: {node}\n")
        
        # Extract relevant info from output
        if isinstance(output, dict):
            if 'agent_messages' in output and output['agent_messages']:
                last_msg = output['agent_messages'][-1]
                lines.append(f"**Action:** {last_msg.get('action', 'N/A')}")
                lines.append(f"**Summary:** {last_msg.get('summary', last_msg.get('reason', 'N/A'))}\n")
            elif node == 'coordinator':
                lines.append(f"**Should Continue:** {output.get('should_continue', 'N/A')}")
                lines.append(f"**Iteration:** {output.get('iteration', 'N/A')}\n")
    
    # Write changelog to project root
    project_root = os.path.dirname(os.path.abspath(__file__))
    changelog_path = os.path.join(project_root, 'CHANGELOG_AGENT.md')
    tools.write_file(changelog_path, "\n".join(lines))
    print(f"Wrote execution log to: {changelog_path}")

    # Return structured results
    return {
        'total_tests': total_tests,
        'passed_tests': passed_tests,
        'failed_tests': failed_tests,
        'failed_test_names': failed_test_names,
        'elapsed_seconds': elapsed_seconds,
        'iterations': iterations,
        'modified_files': final_state.get('modified_files', []),
    }


if __name__ == '__main__':
    main()
