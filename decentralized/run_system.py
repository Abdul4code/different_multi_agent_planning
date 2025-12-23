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
    
    # Create a coordinator node that checks termination conditions
    # This is NOT a planner - it only checks if we should continue
    def coordinator_check(state: Dict) -> Dict:
        """Check termination conditions based on state convergence.
        
        This is not a planning agent - it's a pure state observer.
        """
        iteration = state.get('iteration', 0)
        max_iterations = state.get('max_iterations', 20)
        
        test_results = state.get('test_results', {})
        total_tests = test_results.get('total_tests', 0)
        failed_tests = test_results.get('failed', 0)
        
        # Determine if system should continue
        should_continue = True
        termination_reason = None
        
        if total_tests > 0 and failed_tests == 0:
            should_continue = False
            termination_reason = "All tests passing"
        elif iteration >= max_iterations:
            should_continue = False
            termination_reason = "Max iterations reached"
        
        return {
            'iteration': iteration + 1,
            'should_continue': should_continue,
            'termination_reason': termination_reason,
        }
    
    g.add_node(Node('coordinator', coordinator_check))
    
    # Define decentralized edges based on state conditions
    # Pattern: Each agent connects to coordinator, coordinator connects to next agent
    
    # Start with analyzer (initial code analysis)
    # analyzer -> coordinator
    g.add_edge('analyzer', 'coordinator', lambda st: True)
    
    # coordinator -> testrunner (if analysis complete and tests need to run)
    def need_test_run(st):
        # Run tests if no results exist OR code was modified
        no_results = st.get('test_results') is None
        code_modified = st.get('code_modified', False)
        return st.get('should_continue', True) and (no_results or code_modified)
    
    g.add_edge('coordinator', 'testrunner', need_test_run)
    
    # testrunner -> coordinator
    g.add_edge('testrunner', 'coordinator', lambda st: True)
    
    # coordinator -> debugger (if tests failed and no pending fixes)
    def need_debug(st):
        has_test_results = st.get('test_results') is not None
        tests_failed = st.get('test_results', {}).get('failed', 0) > 0
        no_pending = not st.get('pending_fixes')
        return st.get('should_continue', True) and has_test_results and tests_failed and no_pending
    
    g.add_edge('coordinator', 'debugger', need_debug)
    
    # debugger -> coordinator
    g.add_edge('debugger', 'coordinator', lambda st: True)
    
    # coordinator -> writer (if pending fixes or optimizations exist)
    def need_write(st):
        has_fixes = bool(st.get('pending_fixes'))
        has_opts = bool(st.get('optimization_suggestions'))
        return st.get('should_continue', True) and (has_fixes or has_opts)
    
    g.add_edge('coordinator', 'writer', need_write)
    
    # writer -> coordinator
    g.add_edge('writer', 'coordinator', lambda st: True)
    
    # coordinator -> perf (if performance mode and tests passing)
    def need_perf(st):
        user_req = st.get('user_request', '').lower()
        is_perf_mode = 'perf' in user_req or 'optimi' in user_req
        tests_pass = st.get('test_results', {}).get('failed', 0) == 0
        has_tests = st.get('test_results', {}).get('total_tests', 0) > 0
        no_pending_opts = not st.get('optimization_suggestions')
        return st.get('should_continue', True) and is_perf_mode and tests_pass and has_tests and no_pending_opts
    
    g.add_edge('coordinator', 'perf', need_perf)
    
    # perf -> coordinator
    g.add_edge('perf', 'coordinator', lambda st: True)
    
    # coordinator -> analyzer (loop back for re-analysis - this should rarely trigger)
    def need_reanalyze(st):
        # Only re-analyze if we're stuck and nothing else to do
        return False  # Disabled to prevent loops
    
    g.add_edge('coordinator', 'analyzer', need_reanalyze)
    
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
    result = graph.run('analyzer', initial_state=initial_state, max_iters=args.max_iterations * 6)  # *6 because each iteration involves multiple agent nodes

    # Track end time
    end_time = time.time()
    elapsed_seconds = end_time - start_time

    # Extract final state
    final_state = result.get('state', {})
    history = result.get('history', [])

    # Get final test results
    final_test_results = final_state.get('test_results', {})
    total_tests = final_test_results.get('total_tests', 0)
    passed_tests = final_test_results.get('passed', 0)
    failed_tests = final_test_results.get('failed', 0)
    failed_test_names = final_test_results.get('failed_test_names', [])

    # Print results
    print(f"\n{'='*60}")
    print(f"EXECUTION COMPLETE")
    print(f"{'='*60}")
    print(f"Elapsed Time: {elapsed_seconds:.2f} seconds")
    print(f"Iterations: {final_state.get('iteration', 0)}")
    print(f"Termination Reason: {final_state.get('termination_reason', 'Unknown')}")
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
    lines.append(f"**Iterations:** {final_state.get('iteration', 0)}")
    lines.append(f"**Termination Reason:** {final_state.get('termination_reason', 'Unknown')}\n")
    
    lines.append("## Final Test Results\n")
    lines.append(f"- **Total Tests:** {total_tests}")
    lines.append(f"- **Passed:** {passed_tests}")
    lines.append(f"- **Failed:** {failed_tests}\n")
    
    if failed_tests > 0:
        lines.append("### Failing Tests\n")
        for test_name in failed_test_names:
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
        'iterations': final_state.get('iteration', 0),
        'modified_files': final_state.get('modified_files', []),
    }


if __name__ == '__main__':
    main()
