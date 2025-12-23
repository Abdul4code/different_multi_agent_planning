"""Entry point to run the centralized LangGraph multi-agent system.

Usage: python3 run_system.py --user "Fix failing tests"

This script wires the PlannerAgent and WorkerAgents into a StateGraph and executes
the graph until tests pass or max iterations are reached.
"""
import argparse
import json
import os
from langgraph_agents.langgraph_wrapper import StateGraph, Node
from langgraph_agents import agents
from langgraph_agents import tools


def build_graph(planner: agents.PlannerAgent, analyzer: agents.CodeAnalyzerAgent, writer: agents.CodeWriterAgent,
                testrunner: agents.TestRunnerAgent, debugger: agents.DebuggerAgent, perf: agents.PerformanceAnalyzerAgent):
    g = StateGraph()

    # Each node wraps exactly one agent invocation
    g.add_node(Node('planner', lambda s: planner.run(s)))
    g.add_node(Node('analyzer', lambda s: analyzer.run(s)))
    g.add_node(Node('writer', lambda s: writer.run(s)))
    g.add_node(Node('testrunner', lambda s: testrunner.run(s)))
    g.add_node(Node('debugger', lambda s: debugger.run(s)))
    g.add_node(Node('perf', lambda s: perf.run(s)))

    # Edges: planner -> analyzer -> planner
    g.add_edge('planner', 'analyzer', lambda st: st.get('next_action') == 'analyze' or st.get('next_action') == 'start')
    g.add_edge('analyzer', 'planner', lambda st: True)

    # planner -> writer -> testrunner -> planner
    g.add_edge('planner', 'writer', lambda st: st.get('next_action') == 'write')
    g.add_edge('writer', 'testrunner', lambda st: True)
    g.add_edge('testrunner', 'planner', lambda st: True)

    # planner -> testrunner -> planner (direct path for baseline test runs)
    g.add_edge('planner', 'testrunner', lambda st: st.get('next_action') == 'testrunner')
    g.add_edge('testrunner', 'planner', lambda st: True)

    # planner -> debugger -> writer -> testrunner -> planner (when debugger provides fixes)
    g.add_edge('planner', 'debugger', lambda st: st.get('next_action') == 'debug')
    g.add_edge('debugger', 'writer', lambda st: st.get('fix_instructions') and len(st.get('fix_instructions', [])) > 0)
    g.add_edge('debugger', 'planner', lambda st: not st.get('fix_instructions') or len(st.get('fix_instructions', [])) == 0)
    g.add_edge('writer', 'testrunner', lambda st: True)
    g.add_edge('testrunner', 'planner', lambda st: True)

    # planner -> perf -> planner
    g.add_edge('planner', 'perf', lambda st: st.get('next_action') == 'profile')
    g.add_edge('perf', 'planner', lambda st: True)

    return g


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--user', '-u', type=str, default='Fix failing tests', help='User request')
    parser.add_argument('--repo', '-r', type=str, default=None, help='Path to the target repository to operate on (will chdir into this directory)')
    parser.add_argument('--max-iterations', '-m', type=int, default=8, help='Maximum planner iterations before stopping')
    args = parser.parse_args()

    # If a target repo is provided, change current working directory to it so all
    # tools operate on the specified repository.
    if args.repo:
        repo_path = os.path.abspath(os.path.expanduser(args.repo))
        if not os.path.isdir(repo_path):
            raise SystemExit(f"Provided repo path does not exist or is not a directory: {repo_path}")
        os.chdir(repo_path)
        print(f"Changed working directory to target repo: {repo_path}")

    base_url = os.environ.get('OPENAI_API_BASE', 'http://localhost:11434/v1')

    planner = agents.PlannerAgent(base_url=base_url)
    analyzer = agents.CodeAnalyzerAgent(base_url=base_url)
    writer = agents.CodeWriterAgent(base_url=base_url)
    testrunner = agents.TestRunnerAgent(base_url=base_url)
    debugger = agents.DebuggerAgent(base_url=base_url)
    perf = agents.PerformanceAnalyzerAgent(base_url=base_url)

    graph = build_graph(planner, analyzer, writer, testrunner, debugger, perf)

    initial_state = {
        'user_request': args.user,
        'next_action': 'analyze',
        'iteration': 0,
        'modified_files': [],
        'max_iterations': int(args.max_iterations),
    }

    # Track start time
    import time
    start_time = time.time()

    # Also cap the graph execution loop to the same max iterations to avoid runaway loops
    result = graph.run('planner', initial_state=initial_state, max_iters=int(args.max_iterations))

    # Track end time
    end_time = time.time()
    elapsed_seconds = end_time - start_time
    
    # result contains state and history
    final_state = result.get('state', {})
    history = result.get('history', [])

    # Get final test results
    final_test_run = final_state.get('last_test_run', {})
    total_tests = final_test_run.get('total_tests', 0)
    passed_tests = final_test_run.get('passed', 0)
    failed_tests = final_test_run.get('failed', 0)

    # After completion, ensure CHANGELOG_AGENT.md is written with detailed history
    lines = []
    lines.append(f"Mode: {final_state.get('mode')}")
    lines.append(f"Iterations: {final_state.get('iteration')}")
    lines.append(f"Planner notes: {final_state.get('planner_notes')}")
    lines.append(f"Elapsed time: {elapsed_seconds:.2f} seconds")
    lines.append("")
    lines.append("Final test results:")
    lines.append(f"  Total tests: {total_tests}")
    lines.append(f"  Passed: {passed_tests}")
    lines.append(f"  Failed: {failed_tests}")
    lines.append("")
    lines.append("Execution history:")
    for entry in history:
        it = entry.get('iteration')
        node = entry.get('node')
        out = entry.get('output')
        # keep output short if large
        try:
            out_str = json.dumps(out, indent=2, default=str)
        except Exception:
            out_str = str(out)
        if len(out_str) > 2000:
            out_str = out_str[:2000] + "\n... (truncated)"
        lines.append(f"- Iteration {it} | Node: {node}")
        lines.append(out_str)
        lines.append("")

    lines.append("Modified files:")
    modified = final_state.get('modified_files', [])
    for m in modified:
        lines.append(f" - {m}")

    # Ensure changelog is written to the repository root (same directory as this script)
    project_root = os.path.dirname(os.path.abspath(__file__))
    changelog_path = os.path.join(project_root, 'CHANGELOG_AGENT.md')
    tools.write_file(changelog_path, "\n".join(lines))
    print(f"Run complete. Wrote {changelog_path}")


if __name__ == '__main__':
    main()
