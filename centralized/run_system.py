"""Entry point to run the centralized LangGraph multi-agent system.

Produces a unified CHANGELOG_AGENT.md with the same structure as the decentralized
version so both can be parsed by the same script.
"""
import argparse
import json
import os
import time
from langgraph_agents.langgraph_wrapper import StateGraph, Node
from langgraph_agents import agents
from langgraph_agents import tools


def build_graph(planner: agents.PlannerAgent, analyzer: agents.CodeAnalyzerAgent, writer: agents.CodeWriterAgent,
                testrunner: agents.TestRunnerAgent, debugger: agents.DebuggerAgent, perf: agents.PerformanceAnalyzerAgent):
    g = StateGraph()

    g.add_node(Node('planner', lambda s: planner.run(s)))
    g.add_node(Node('analyzer', lambda s: analyzer.run(s)))
    g.add_node(Node('writer', lambda s: writer.run(s)))
    g.add_node(Node('testrunner', lambda s: testrunner.run(s)))
    g.add_node(Node('debugger', lambda s: debugger.run(s)))
    g.add_node(Node('perf', lambda s: perf.run(s)))

    g.add_edge('planner', 'analyzer', lambda st: st.get('next_action') in ('analyze', 'start'))
    g.add_edge('analyzer', 'planner', lambda st: True)

    g.add_edge('planner', 'writer', lambda st: st.get('next_action') == 'write')
    g.add_edge('writer', 'testrunner', lambda st: True)
    g.add_edge('testrunner', 'planner', lambda st: True)

    g.add_edge('planner', 'testrunner', lambda st: st.get('next_action') == 'testrunner')
    g.add_edge('testrunner', 'planner', lambda st: True)

    g.add_edge('planner', 'debugger', lambda st: st.get('next_action') == 'debug')
    g.add_edge('debugger', 'writer', lambda st: bool(st.get('fix_instructions')))
    g.add_edge('debugger', 'planner', lambda st: not st.get('fix_instructions'))

    g.add_edge('planner', 'perf', lambda st: st.get('next_action') == 'profile')
    g.add_edge('perf', 'planner', lambda st: True)

    return g


def write_unified_changelog(path: str, meta: dict, final_test_run: dict, aggregated_failed_list: list,
                            history: list, modified_files: list, agent_messages: list):
    # Use history length as authoritative iteration count (node invocations)
    iterations = len(history or [])

    # Always search history for the LAST test run (most recent), regardless of final_test_run
    # This ensures we get the actual final test state, not a stale cached version
    final = None
    for entry in reversed(history or []):
        out = entry.get('output') or {}
        if isinstance(out, dict):
            cand = out.get('last_test_run') or out.get('test_results')
            if isinstance(cand, dict) and (cand.get('failed') is not None or cand.get('total_tests') is not None or cand.get('failed_test_names')):
                final = cand
                break
    # Fallback to passed final_test_run if history search found nothing
    if not final:
        final = final_test_run if isinstance(final_test_run, dict) and final_test_run else {}

    # Robust computation of totals
    total_tests = int(final.get('total_tests') or final.get('total') or 0)
    failed_tests = int(final.get('failed') or final.get('failed_tests') or len(final.get('failed_test_names', []) or []))
    # prefer explicit passed value, else derive
    if final.get('passed') is not None:
        passed_tests = int(final.get('passed'))
    else:
        if total_tests:
            passed_tests = max(0, total_tests - failed_tests)
        else:
            passed_tests = 0
    final_failed_names = final.get('failed_test_names', []) if isinstance(final, dict) else []

    lines = []
    lines.append("# Centralized Multi-Agent System Execution Log\n")
    lines.append(f"**User Request:** {meta.get('user_request', '')}")
    lines.append(f"**Mode:** {meta.get('mode', 'N/A')}")
    lines.append(f"**Iterations:** {iterations}")
    lines.append(f"**Planner notes:** {meta.get('planner_notes', '')}")
    lines.append(f"**Execution Time:** {meta.get('elapsed_seconds', 0.0):.2f} seconds\n")

    lines.append("## Final Test Results\n")
    lines.append(f"- **Total Tests:** {total_tests}")
    lines.append(f"- **Passed:** {passed_tests}")
    lines.append(f"- **Failed:** {failed_tests}\n")

    if failed_tests > 0 and final_failed_names:
        lines.append("### Failing Tests (final run)\n")
        for t in final_failed_names:
            lines.append(f"- `{t}`")
        lines.append("")

    if aggregated_failed_list:
        lines.append("### Detected failing tests (unique across run)\n")
        for t in aggregated_failed_list:
            lines.append(f"- `{t}`")
        lines.append("")

    lines.append("## Modified Files\n")
    for f in modified_files:
        lines.append(f"- `{f}`")
    lines.append("")

    lines.append("## Agent Actions\n")
    for msg in agent_messages or []:
        agent = msg.get('agent', 'Unknown')
        action = msg.get('action', 'unknown')
        summary = msg.get('summary', msg.get('reason', 'No details'))
        lines.append(f"- **{agent}** ({action}): {summary}")
    lines.append("")

    lines.append("## Execution History\n")
    for entry in history or []:
        iteration = entry.get('iteration')
        node = entry.get('node')
        output = entry.get('output')
        lines.append(f"### Iteration {iteration} - Node: {node}\n")
        try:
            out_str = json.dumps(output, indent=2, default=str)
        except Exception:
            out_str = str(output)
        if len(out_str) > 2000:
            out_str = out_str[:2000] + "\n... (truncated)"
        lines.append(out_str)
        lines.append("")

    tools.write_file(path, "\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--user', '-u', type=str, default='Fix failing tests')
    parser.add_argument('--repo', '-r', type=str, default=None)
    parser.add_argument('--max-iterations', '-m', type=int, default=8)
    args = parser.parse_args()

    if args.repo:
        repo_path = os.path.abspath(os.path.expanduser(args.repo))
        if not os.path.isdir(repo_path):
            raise SystemExit(f"Provided repo path does not exist or is not a directory: {repo_path}")
        os.chdir(repo_path)

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

    start_time = time.time()
    result = graph.run('planner', initial_state=initial_state, max_iters=int(args.max_iterations))
    end_time = time.time()
    elapsed_seconds = end_time - start_time

    final_state = result.get('state', {})
    history = result.get('history', [])

    # Always run tests one final time to capture actual end state
    # This ensures the changelog reflects reality even if the graph stopped mid-cycle
    final_test_run = tools.run_tests("pytest -q")
    # Append to history so changelog writer finds it
    history.append({
        'iteration': len(history) + 1,
        'node': 'testrunner (final)',
        'output': {'last_test_run': final_test_run}
    })

    total_tests = final_test_run.get('total_tests', 0)
    passed_tests = final_test_run.get('passed', 0)
    failed_tests = final_test_run.get('failed', 0)

    aggregated_failed = set()
    for entry in history:
        out = entry.get('output', {})
        if isinstance(out, dict):
            lt = out.get('last_test_run') or out.get('test_results') or {}
            if isinstance(lt, dict):
                names = lt.get('failed_test_names') or lt.get('failed_tests') or []
                for n in names:
                    aggregated_failed.add(n)

    aggregated_failed_list = sorted(aggregated_failed)

    # metadata for changelog
    meta = {
        'user_request': args.user,
        'mode': final_state.get('mode', 'N/A'),
        'iteration': final_state.get('iteration', 0),
        'planner_notes': final_state.get('planner_notes', ''),
        'elapsed_seconds': elapsed_seconds,
    }

    # write unified changelog to centralized folder
    changelog_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'CHANGELOG_AGENT.md')
    write_unified_changelog(changelog_path, meta, final_test_run, aggregated_failed_list,
                            history, final_state.get('modified_files', []), final_state.get('agent_messages', []))

    print(f"Run complete. Wrote {changelog_path}")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import sys
        print(f"Error: {e}", file=sys.stderr)
