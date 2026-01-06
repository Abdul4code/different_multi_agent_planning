"""Decentralized Multi-Agent System Runner.

CANONICAL DECENTRALIZED PLANNER (Peer-to-Peer Architecture)
===========================================================
This runner implements TRUE decentralized coordination:

1. NO central planner - agents are autonomous peers
2. NO fixed control flow - task ordering emerges from interactions
3. ASYNCHRONOUS execution - agents run concurrently in separate threads
4. Communication via blackboard - not via graph edges

Execution Model:
  User Goal → Broadcast to Blackboard
                    ↓
    Agent A ⇄ Agent B ⇄ Agent C (all peers, running concurrently)
         ↓        ↓        ↓
    Local Actions & Messages on Blackboard
                    ↓
    Goal Satisfied (emergent termination)

There is NO orchestration layer controlling the agents.
Agents run ASYNCHRONOUSLY - this is the canonical decentralized behavior.
"""
import argparse
import json
import os
import time
import threading
from typing import Dict, List, Any

from langgraph_agents.blackboard import Blackboard, Message
from langgraph_agents.autonomous_agents import (
    CodeAnalyzerAgent,
    TestRunnerAgent,
    DebuggerAgent,
    CodeWriterAgent,
    PerformanceAnalyzerAgent
)
from langgraph_agents import tools


def goal_satisfied(blackboard: Blackboard) -> bool:
    """Check if the user's goal has been achieved.
    
    This is NOT a central controller - it's just a termination check.
    Each agent could also have this logic internally.
    """
    test_results = blackboard.get_shared("test_results")
    if not test_results:
        return False
    
    # Goal: all tests pass
    total = test_results.get("total_tests", 0)
    failed = test_results.get("failed", 0)
    
    return total > 0 and failed == 0


def run_agents_async(agents: List, blackboard: Blackboard, max_iterations: int = 10, 
                     timeout: float = 900.0) -> Dict[str, Any]:
    """Run agents asynchronously in separate threads.
    
    This is the CANONICAL decentralized execution:
    - Each agent runs independently in its own thread
    - No central coordination or sequencing
    - Agents communicate ONLY via blackboard
    - Task ordering EMERGES from agent interactions
    
    This is fundamentally different from centralized where a planner
    controls the execution order.
    """
    print("\nStarting agents asynchronously (true peer-to-peer)...")
    print("Each agent runs independently in its own thread.\n")
    
    # Start all agents concurrently - they each run their own loop
    for agent in agents:
        print(f"  → Starting {agent.name}")
        agent.start_async(max_iterations=max_iterations, stop_condition=goal_satisfied)
    
    # Monitor progress without controlling agents
    start = time.time()
    last_status = ""
    goal_achieved = False
    
    while time.time() - start < timeout:
        # Check if goal achieved
        if goal_satisfied(blackboard):
            print("\n✓ Goal satisfied - all tests passing!")
            goal_achieved = True
            # Immediately signal all agents to stop
            for agent in agents:
                agent.stop()
            break
        
        # Check if all agents stopped
        running_agents = [a for a in agents if a._running]
        if not running_agents:
            print("\n⚠ All agents have stopped")
            break
        
        # Print status updates (observation only, not control)
        test_results = blackboard.get_shared("test_results") or {}
        status = f"Running: {len(running_agents)} agents | Tests: {test_results.get('passed', '?')}/{test_results.get('total_tests', '?')} passed"
        if status != last_status:
            print(f"  [{time.time() - start:.1f}s] {status}")
            last_status = status
        
        time.sleep(0.25)  # Check more frequently (every 250ms instead of 500ms)
    
    elapsed = time.time() - start
    
    # Signal agents to stop (if timeout reached or not already stopped)
    if not goal_achieved:
        print("\nStopping agents...")
        for agent in agents:
            agent.stop()
    
    # Collect history from all agents
    all_history = []
    for agent in agents:
        for entry in agent.state.action_history:
            entry["agent"] = agent.name
            all_history.append(entry)
    
    # Sort by timestamp to see true interleaving
    all_history.sort(key=lambda x: x.get("timestamp", 0))
    
    return {
        "history": all_history,
        "elapsed": elapsed,
        "iterations": len(all_history)
    }


def write_changelog(path: str, meta: Dict, history: List, blackboard: Blackboard):
    """Write execution log."""
    test_results = blackboard.get_shared("test_results") or {}
    modified_files = blackboard.get_shared("modified_files") or []
    
    lines = []
    lines.append("# Decentralized Multi-Agent System Execution Log\n")
    lines.append("## Architecture: CANONICAL DECENTRALIZED PLANNER (Peer-to-Peer)\n")
    lines.append("Key Properties:")
    lines.append("- ✓ Multiple Planning Authorities: Each agent plans locally")
    lines.append("- ✓ Autonomous Stateful Agents: Maintain beliefs, goals, commitments")
    lines.append("- ✓ Distributed Plan Representation: No global plan, emergent ordering")
    lines.append("- ✓ Asynchronous Execution: Agents run concurrently in separate threads\n")
    
    lines.append(f"**User Request:** {meta.get('user_request', '')}")
    lines.append(f"**Execution Mode:** Asynchronous (concurrent threads)")
    lines.append(f"**Total Actions:** {meta.get('iterations', 0)}")
    lines.append(f"**Execution Time:** {meta.get('elapsed_seconds', 0):.2f} seconds\n")
    
    lines.append("## Final Test Results\n")
    lines.append(f"- **Total Tests:** {test_results.get('total_tests', 0)}")
    lines.append(f"- **Passed:** {test_results.get('passed', 0)}")
    lines.append(f"- **Failed:** {test_results.get('failed', 0)}\n")
    
    failed_names = test_results.get('failed_test_names', [])
    if failed_names:
        lines.append("### Failing Tests\n")
        for name in failed_names:
            lines.append(f"- `{name}`")
        lines.append("")
    
    lines.append("## Modified Files\n")
    for f in modified_files:
        lines.append(f"- `{f}`")
    lines.append("")
    
    lines.append("## Execution History (Async - True Interleaving)\n")
    lines.append("Actions are sorted by timestamp, showing how agents ran concurrently:\n")
    
    for i, entry in enumerate(history[:50]):  # Limit to 50 entries
        agent = entry.get("agent", "?")
        plan = entry.get("plan", {})
        action = plan.get("action", "?")
        reason = plan.get("reason", "")[:60]
        ts = entry.get("timestamp", 0)
        lines.append(f"{i+1}. [{agent}] {action}: {reason}")
    
    if len(history) > 50:
        lines.append(f"\n... and {len(history) - 50} more actions")
    lines.append("")
    
    # Write blackboard messages
    lines.append("## Blackboard Messages (Last 30)\n")
    for msg in blackboard.get_messages()[-30:]:
        lines.append(f"- [{msg.sender}] {msg.msg_type}: {str(msg.content)[:80]}")
    
    tools.write_file(path, "\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Canonical Decentralized Multi-Agent System (Async)")
    parser.add_argument('--user', '-u', type=str, default='Fix failing tests',
                        help='User request/goal')
    parser.add_argument('--repo', '-r', type=str, default=None,
                        help='Path to target repository')
    parser.add_argument('--max-iterations', '-m', type=int, default=15,
                        help='Maximum iterations per agent')
    parser.add_argument('--timeout', '-t', type=float, default=900.0,
                        help='Maximum execution time in seconds (15 minutes)')
    args = parser.parse_args()

    # Change to target repo
    if args.repo:
        repo_path = os.path.abspath(os.path.expanduser(args.repo))
        if not os.path.isdir(repo_path):
            raise SystemExit(f"Repo path does not exist: {repo_path}")
        os.chdir(repo_path)
        print(f"Working directory: {repo_path}")

    base_url = os.environ.get('OPENAI_API_BASE', 'http://localhost:11434/v1')

    print("=" * 60)
    print("CANONICAL DECENTRALIZED PLANNER")
    print("Peer-to-Peer Multi-Agent System (Asynchronous)")
    print("=" * 60)
    print(f"User Request: {args.user}")
    print(f"Max Iterations per Agent: {args.max_iterations}")
    print(f"Timeout: {args.timeout}s")
    print()
    print("Architecture Properties:")
    print("  • No central planner - each agent plans locally")
    print("  • No fixed control flow - ordering emerges from interactions")
    print("  • Agents maintain local state (beliefs, goals, commitments)")
    print("  • ASYNCHRONOUS execution - agents run concurrently")
    print("  • Communication via blackboard messages only")
    print("=" * 60)

    # Create shared blackboard (communication medium, NOT controller)
    blackboard = Blackboard()
    blackboard.update_shared("user_goal", args.user)

    # Create autonomous peer agents
    analyzer = CodeAnalyzerAgent(blackboard, base_url=base_url)
    testrunner = TestRunnerAgent(blackboard, base_url=base_url)
    debugger = DebuggerAgent(blackboard, base_url=base_url)
    writer = CodeWriterAgent(blackboard, base_url=base_url)
    perf = PerformanceAnalyzerAgent(blackboard, base_url=base_url)

    agents = [analyzer, testrunner, debugger, writer, perf]

    # Broadcast initial goal to blackboard
    blackboard.post(Message(
        sender="User",
        msg_type="goal_posted",
        content={"goal": args.user}
    ))

    # Run agents asynchronously (the ONLY mode - canonical decentralized)
    result = run_agents_async(
        agents, 
        blackboard, 
        max_iterations=args.max_iterations,
        timeout=args.timeout
    )
    
    history = result["history"]
    elapsed = result["elapsed"]
    iterations = result["iterations"]

    # Final test run to capture actual state
    final_results = tools.run_tests("pytest -q")
    blackboard.update_shared("test_results", {
        "total_tests": int(final_results.get("total_tests", 0)),
        "passed": int(final_results.get("passed", 0)),
        "failed": int(final_results.get("failed", 0)),
        "failed_test_names": final_results.get("failed_test_names", []),
    })

    # Print results
    test_results = blackboard.get_shared("test_results") or {}
    print()
    print("=" * 60)
    print("EXECUTION COMPLETE")
    print("=" * 60)
    print(f"Execution Time: {elapsed:.2f} seconds")
    print(f"Total Actions: {iterations}")
    print(f"\nTest Results:")
    print(f"  Total: {test_results.get('total_tests', 0)}")
    print(f"  Passed: {test_results.get('passed', 0)}")
    print(f"  Failed: {test_results.get('failed', 0)}")
    print(f"\nModified Files: {blackboard.get_shared('modified_files', [])}")
    print("=" * 60)

    # Write changelog
    meta = {
        "user_request": args.user,
        "iterations": iterations,
        "elapsed_seconds": elapsed,
    }
    
    changelog_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'CHANGELOG_AGENT.md')
    write_changelog(changelog_path, meta, history, blackboard)
    print(f"\nWrote log to: {changelog_path}")

    return {
        "total_tests": test_results.get("total_tests", 0),
        "passed": test_results.get("passed", 0),
        "failed": test_results.get("failed", 0),
        "elapsed_seconds": elapsed,
        "iterations": iterations,
    }


if __name__ == '__main__':
    main()
