"""
Example: Using the decentralized multi-agent system programmatically.

This demonstrates how to integrate the system into your own code.
"""
import os
import sys

# Add the decentralized_planner to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from langgraph_agents.langgraph_wrapper import StateGraph, Node
from langgraph_agents import agents, tools


def run_decentralized_system(repo_path, user_request, max_iterations=20):
    """Run the decentralized multi-agent system programmatically.
    
    Args:
        repo_path: Path to the repository to operate on
        user_request: User's task description
        max_iterations: Maximum iterations before stopping
        
    Returns:
        dict: Test results and execution summary
    """
    # Change to target repository
    original_dir = os.getcwd()
    repo_path = os.path.abspath(os.path.expanduser(repo_path))
    os.chdir(repo_path)
    
    try:
        # Create autonomous peer agents
        base_url = os.environ.get('OPENAI_API_BASE', 'http://localhost:11434/v1')
        analyzer = agents.CodeAnalyzerAgent(base_url=base_url)
        testrunner = agents.TestRunnerAgent(base_url=base_url)
        debugger = agents.DebuggerAgent(base_url=base_url)
        writer = agents.CodeWriterAgent(base_url=base_url)
        perf = agents.PerformanceAnalyzerAgent(base_url=base_url)

        # Build decentralized graph
        g = StateGraph()
        g.add_node(Node('analyzer', lambda s: analyzer.run(s)))
        g.add_node(Node('testrunner', lambda s: testrunner.run(s)))
        g.add_node(Node('debugger', lambda s: debugger.run(s)))
        g.add_node(Node('writer', lambda s: writer.run(s)))
        g.add_node(Node('perf', lambda s: perf.run(s)))
        
        # Coordinator checks termination conditions (not a planner)
        def coordinator_check(state):
            iteration = state.get('iteration', 0)
            test_results = state.get('test_results', {})
            total_tests = test_results.get('total_tests', 0)
            failed_tests = test_results.get('failed', 0)
            
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
        
        # Define edges (state-based routing, no planner control)
        g.add_edge('analyzer', 'coordinator', lambda st: True)
        
        need_test_run = lambda st: (
            st.get('should_continue', True) and 
            (st.get('test_results') is None or st.get('code_modified', False))
        )
        g.add_edge('coordinator', 'testrunner', need_test_run)
        g.add_edge('testrunner', 'coordinator', lambda st: True)
        
        need_debug = lambda st: (
            st.get('should_continue', True) and
            st.get('test_results') is not None and
            st.get('test_results', {}).get('failed', 0) > 0 and
            not st.get('pending_fixes')
        )
        g.add_edge('coordinator', 'debugger', need_debug)
        g.add_edge('debugger', 'coordinator', lambda st: True)
        
        need_write = lambda st: (
            st.get('should_continue', True) and
            (bool(st.get('pending_fixes')) or bool(st.get('optimization_suggestions')))
        )
        g.add_edge('coordinator', 'writer', need_write)
        g.add_edge('writer', 'coordinator', lambda st: True)
        
        need_perf = lambda st: (
            st.get('should_continue', True) and
            ('perf' in st.get('user_request', '').lower() or 'optimi' in st.get('user_request', '').lower()) and
            st.get('test_results', {}).get('failed', 0) == 0 and
            st.get('test_results', {}).get('total_tests', 0) > 0 and
            not st.get('optimization_suggestions')
        )
        g.add_edge('coordinator', 'perf', need_perf)
        g.add_edge('perf', 'coordinator', lambda st: True)

        # Initial state
        initial_state = {
            'user_request': user_request,
            'iteration': 0,
            'max_iterations': max_iterations,
            'modified_files': [],
            'agent_messages': [],
            'should_continue': True,
        }

        # Execute graph
        result = g.run('analyzer', initial_state=initial_state, max_iters=max_iterations * 6)

        # Extract results
        final_state = result.get('state', {})
        final_test_results = final_state.get('test_results', {})
        
        return {
            'success': True,
            'total_tests': final_test_results.get('total_tests', 0),
            'passed_tests': final_test_results.get('passed', 0),
            'failed_tests': final_test_results.get('failed', 0),
            'failed_test_names': final_test_results.get('failed_test_names', []),
            'iterations': final_state.get('iteration', 0),
            'modified_files': final_state.get('modified_files', []),
            'termination_reason': final_state.get('termination_reason', 'Unknown'),
        }
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
        }
    
    finally:
        # Restore original directory
        os.chdir(original_dir)


if __name__ == '__main__':
    # Example usage
    print("Example: Running decentralized multi-agent system programmatically\n")
    
    # Run on a test repository
    result = run_decentralized_system(
        repo_path='../agent_benchmark/repo_1_shopping_cart',
        user_request='Fix all failing tests',
        max_iterations=10
    )
    
    # Print results
    if result['success']:
        print(f"✅ Execution completed successfully!")
        print(f"\nTest Results:")
        print(f"  Total: {result['total_tests']}")
        print(f"  Passed: {result['passed_tests']}")
        print(f"  Failed: {result['failed_tests']}")
        print(f"\nIterations: {result['iterations']}")
        print(f"Termination: {result['termination_reason']}")
        print(f"\nModified Files:")
        for f in result['modified_files']:
            print(f"  - {f}")
        
        if result['failed_tests'] > 0:
            print(f"\nFailing Tests:")
            for test in result['failed_test_names']:
                print(f"  - {test}")
    else:
        print(f"❌ Execution failed: {result['error']}")
