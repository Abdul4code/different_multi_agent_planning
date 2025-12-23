"""Simple test to verify the decentralized system works without LLM calls."""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langgraph_agents import tools

# Test running tests
print("Testing test runner...")
os.chdir("../agent_benchmark/repo_1_shopping_cart")
result = tools.run_tests("pytest -q")
print(f"Test Results: {result}")
print(f"Total: {result['total_tests']}, Passed: {result['passed']}, Failed: {result['failed']}")
print(f"Failed tests: {result['failed_test_names']}")
