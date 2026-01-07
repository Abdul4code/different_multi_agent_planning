"""
Minimal LangGraph wrapper for CANONICAL CENTRALIZED PLANNER.

This wrapper supports the plan-then-execute paradigm:
1. Planner creates a COMPLETE plan upfront (task list)
2. Workers execute tasks sequentially from the plan
3. Replanning occurs ONLY in the planner when needed

Key architectural properties:
- Linear plan representation (task list)
- Synchronous control flow (sequential execution)
- Workers are stateless (they don't modify the plan)
"""
from typing import Callable, Dict, Any, List, Optional


class Node:
    """A node in the execution graph representing a worker agent."""
    
    def __init__(self, name: str, func: Callable[[Dict[str, Any]], Dict[str, Any]]):
        self.name = name
        self.func = func

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return self.func(state)


class PlanExecutor:
    """Executes a pre-defined plan created by the central planner.
    
    CANONICAL CENTRALIZED PLANNER BEHAVIOR:
    - Takes a complete plan (task list) from the planner
    - Executes each task sequentially
    - Workers cannot modify the plan
    - Returns to planner only for replanning (on failure)
    """
    
    def __init__(self):
        self.workers: Dict[str, Node] = {}
    
    def add_worker(self, node: Node) -> None:
        """Register a worker node."""
        self.workers[node.name] = node
    
    def execute_plan(self, plan: List[str], initial_state: Dict[str, Any], 
                     planner=None, max_iterations: int = 50) -> Dict[str, Any]:
        """Execute the plan sequentially.
        
        Args:
            plan: List of task names to execute (e.g., ["testrunner", "debug", "writer"])
            initial_state: Initial state dictionary
            planner: PlannerAgent instance for replanning if needed
            max_iterations: Maximum total iterations allowed (safety limit)
            
        Returns:
            Dict with 'state' and 'history' keys
        """
        state = initial_state.copy()
        state["plan"] = plan
        state["plan_index"] = 0
        state["replan_count"] = 0
        history: List[Dict[str, Any]] = []
        
        iteration = 0
        current_plan = list(plan)  # Copy to track original
        
        while state["plan_index"] < len(current_plan):
            iteration += 1
            
            # Check max iterations limit (safety timeout)
            if iteration > max_iterations:
                history.append({
                    "iteration": iteration,
                    "node": "max_iterations_reached",
                    "output": {"status": f"Stopped: max iterations ({max_iterations}) reached"}
                })
                break
            
            task = current_plan[state["plan_index"]]
            
            # Check for "done" signal - but only stop if all tests pass
            if task == "done":
                last_test_run = state.get("last_test_run", {})
                failed = last_test_run.get("failed", 0)
                total = last_test_run.get("total_tests", 0)
                
                # Only truly done if all tests pass
                if failed == 0 and total > 0:
                    history.append({
                        "iteration": iteration,
                        "node": "done",
                        "output": {"status": "All tests passed - completed successfully"}
                    })
                    break
                else:
                    # Tests still failing - replan instead of stopping
                    if planner is not None:
                        replan_result = planner.replan(state)
                        state.update(replan_result)
                        current_plan = replan_result.get("plan", ["debug", "writer", "testrunner", "done"])
                        state["plan_index"] = 0
                        history.append({
                            "iteration": iteration,
                            "node": "planner_replan",
                            "output": replan_result
                        })
                        continue
                    else:
                        # No planner available, just continue with default
                        current_plan = ["debug", "writer", "testrunner", "done"]
                        state["plan_index"] = 0
                        continue
            
            # Execute the worker task
            worker = self.workers.get(task)
            if worker is None:
                history.append({
                    "iteration": iteration,
                    "node": task,
                    "output": {"error": f"Unknown worker: {task}"}
                })
                state["plan_index"] += 1
                continue
            
            # Run the worker (workers are stateless - they just execute)
            output = worker.run(state)
            
            # Record history
            history.append({
                "iteration": iteration,
                "node": task,
                "output": output
            })
            
            # Merge output into state
            if isinstance(output, dict):
                state.update(output)
            
            # Move to next task in plan
            state["plan_index"] += 1
            
            # Check if all tests pass after testrunner - if so, we're done!
            if task == "testrunner":
                last_test_run = state.get("last_test_run", {})
                failed = last_test_run.get("failed", 0)
                total = last_test_run.get("total_tests", 0)
                
                if failed == 0 and total > 0:
                    # All tests pass - we're done!
                    history.append({
                        "iteration": iteration + 1,
                        "node": "done",
                        "output": {"status": "All tests passing"}
                    })
                    break
        
        return {"state": state, "history": history}


# Keep backward compatibility with old StateGraph interface
class ConditionalEdge:
    """Legacy edge type - kept for compatibility."""
    def __init__(self, src: str, dst: str, condition: Callable[[Dict[str, Any]], bool]):
        self.src = src
        self.dst = dst
        self.condition = condition


class StateGraph:
    """Graph-based executor - now wraps PlanExecutor for canonical behavior.
    
    This class maintains backward compatibility while internally using
    the plan-based execution model.
    """
    
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: List[ConditionalEdge] = []
        self._executor = PlanExecutor()

    def add_node(self, node: Node) -> None:
        self.nodes[node.name] = node
        self._executor.add_worker(node)

    def add_edge(self, src: str, dst: str, condition: Callable[[Dict[str, Any]], bool]):
        """Legacy method - edges are now implicit in the plan."""
        self.edges.append(ConditionalEdge(src, dst, condition))

    def run(self, start: str, initial_state: Optional[Dict[str, Any]] = None, 
            max_iters: int = 10, planner=None, plan: List[str] = None) -> Dict[str, Any]:
        """Run the execution graph.
        
        If a plan is provided, uses the canonical plan-based execution.
        Otherwise falls back to edge-based execution for compatibility.
        """
        state = initial_state or {}
        
        # CANONICAL MODE: Execute pre-defined plan
        if plan is not None:
            return self._executor.execute_plan(
                plan=plan,
                initial_state=state,
                planner=planner,
                max_iterations=max_iters
            )
        
        # LEGACY MODE: Edge-based execution (for backward compatibility)
        current = start
        iters = 0
        history: List[Dict[str, Any]] = []
        
        while current and iters < max_iters:
            iters += 1
            node = self.nodes.get(current)
            if node is None:
                break
            out = node.run(state)
            history.append({"iteration": iters, "node": current, "output": out})

            if isinstance(out, dict):
                state.update(out)

            next_node = None
            for e in self.edges:
                if e.src == current:
                    try:
                        if e.condition(state):
                            next_node = e.dst
                            break
                    except Exception:
                        continue

            if next_node is None:
                break
            current = next_node

        return {"state": state, "history": history}
