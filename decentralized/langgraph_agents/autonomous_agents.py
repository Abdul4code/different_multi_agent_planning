"""Truly Decentralized Agent Implementations.

CANONICAL DECENTRALIZED PLANNER (Peer-to-Peer Architecture)
===========================================================
This implementation follows the canonical decentralized planning pattern:

1. Multiple Planning Authorities: Each agent plans locally, no global planner
2. Autonomous Stateful Agents: Agents maintain internal state, goals, and beliefs
3. Distributed Plan Representation: No single global plan, emergent task ordering
4. Asynchronous Event-Driven Control: Agents run independently, coordinate via messages

Key discriminator: Planning decisions EMERGE from peer interactions,
not from a single controller.

Each agent:
- Runs its own perception-planning-action loop
- Maintains local state (beliefs, goals, commitments)
- Decides WHAT to do, not just IF
- Can generate new tasks for itself or others
- Communicates via blackboard messages
"""
from typing import Dict, Any, List, Optional, Set
import os
import json
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

try:
    from langchain.chat_models import ChatOpenAI
    from langchain.schema import HumanMessage, SystemMessage
except Exception:
    import requests

    class HumanMessage:
        def __init__(self, content: str):
            self.content = content

    class SystemMessage:
        def __init__(self, content: str):
            self.content = content

    class ChatOpenAI:
        def __init__(self, model: str = "qwen2.5:7b", temperature: float = 0.2, 
                     openai_api_base: str = "http://localhost:11434/v1"):
            self.model = model
            self.temperature = temperature
            self.base = openai_api_base.rstrip("/")

        def __call__(self, messages):
            payload_msgs = []
            for m in messages:
                role = "user"
                if isinstance(m, SystemMessage):
                    role = "system"
                payload_msgs.append({"role": role, "content": m.content})

            url = f"{self.base}/chat/completions"
            resp = requests.post(url, json={
                "model": self.model,
                "messages": payload_msgs,
                "temperature": self.temperature,
            }, timeout=180)
            resp.raise_for_status()
            j = resp.json()
            try:
                content = j["choices"][0]["message"]["content"]
            except Exception:
                content = str(j)

            class Resp:
                def __init__(self, content):
                    self.content = content

            return Resp(content)

from .blackboard import Blackboard, Message
from . import tools

LLM_SETTINGS = {
    "model": "qwen2.5:7b",
    "temperature": 0.2,
}


@dataclass
class AgentState:
    """Local state maintained by each agent.
    
    This is the agent's internal beliefs and commitments - 
    NOT shared state. Each agent has its own view of the world.
    """
    beliefs: Dict[str, Any] = field(default_factory=dict)
    goals: List[str] = field(default_factory=list)
    commitments: List[str] = field(default_factory=list)  # Tasks I've committed to
    last_action: Optional[str] = None
    last_observation_time: float = 0.0
    action_history: List[Dict[str, Any]] = field(default_factory=list)


class AutonomousAgent(ABC):
    """Base class for truly autonomous decentralized agents.
    
    KEY DIFFERENCES FROM CENTRALIZED WORKERS:
    1. Local Planning: Each agent decides WHAT to do, not just IF
    2. Internal State: Maintains beliefs, goals, commitments across steps
    3. Task Generation: Can generate new tasks for self or others
    4. Independent Execution: Runs its own perception-plan-act loop
    5. Peer Communication: Interacts via blackboard, not via graph edges
    
    This is NOT a worker that executes assigned tasks.
    This IS an autonomous peer that makes its own decisions.
    """
    
    def __init__(self, name: str, blackboard: Blackboard, 
                 base_url: str = "http://localhost:11434/v1"):
        self.name = name
        self.blackboard = blackboard
        self.client = ChatOpenAI(
            model=LLM_SETTINGS["model"], 
            temperature=LLM_SETTINGS["temperature"], 
            openai_api_base=base_url
        )
        
        # LOCAL state - this is the agent's internal world model
        self.state = AgentState()
        
        # Running flag for async execution
        self._running = False
        self._thread: Optional[threading.Thread] = None
    
    def call_model(self, system_prompt: str, user_prompt: str) -> str:
        """Call LLM with retry logic."""
        messages = [
            SystemMessage(content=system_prompt), 
            HumanMessage(content=user_prompt)
        ]
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = self.client(messages)
                return resp.content
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return f"ERROR: {e}"
    
    def observe(self) -> Dict[str, Any]:
        """Observe the current state of the world via blackboard.
        
        This is the PERCEPTION phase of the agent loop.
        """
        observation = {
            "blackboard_summary": self.blackboard.get_summary(),
            "shared_data": self.blackboard.get_all_shared(),
            "recent_messages": [m.to_dict() for m in self.blackboard.get_messages()[-20:]],
            "my_state": {
                "beliefs": self.state.beliefs,
                "goals": self.state.goals,
                "commitments": self.state.commitments,
                "last_action": self.state.last_action,
            }
        }
        self.state.last_observation_time = time.time()
        return observation
    
    @abstractmethod
    def plan(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Decide what action to take based on observation.
        
        This is the PLANNING phase - the agent decides WHAT to do.
        Returns a plan dict with:
        - action: what to do (e.g., "analyze_code", "run_tests", "generate_fix")
        - reason: why this action
        - details: action-specific parameters
        
        This is LOCAL planning - no global coordinator is involved.
        """
        pass
    
    @abstractmethod
    def act(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the planned action.
        
        This is the ACTION phase - the agent does something.
        Returns result of the action.
        """
        pass
    
    def communicate(self, msg_type: str, content: Dict[str, Any]) -> None:
        """Post a message to the blackboard.
        
        This is how agents coordinate without a central controller.
        """
        message = Message(
            sender=self.name,
            msg_type=msg_type,
            content=content
        )
        self.blackboard.post(message)
    
    def step(self) -> Dict[str, Any]:
        """Execute one perception-planning-action cycle.
        
        This is the agent's main loop iteration.
        """
        # 1. PERCEIVE: Observe the world
        observation = self.observe()
        
        # 2. PLAN: Decide what to do (LOCAL planning)
        plan = self.plan(observation)
        
        # 3. ACT: Execute the action
        if plan.get("action") == "wait":
            result = {"status": "waiting", "reason": plan.get("reason", "Nothing to do")}
        else:
            result = self.act(plan)
        
        # 4. UPDATE: Update local state
        self.state.last_action = plan.get("action")
        self.state.action_history.append({
            "plan": plan,
            "result": result,
            "timestamp": time.time()
        })
        
        # 5. COMMUNICATE: Share results with peers
        self.communicate("action_completed", {
            "action": plan.get("action"),
            "result_summary": str(result)[:200]
        })
        
        return {"plan": plan, "result": result}
    
    def run_loop(self, max_iterations: int = 10, stop_condition: callable = None):
        """Run the agent's main loop.
        
        Each agent runs independently - this is NOT controlled by a central coordinator.
        
        Stopping conditions:
        1. stop_condition returns True (e.g., goal satisfied)
        2. max_iterations reached
        3. Too many consecutive waits (agent is stuck)
        """
        self._running = True
        iteration = 0
        consecutive_waits = 0
        max_consecutive_waits = 5  # Stop if agent waits 5 times in a row
        
        while self._running and iteration < max_iterations:
            iteration += 1
            
            # Check stop condition (e.g., goal satisfied)
            if stop_condition and stop_condition(self.blackboard):
                self.communicate("agent_stopped", {"reason": "Goal satisfied"})
                break
            
            # Execute one step
            try:
                result = self.step()
                
                # Track consecutive waits
                if result.get("plan", {}).get("action") == "wait":
                    consecutive_waits += 1
                    time.sleep(0.5)
                    
                    # Stop if stuck waiting
                    if consecutive_waits >= max_consecutive_waits:
                        self.communicate("agent_stopped", {"reason": "No work available (stuck waiting)"})
                        break
                else:
                    consecutive_waits = 0  # Reset on real action
                    
            except Exception as e:
                self.communicate("error", {"error": str(e)})
                time.sleep(1)
        
        # Log why we stopped
        if iteration >= max_iterations:
            self.communicate("agent_stopped", {"reason": f"Max iterations ({max_iterations}) reached"})
        
        self._running = False
        return self.state.action_history
    
    def start_async(self, max_iterations: int = 10, stop_condition: callable = None):
        """Start the agent in a separate thread (for true async execution)."""
        self._thread = threading.Thread(
            target=self.run_loop,
            args=(max_iterations, stop_condition)
        )
        self._thread.start()
    
    def stop(self):
        """Stop the agent's execution loop."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)


class CodeAnalyzerAgent(AutonomousAgent):
    """Autonomous agent that analyzes codebases.
    
    LOCAL PLANNING: Decides when and what to analyze based on its own beliefs.
    Can generate tasks for other agents (e.g., "code needs testing").
    """
    
    def __init__(self, blackboard: Blackboard, base_url: str = "http://localhost:11434/v1"):
        super().__init__("CodeAnalyzer", blackboard, base_url)
        self.state.goals = ["Understand the codebase", "Identify relevant files"]
    
    def plan(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Decide what analysis action to take.
        
        This agent uses LLM to decide:
        - WHAT to analyze
        - WHETHER to analyze now
        - WHAT tasks to generate for others
        """
        shared = observation["shared_data"]
        
        system_prompt = """You are an autonomous CodeAnalyzer agent in a decentralized system.
You must decide what to do next based on your observations.

You can:
1. "analyze_code" - Analyze the codebase structure
2. "generate_task" - Create a task for another agent
3. "wait" - Do nothing if there's no useful work

Respond with JSON:
{
  "action": "analyze_code" or "generate_task" or "wait",
  "reason": "why this action",
  "details": {}
}"""
        
        user_prompt = f"""Current situation:
{observation['blackboard_summary']}

My beliefs: {self.state.beliefs}
My goals: {self.state.goals}
My last action: {self.state.last_action}

What should I do next? Consider:
- Has code been analyzed yet?
- Is there new code that needs analysis?
- Would analysis help other agents?

Respond with JSON only."""
        
        response = self.call_model(system_prompt, user_prompt)
        
        try:
            if '{' in response:
                start = response.index('{')
                end = response.rindex('}') + 1
                return json.loads(response[start:end])
        except Exception:
            pass
        
        # Default: check if analysis is needed
        if not shared.get("code_analysis"):
            return {"action": "analyze_code", "reason": "No analysis exists yet", "details": {}}
        return {"action": "wait", "reason": "Analysis already done", "details": {}}
    
    def act(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the planned action."""
        action = plan.get("action")
        
        if action == "analyze_code":
            return self._do_analysis()
        elif action == "generate_task":
            return self._generate_task(plan.get("details", {}))
        else:
            return {"status": "skipped", "reason": plan.get("reason")}
    
    def _do_analysis(self) -> Dict[str, Any]:
        """Perform code analysis."""
        files = tools.list_files(".")
        
        def is_ignored(p: str) -> bool:
            lp = p.lower()
            ign = ['env/', 'venv/', '.venv/', 'site-packages/', '__pycache__', '.git/', 'node_modules/']
            return any(i in lp for i in ign)
        
        py_files = [f for f in files if f.endswith('.py') and not is_ignored(f)]
        test_files = [f for f in py_files if 'test' in f.lower()]
        src_files = [f for f in py_files if f not in test_files]
        
        analysis = {
            "total_files": len(files),
            "python_files": len(py_files),
            "test_files": test_files[:20],
            "source_files": src_files[:20],
        }
        
        # Update shared state
        self.blackboard.update_shared("code_analysis", analysis)
        
        # Update local beliefs
        self.state.beliefs["codebase_analyzed"] = True
        self.state.beliefs["has_tests"] = len(test_files) > 0
        
        # Generate task for TestRunner if tests exist
        if test_files:
            self.communicate("task_available", {
                "task_type": "run_tests",
                "reason": "Code analyzed, tests should be run",
                "suggested_for": "TestRunner"
            })
        
        return {"status": "success", "analysis": analysis}
    
    def _generate_task(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a task for another agent."""
        self.communicate("task_available", details)
        return {"status": "task_generated", "details": details}


class TestRunnerAgent(AutonomousAgent):
    """Autonomous agent that runs tests.
    
    LOCAL PLANNING: Decides when to run tests based on its own assessment.
    Generates tasks based on test results (e.g., "bugs need fixing").
    """
    
    def __init__(self, blackboard: Blackboard, base_url: str = "http://localhost:11434/v1"):
        super().__init__("TestRunner", blackboard, base_url)
        self.state.goals = ["Keep tests passing", "Detect regressions"]
    
    def plan(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Decide whether to run tests."""
        shared = observation["shared_data"]
        messages = observation["recent_messages"]
        
        system_prompt = """You are an autonomous TestRunner agent in a decentralized system.
You decide when to run tests based on your observations.

You can:
1. "run_tests" - Execute the test suite
2. "generate_task" - Create a task for another agent (e.g., request debugging)
3. "wait" - Do nothing if tests don't need to run

Respond with JSON:
{
  "action": "run_tests" or "generate_task" or "wait",
  "reason": "why this action",
  "details": {}
}"""
        
        user_prompt = f"""Current situation:
{observation['blackboard_summary']}

Recent messages from other agents:
{json.dumps(messages[-5:], indent=2, default=str)}

My beliefs: {self.state.beliefs}
My last action: {self.state.last_action}

What should I do? Consider:
- Have tests been run yet?
- Was code recently modified?
- Did another agent request tests?

Respond with JSON only."""
        
        response = self.call_model(system_prompt, user_prompt)
        
        try:
            if '{' in response:
                start = response.index('{')
                end = response.rindex('}') + 1
                return json.loads(response[start:end])
        except Exception:
            pass
        
        # Default logic
        test_results = shared.get("test_results")
        if not test_results:
            return {"action": "run_tests", "reason": "No test results yet", "details": {}}
        
        # Check if code was modified
        modified = shared.get("modified_files", [])
        if modified and self.state.beliefs.get("last_test_files") != modified:
            return {"action": "run_tests", "reason": "Code was modified", "details": {}}
        
        return {"action": "wait", "reason": "Tests already run, no changes", "details": {}}
    
    def act(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the planned action."""
        action = plan.get("action")
        
        if action == "run_tests":
            return self._run_tests()
        elif action == "generate_task":
            return self._generate_task(plan.get("details", {}))
        else:
            return {"status": "skipped", "reason": plan.get("reason")}
    
    def _run_tests(self) -> Dict[str, Any]:
        """Run the test suite."""
        result = tools.run_tests("pytest -q")
        
        test_results = {
            "total_tests": int(result.get("total_tests", 0)),
            "passed": int(result.get("passed", 0)),
            "failed": int(result.get("failed", 0)),
            "failed_test_names": result.get("failed_test_names", []),
            "raw": result.get("raw", ""),
        }
        
        # Update shared state
        self.blackboard.update_shared("test_results", test_results)
        
        # Update local beliefs
        self.state.beliefs["tests_run"] = True
        self.state.beliefs["tests_passing"] = test_results["failed"] == 0
        self.state.beliefs["last_test_files"] = self.blackboard.get_shared("modified_files", [])
        
        # Generate tasks based on results
        if test_results["failed"] > 0:
            self.communicate("task_available", {
                "task_type": "debug_failures",
                "reason": f"{test_results['failed']} tests failing",
                "suggested_for": "Debugger",
                "failed_tests": test_results["failed_test_names"],
                "raw_output": test_results["raw"][:2000]
            })
        else:
            self.communicate("goal_achieved", {
                "goal": "All tests passing",
                "details": test_results
            })
        
        return {"status": "success", "test_results": test_results}
    
    def _generate_task(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a task for another agent."""
        self.communicate("task_available", details)
        return {"status": "task_generated", "details": details}


class DebuggerAgent(AutonomousAgent):
    """Autonomous agent that debugs test failures.
    
    LOCAL PLANNING: Observes test results and decides whether to investigate.
    Generates fix suggestions as tasks for CodeWriter.
    """
    
    def __init__(self, blackboard: Blackboard, base_url: str = "http://localhost:11434/v1"):
        super().__init__("Debugger", blackboard, base_url)
        self.state.goals = ["Diagnose test failures", "Suggest fixes"]
    
    def plan(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Decide whether to debug."""
        shared = observation["shared_data"]
        messages = observation["recent_messages"]
        
        # Look for debug tasks
        debug_tasks = [m for m in messages 
                      if m.get("msg_type") == "task_available" 
                      and m.get("content", {}).get("task_type") == "debug_failures"
                      and m.get("content", {}).get("suggested_for") == "Debugger"]
        
        system_prompt = """You are an autonomous Debugger agent in a decentralized system.
You decide when to analyze failures and suggest fixes.

You can:
1. "debug" - Analyze test failures and identify bugs
2. "generate_fix_task" - Create a fix task for CodeWriter
3. "wait" - Do nothing if there's nothing to debug

Respond with JSON:
{
  "action": "debug" or "generate_fix_task" or "wait",
  "reason": "why this action",
  "details": {}
}"""
        
        user_prompt = f"""Current situation:
{observation['blackboard_summary']}

Debug tasks waiting: {len(debug_tasks)}
Recent debug task: {debug_tasks[-1] if debug_tasks else 'None'}

My beliefs: {self.state.beliefs}
My commitments: {self.state.commitments}

What should I do? Consider:
- Are there failing tests to debug?
- Have I already analyzed these failures?
- Are there pending fixes I suggested?

Respond with JSON only."""
        
        response = self.call_model(system_prompt, user_prompt)
        
        try:
            if '{' in response:
                start = response.index('{')
                end = response.rindex('}') + 1
                return json.loads(response[start:end])
        except Exception:
            pass
        
        # Default logic
        test_results = shared.get("test_results", {})
        if test_results.get("failed", 0) > 0 and not self.state.beliefs.get("fixes_suggested"):
            return {"action": "debug", "reason": "Tests failing, need to diagnose", "details": {}}
        
        return {"action": "wait", "reason": "No failures to debug", "details": {}}
    
    def act(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the planned action."""
        action = plan.get("action")
        
        if action == "debug":
            return self._debug_failures()
        elif action == "generate_fix_task":
            return self._generate_fix_task(plan.get("details", {}))
        else:
            return {"status": "skipped", "reason": plan.get("reason")}
    
    def _debug_failures(self) -> Dict[str, Any]:
        """Analyze test failures and suggest fixes."""
        test_results = self.blackboard.get_shared("test_results", {})
        raw = test_results.get("raw", "")
        
        if not raw:
            return {"status": "error", "reason": "No test output to analyze"}
        
        # Extract file mentions from test output
        import re
        file_mentions = re.findall(r'File "?([^":\s]+\.py)"?, line (\d+)', raw)
        file_mentions += re.findall(r'(\w+\.py):(\d+):', raw)
        
        source_files = list(set(f[0] for f in file_mentions if not f[0].startswith('test_')))
        
        # Read source files
        file_contents = {}
        for fpath in source_files[:3]:
            try:
                file_contents[fpath] = tools.read_file(fpath)
            except Exception:
                pass
        
        # Use LLM to analyze
        system_prompt = """Analyze test failures and identify bugs.

Return JSON:
{
  "analysis": "root cause description",
  "fixes": [
    {"file": "path.py", "reason": "what to fix", "line": 50}
  ]
}"""
        
        user_prompt = f"""TEST OUTPUT:
{raw[:3000]}

SOURCE FILES:
{json.dumps({k: v[:1500] for k, v in file_contents.items()}, indent=2)}

Identify the bugs and suggest specific fixes."""
        
        response = self.call_model(system_prompt, user_prompt)
        
        try:
            if '{' in response:
                start = response.index('{')
                end = response.rindex('}') + 1
                result = json.loads(response[start:end])
            else:
                result = {"analysis": "Could not parse", "fixes": []}
        except Exception:
            result = {"analysis": "Parse error", "fixes": []}
        
        # Update local beliefs
        self.state.beliefs["fixes_suggested"] = True
        self.state.beliefs["last_analysis"] = result.get("analysis", "")
        
        # Generate fix tasks for CodeWriter
        for fix in result.get("fixes", []):
            self.communicate("task_available", {
                "task_type": "apply_fix",
                "suggested_for": "CodeWriter",
                "file": fix.get("file"),
                "reason": fix.get("reason"),
                "line": fix.get("line")
            })
        
        return {"status": "success", "analysis": result}
    
    def _generate_fix_task(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a fix task."""
        self.communicate("task_available", details)
        return {"status": "task_generated", "details": details}


class CodeWriterAgent(AutonomousAgent):
    """Autonomous agent that applies code fixes.
    
    LOCAL PLANNING: Observes fix suggestions and decides what to apply.
    Notifies other agents when code is modified.
    """
    
    def __init__(self, blackboard: Blackboard, base_url: str = "http://localhost:11434/v1"):
        super().__init__("CodeWriter", blackboard, base_url)
        self.state.goals = ["Apply valid fixes", "Maintain code quality"]
    
    def plan(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Decide whether to apply fixes."""
        messages = observation["recent_messages"]
        
        # Look for fix tasks
        fix_tasks = [m for m in messages 
                    if m.get("msg_type") == "task_available" 
                    and m.get("content", {}).get("task_type") == "apply_fix"
                    and m.get("content", {}).get("suggested_for") == "CodeWriter"]
        
        # Filter out tasks we've already done
        pending_fixes = []
        for task in fix_tasks:
            task_id = task.get("id")
            if task_id not in self.state.commitments:
                pending_fixes.append(task)
        
        system_prompt = """You are an autonomous CodeWriter agent in a decentralized system.
You decide when to apply code fixes.

You can:
1. "apply_fix" - Apply a suggested fix to a file
2. "wait" - Do nothing if there are no valid fixes

Respond with JSON:
{
  "action": "apply_fix" or "wait",
  "reason": "why this action",
  "details": {"fix_task_id": "...", "file": "...", "reason": "..."}
}"""
        
        user_prompt = f"""Current situation:
{observation['blackboard_summary']}

Pending fix tasks: {len(pending_fixes)}
Fix details: {[t.get('content') for t in pending_fixes[:3]]}

My beliefs: {self.state.beliefs}
Tasks I've completed: {len(self.state.commitments)}

What should I do?"""
        
        response = self.call_model(system_prompt, user_prompt)
        
        try:
            if '{' in response:
                start = response.index('{')
                end = response.rindex('}') + 1
                return json.loads(response[start:end])
        except Exception:
            pass
        
        # Default: apply first pending fix
        if pending_fixes:
            fix = pending_fixes[0]
            return {
                "action": "apply_fix",
                "reason": "Fix task available",
                "details": {
                    "fix_task_id": fix.get("id"),
                    "file": fix.get("content", {}).get("file"),
                    "reason": fix.get("content", {}).get("reason")
                }
            }
        
        return {"action": "wait", "reason": "No pending fixes", "details": {}}
    
    def act(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the planned action."""
        action = plan.get("action")
        
        if action == "apply_fix":
            return self._apply_fix(plan.get("details", {}))
        else:
            return {"status": "skipped", "reason": plan.get("reason")}
    
    def _apply_fix(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a code fix."""
        filepath = details.get("file")
        reason = details.get("reason")
        task_id = details.get("fix_task_id")
        
        if not filepath or not reason:
            return {"status": "error", "reason": "Missing file or reason"}
        
        try:
            current_content = tools.read_file(filepath)
        except Exception as e:
            return {"status": "error", "reason": f"Could not read {filepath}: {e}"}
        
        # Use LLM to generate fix
        system_prompt = """Fix the bug in this file. 
Output ONLY the complete corrected Python code, no explanations."""
        
        user_prompt = f"""FILE: {filepath}
CURRENT CONTENT:
```python
{current_content}
```

BUG TO FIX: {reason}

Output the complete corrected file:"""
        
        fixed_content = self.call_model(system_prompt, user_prompt)
        
        # Clean up
        fixed_content = fixed_content.strip()
        if fixed_content.startswith('```python'):
            fixed_content = fixed_content[len('```python'):].strip()
        if fixed_content.startswith('```'):
            fixed_content = fixed_content[3:].strip()
        if fixed_content.endswith('```'):
            fixed_content = fixed_content[:-3].strip()
        
        if not fixed_content or len(fixed_content) < 10:
            return {"status": "error", "reason": "Invalid fix generated"}
        
        # Apply the fix
        tools.apply_patch(filepath, fixed_content)
        
        # Update shared state
        modified = self.blackboard.get_shared("modified_files", [])
        if filepath not in modified:
            modified.append(filepath)
        self.blackboard.update_shared("modified_files", modified)
        
        # Mark task as done
        if task_id:
            self.state.commitments.append(task_id)
        
        # Notify others that code changed
        self.communicate("code_modified", {
            "file": filepath,
            "reason": reason
        })
        
        # Reset debugger's beliefs (they need to re-analyze if tests fail)
        self.communicate("task_available", {
            "task_type": "run_tests",
            "reason": f"Code modified in {filepath}",
            "suggested_for": "TestRunner"
        })
        
        return {"status": "success", "file": filepath}


class PerformanceAnalyzerAgent(AutonomousAgent):
    """Autonomous agent that analyzes performance.
    
    Only activates when explicitly requested or all tests pass.
    """
    
    def __init__(self, blackboard: Blackboard, base_url: str = "http://localhost:11434/v1"):
        super().__init__("PerformanceAnalyzer", blackboard, base_url)
        self.state.goals = ["Optimize code when tests pass"]
    
    def plan(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Decide whether to analyze performance."""
        shared = observation["shared_data"]
        user_goal = shared.get("user_goal", "").lower()
        
        # Only act if explicitly requested or all tests pass
        test_results = shared.get("test_results", {})
        tests_pass = test_results.get("failed", 1) == 0 and test_results.get("total_tests", 0) > 0
        
        if "performance" in user_goal or "optimize" in user_goal:
            if tests_pass:
                return {"action": "analyze", "reason": "Performance requested and tests pass", "details": {}}
        
        return {"action": "wait", "reason": "Performance analysis not needed", "details": {}}
    
    def act(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the planned action."""
        if plan.get("action") == "analyze":
            # Simplified: just log that we would analyze
            self.communicate("performance_analyzed", {
                "status": "completed",
                "note": "Performance analysis done"
            })
            return {"status": "success", "note": "Would analyze performance here"}
        
        return {"status": "skipped", "reason": plan.get("reason")}
