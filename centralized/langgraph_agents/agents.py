"""Agent implementations for the centralized multi-agent system.

CANONICAL CENTRALIZED PLANNER (Manager-Worker Architecture)
===========================================================
This implementation follows the canonical centralized planning pattern:

1. Single Planning Authority: PlannerAgent produces a COMPLETE task plan
   in ONE LLM call (goal decomposition + task sequencing)
2. Stateless Workers: Workers execute assigned tasks without re-planning
3. Linear Plan Representation: Plan is a simple task list
4. Synchronous Control Flow: Sequential execution with planner aggregation

Key discriminator: The planner decides WHAT should be done and in WHAT ORDER
upfront, not incrementally.
"""
from typing import Dict, Any, List
import os
import json

try:
    from langchain.chat_models import ChatOpenAI
    from langchain.schema import HumanMessage, SystemMessage
except Exception:
    # Fallback shim if langchain's ChatOpenAI is not available in this environment.
    import requests

    class HumanMessage:
        def __init__(self, content: str):
            self.content = content

    class SystemMessage:
        def __init__(self, content: str):
            self.content = content

    class ChatOpenAI:
        def __init__(self, model: str = "qwen2.5:7b", temperature: float = 0.2, openai_api_base: str = "http://localhost:11434/v1"):
            self.model = model
            self.temperature = temperature
            self.base = openai_api_base.rstrip("/")

        def __call__(self, messages):
            # messages: list of SystemMessage/HumanMessage
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
            # Create a simple response object with .content
            try:
                content = j["choices"][0]["message"]["content"]
            except Exception:
                content = str(j)

            class Resp:
                def __init__(self, content):
                    self.content = content

            return Resp(content)

from . import tools

# Local-only model settings
LLM_SETTINGS = {
    "model": "qwen2.5:7b",
    "temperature": 0.2,
}


class BaseAgent:
    """Base class for all agents with LLM communication."""
    
    def __init__(self, system_prompt: str, name: str = "agent", base_url: str = "http://localhost:11434/v1"):
        self.name = name
        self.system_prompt = system_prompt
        # Configure ChatOpenAI to use local OpenAI-compatible endpoint
        self.client = ChatOpenAI(model=LLM_SETTINGS["model"], temperature=LLM_SETTINGS["temperature"], openai_api_base=base_url)

    def call_model(self, user_prompt: str) -> str:
        messages = [SystemMessage(content=self.system_prompt), HumanMessage(content=user_prompt)]
        resp = self.client(messages)
        # LangChain ChatOpenAI returns a ChatGeneration; pick content
        try:
            return resp.content
        except Exception:
            # fallback
            return str(resp)


class PlannerAgent(BaseAgent):
    """Central Planner: The ONLY agent that creates plans.
    
    CANONICAL BEHAVIOR:
    - Produces a COMPLETE task plan in ONE LLM call
    - Workers execute tasks without modifying the plan
    - Replanning ONLY occurs if execution fails unexpectedly
    
    Plan format: List of tasks like ["testrunner", "debug", "writer", "testrunner"]
    """

    def __init__(self, base_url: str = "http://localhost:11434/v1"):
        super().__init__(
            system_prompt="""You are PlannerAgent: the CENTRAL PLANNER for a multi-agent coding system.

YOUR ROLE: Create a COMPLETE execution plan UPFRONT in a single response.

Available worker actions (in order of typical use):
- "analyze": CodeAnalyzer examines the codebase structure
- "testrunner": Run tests to see current status  
- "debug": Debugger analyzes failures and creates fix instructions
- "writer": CodeWriter applies fixes to code
- "done": Signal that work is complete

IMPORTANT RULES:
1. You create the ENTIRE plan in ONE response
2. Workers execute your plan - they cannot modify it
3. Your plan should be a logical sequence to achieve the goal
4. Always start with "testrunner" to assess current state
5. For fixing bugs: testrunner -> debug -> writer -> testrunner
6. End with "done" when you expect success

You MUST respond with valid JSON:
{
  "mode": "debug" or "performance" or "feature",
  "plan": ["testrunner", "debug", "writer", "testrunner", "done"],
  "planner_notes": "brief explanation of your plan"
}""",
            name="Planner",
            base_url=base_url
        )

    def create_plan(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate the COMPLETE execution plan in ONE LLM call.
        
        This is the canonical centralized planner behavior:
        - One LLM call produces the full plan
        - Only one agent (this one) decides task ordering
        """
        user_request = state.get("user_request", "Fix failing tests")
        max_iterations = state.get("max_iterations", 8)
        
        # Get initial context if available
        files_info = ""
        try:
            files = tools.list_files(".")
            py_files = [f for f in files if f.endswith('.py') and 'env/' not in f and '__pycache__' not in f]
            files_info = f"Python files in repo: {py_files[:20]}"
        except Exception:
            files_info = "Could not list files"
        
        prompt = f"""Create a COMPLETE execution plan for this request:

USER REQUEST: {user_request}
MAX STEPS ALLOWED: {max_iterations}
CODEBASE INFO: {files_info}

Based on the request, create a full plan. Common patterns:

For "Fix failing tests" or bug fixing:
  ["testrunner", "debug", "writer", "testrunner", "done"]
  
For "Optimize performance":
  ["testrunner", "analyze", "profile", "writer", "testrunner", "done"]

For "Add new feature":
  ["analyze", "writer", "testrunner", "done"]

Create your plan now. Respond with JSON only:
{{"mode": "...", "plan": [...], "planner_notes": "..."}}"""

        response = self.call_model(prompt)
        
        # Parse JSON from response
        try:
            if '{' in response:
                start = response.index('{')
                end = response.rindex('}') + 1
                result = json.loads(response[start:end])
            else:
                # Default plan for bug fixing
                result = {
                    "mode": "debug",
                    "plan": ["testrunner", "debug", "writer", "testrunner", "done"],
                    "planner_notes": "Default bug-fixing plan (LLM response parsing failed)"
                }
        except Exception as e:
            result = {
                "mode": "debug", 
                "plan": ["testrunner", "debug", "writer", "testrunner", "done"],
                "planner_notes": f"JSON parse error, using default plan: {e}"
            }
        
        # Validate plan
        valid_actions = {"analyze", "testrunner", "debug", "writer", "profile", "done"}
        plan = result.get("plan", [])
        plan = [a for a in plan if a in valid_actions]
        
        if not plan:
            plan = ["testrunner", "debug", "writer", "testrunner", "done"]
        
        # Ensure plan doesn't exceed max_iterations
        if len(plan) > max_iterations:
            plan = plan[:max_iterations-1] + ["done"]
        
        return {
            "mode": result.get("mode", "debug"),
            "plan": plan,
            "plan_index": 0,  # Current position in plan
            "planner_notes": result.get("planner_notes", ""),
            "original_plan": plan.copy(),  # Keep original for logging
        }

    def should_replan(self, state: Dict[str, Any]) -> bool:
        """Check if replanning is needed (only on unexpected failures).
        
        Replanning happens ONLY in the planner, and only when:
        - Writer made changes but tests still fail
        - We've tried the same fix multiple times
        """
        replan_count = state.get("replan_count", 0)
        max_replans = 2  # Limit replanning attempts
        
        if replan_count >= max_replans:
            return False
        
        # Check if we just ran tests after a write and tests still fail
        last_test_run = state.get("last_test_run", {})
        last_writer_change = state.get("last_writer_change", False)
        failed = last_test_run.get("failed", 0)
        
        # Replan if writer made changes but tests still fail
        if last_writer_change and failed > 0:
            return True
        
        return False

    def replan(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Create a NEW plan when the original plan didn't work.
        
        This is the ONLY place where replanning occurs (in the planner).
        """
        user_request = state.get("user_request", "Fix failing tests")
        last_test_run = state.get("last_test_run", {})
        replan_count = state.get("replan_count", 0) + 1
        previous_plan = state.get("original_plan", [])
        
        prompt = f"""Your previous plan did not fully succeed. Create a NEW plan.

USER REQUEST: {user_request}
PREVIOUS PLAN: {previous_plan}
REPLAN ATTEMPT: {replan_count}

CURRENT TEST STATUS:
- Total: {last_test_run.get('total_tests', 0)}
- Failed: {last_test_run.get('failed', 0)}
- Failing tests: {last_test_run.get('failed_test_names', [])}

Create a new plan to fix remaining issues. Try a different approach.
Respond with JSON only:
{{"mode": "...", "plan": [...], "planner_notes": "..."}}"""

        response = self.call_model(prompt)
        
        try:
            if '{' in response:
                start = response.index('{')
                end = response.rindex('}') + 1
                result = json.loads(response[start:end])
                plan = result.get("plan", ["debug", "writer", "testrunner", "done"])
            else:
                plan = ["debug", "writer", "testrunner", "done"]
                result = {"mode": "debug", "planner_notes": "Replan with default"}
        except Exception:
            plan = ["debug", "writer", "testrunner", "done"]
            result = {"mode": "debug", "planner_notes": "Replan parse error"}
        
        return {
            "mode": result.get("mode", "debug"),
            "plan": plan,
            "plan_index": 0,
            "planner_notes": result.get("planner_notes", ""),
            "original_plan": plan.copy(),
            "replan_count": replan_count,
        }


class CodeAnalyzerAgent(BaseAgent):
    """STATELESS WORKER: Analyzes codebase structure.
    
    This worker:
    - Executes the assigned analysis task
    - Returns results to the planner
    - Does NOT modify the global plan
    - Does NOT split/reassign/reject tasks
    """
    
    def __init__(self, base_url: str = "http://localhost:11434/v1"):
        super().__init__(system_prompt="You are CodeAnalyzer: produce a JSON with keys 'summary' and 'files' listing key files to inspect.", name="CodeAnalyzer", base_url=base_url)

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # Simple analysis: list files and read a few
        files = tools.list_files(".")

        # Filter out virtualenvs, caches and third-party packages which introduce noise
        def is_ignored(p: str) -> bool:
            lp = p.lower()
            ign = ['env/', 'venv/', '.venv/', 'site-packages/', '__pycache__', '.git/', 'node_modules/']
            for i in ign:
                if lp.startswith(i) or ('/' + i) in ('/' + lp):
                    return True
            # also ignore files inside env/ path fragments
            if '/env/' in lp or '/venv/' in lp:
                return True
            return False

        py_files = [f for f in files if f.endswith('.py') and not is_ignored(f)]

        # Prefer test files and src-like files
        priority = [f for f in py_files if '/tests/' in f or f.startswith('tests/') or f.endswith('_test.py') or f.endswith('test.py')]
        srcs = [f for f in py_files if f.startswith('src/') or '/src/' in f or f.startswith('lib/')]
        top_level = [f for f in py_files if '/' not in f]

        # Compose key_files with priority order, limit to 30
        key_files = []
        for lst in (priority, srcs, top_level, py_files):
            for f in lst:
                if f not in key_files:
                    key_files.append(f)
                if len(key_files) >= 30:
                    break
            if len(key_files) >= 30:
                break

        snippets = {}
        for f in key_files[:10]:
            try:
                snippets[f] = tools.read_file(f)[:2000]
            except Exception:
                snippets[f] = "<could not read>"

        return {"analysis_summary": f"Found {len(files)} files, {len(py_files)} python files after filtering.", "files": key_files, "snippets": snippets}


class CodeWriterAgent(BaseAgent):
    """STATELESS WORKER: Applies code fixes based on instructions.
    
    This worker:
    - Executes the assigned fix task
    - Writes corrected code to files
    - Does NOT modify the global plan
    - Does NOT decide what to fix next (that's the planner's job)
    """
    
    def __init__(self, base_url: str = "http://localhost:11434/v1"):
        super().__init__(system_prompt="""You are CodeWriter: an expert at fixing bugs in code.

IMPORTANT RULES:
1. NEVER modify test files (files starting with 'test_' or ending with '_test.py')
2. Test files define the expected behavior - they are the specification
3. Your job is to fix the SOURCE code to make the tests pass, not to change the tests
4. If a test is failing, the bug is in the source code, not in the test

Given a file and instructions about what bugs to fix, you will:
1. Read and understand the entire file
2. Identify the buggy lines based on the instructions
3. Generate a corrected version of the ENTIRE file
4. Return ONLY the fixed file content, nothing else

Your output should be the complete, corrected file content ready to be written.""", 
        name="CodeWriter", base_url=base_url)

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # Get fix instructions from debugger
        fix_instructions = state.get("fix_instructions", [])
        patched = []
        notes = ""
        
        if not fix_instructions:
            notes = "No fix instructions provided by debugger"
            return {"patched_files": patched, "notes": notes, "last_writer_change": False}
        
        # Group instructions by file
        files_to_fix = {}
        for instr in fix_instructions:
            filepath = instr.get("file")
            if filepath:
                if filepath not in files_to_fix:
                    files_to_fix[filepath] = []
                files_to_fix[filepath].append(instr)
        
        # Process each file
        for filepath, instructions in files_to_fix.items():
            try:
                # Read current file content
                current_content = tools.read_file(filepath)
                
                # Build prompt for LLM
                prompt = f"""Fix the bugs in this file based on the instructions below.

FILE: {filepath}
CURRENT CONTENT:
```python
{current_content}
```

BUG FIX INSTRUCTIONS:
"""
                for idx, instr in enumerate(instructions, 1):
                    prompt += f"\n{idx}. {instr.get('reason', 'Fix bug')}"
                    if 'line' in instr:
                        prompt += f" (around line {instr.get('line')})"
                
                prompt += """

Generate the COMPLETE corrected file content. Output ONLY the Python code, no explanations, no markdown fences.
Keep all existing code structure, imports, and comments. Only fix the specific bugs mentioned above."""
                
                # Call LLM to generate fixed content
                fixed_content = self.call_model(prompt)
                
                # Clean up response (remove markdown fences if present)
                fixed_content = fixed_content.strip()
                if fixed_content.startswith('```python'):
                    fixed_content = fixed_content[len('```python'):].strip()
                if fixed_content.startswith('```'):
                    fixed_content = fixed_content[3:].strip()
                if fixed_content.endswith('```'):
                    fixed_content = fixed_content[:-3].strip()
                
                # Verify it's valid Python (basic check)
                if not fixed_content or len(fixed_content) < 10:
                    notes += f"LLM generated invalid/empty content for {filepath}. "
                    continue
                
                # Write the fixed file
                tools.apply_patch(filepath, fixed_content)
                patched.append(filepath)
                notes += f"✓ Fixed {filepath} using LLM-directed editing. "
            
            except Exception as e:
                notes += f"Error fixing {filepath}: {e}. "
        
        return {
            "patched_files": patched,
            "notes": notes.strip(),
            "last_writer_change": bool(patched)
        }


class TestRunnerAgent(BaseAgent):
    """STATELESS WORKER: Runs tests and reports results.
    
    This worker:
    - Executes the test suite
    - Returns structured test results
    - Does NOT modify the global plan
    - Does NOT decide what to do if tests fail (that's the planner's job)
    """
    
    def __init__(self, base_url: str = "http://localhost:11434/v1"):
        super().__init__(system_prompt="You are TestRunner: run the project's tests and return structured JSON exactly as required.", name="TestRunner", base_url=base_url)

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        cmd = state.get("test_command", "pytest -q")
        res = tools.run_tests(cmd)
        # Ensure the dictionary matches the required shape
        structured = {
            "total_tests": int(res.get("total_tests", 0)),
            "passed": int(res.get("passed", 0)),
            "failed": int(res.get("failed", 0)),
            "failed_test_names": res.get("failed_test_names", []),
            "raw": res.get("raw", ""),
            "returncode": res.get("returncode", 1),
        }
        return {"last_test_run": structured}


class DebuggerAgent(BaseAgent):
    """STATELESS WORKER: Analyzes test failures and creates fix instructions.
    
    This worker:
    - Analyzes test output to identify bugs
    - Creates fix instructions for CodeWriter
    - Does NOT modify the global plan
    - Does NOT execute fixes (that's CodeWriter's job)
    """
    
    def __init__(self, base_url: str = "http://localhost:11434/v1"):
        super().__init__(system_prompt="""You are DebuggerAgent: an expert at analyzing test failures and diagnosing bugs.

IMPORTANT RULES:
1. NEVER suggest modifying test files (files starting with 'test_' or ending with '_test.py')
2. Test files define the expected behavior - they are the specification
3. The bug is ALWAYS in the source code, not in the tests
4. Your fix_instructions must ONLY target source files, never test files

Given test output with failures/errors, you must:
1. Read the relevant source files to understand the bug
2. Analyze the error messages and tracebacks
3. Return a JSON dict with:
   - "analysis": brief description of the root cause
   - "fix_instructions": list of dicts, each with:
     * "file": path to SOURCE file (never test files!)
     * "reason": clear description of what bug to fix (e.g., "Remove the + 1.0 from subtotal calculation")
     * "line": approximate line number (optional)

Be clear and specific about WHAT is wrong, not HOW to fix the syntax.""", name="Debugger", base_url=base_url)

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        tr = state.get("last_test_run", {})
        raw = tr.get("raw", "")
        failed = tr.get("failed", 0)
        
        if failed == 0:
            return {"analysis": "No failures to debug", "fix_instructions": []}
        
        # Extract file paths from test output
        import re
        file_mentions = re.findall(r'File "?([^":\s]+\.py)"?, line (\d+)', raw)
        # Also look for simpler format: cart.py:14
        file_mentions += re.findall(r'(\w+\.py):(\d+):', raw)
        
        # Get unique files - separate source files and test files
        source_files = list(set(f[0] for f in file_mentions if not f[0].startswith('test_')))
        test_files = list(set(f[0] for f in file_mentions if f[0].startswith('test_')))
        
        # Read the source files AND test files (full content, not truncated)
        file_contents = {}
        for fpath in source_files[:3]:  # Limit to 3 source files
            try:
                file_contents[fpath] = tools.read_file(fpath)  # Read FULL file
            except Exception:
                pass
        
        # Also read test files for context
        for fpath in test_files[:2]:  # Include up to 2 test files
            try:
                file_contents[fpath] = tools.read_file(fpath)
            except Exception:
                pass
        
        # Build prompt for LLM
        prompt = f"""Analyze these test failures and identify the bugs.

FAILED TESTS: {failed}
TEST OUTPUT (errors/tracebacks):
{raw[:4000]}

SOURCE FILES:
"""
        for fpath, content in file_contents.items():
            prompt += f"\n=== {fpath} ===\n{content}\n"  # Include FULL content, no truncation
        
        prompt += """

Provide a JSON response with:
{
  "analysis": "brief root cause description",
  "fix_instructions": [
    {
      "file": "path/to/file.py",
      "reason": "Clear description of the bug to fix (e.g., 'The subtotal method adds 1.0 which should be removed' or 'Tax rate is multiplied by 1.1 but should not be')",
      "line": 50
    }
  ]
}

IMPORTANT:
- Focus on WHAT is wrong, not exact code patterns
- Be specific about the bug (e.g., "remove extra addition", "change subtraction to addition", "fix validation to reject negative values")
- Identify ALL bugs you can find, even if tests don't catch them yet
- Each instruction should describe ONE specific bug clearly
- Fix the implementation bugs, not the tests

Example good instructions:
- "Remove the + 1.0 that's incorrectly added to the subtotal return value"
- "Change the tax_rate multiplication by 1.1 to just use tax_rate directly"
- "Fix add_item to add quantity instead of subtracting it"
- "Change validation to reject values < 0 instead of < -10"
"""
        
        # Call LLM
        response = self.call_model(prompt)
        
        # Parse JSON from response
        import json
        try:
            # Try to extract JSON from response
            if '{' in response:
                start = response.index('{')
                end = response.rindex('}') + 1
                result = json.loads(response[start:end])
            else:
                result = {"analysis": "Could not parse LLM response", "fix_instructions": []}
        except Exception as e:
            result = {"analysis": f"Error parsing LLM response: {e}", "fix_instructions": []}
        
        return result


class PerformanceAnalyzerAgent(BaseAgent):
    """STATELESS WORKER: Profiles code performance.
    
    This worker:
    - Profiles specified functions
    - Returns performance metrics
    - Does NOT modify the global plan
    - Does NOT decide optimization strategy (that's the planner's job)
    """
    
    def __init__(self, base_url: str = "http://localhost:11434/v1"):
        super().__init__(system_prompt="You are PerformanceAnalyzer: profile a named function and return 'profile' summary.", name="PerformanceAnalyzer", base_url=base_url)

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        fp = state.get("profile_target_path")
        fn = state.get("profile_target_name")
        if fp and fn:
            summary = tools.profile_function(fp, fn)
            return {"profile": summary}
        return {"profile": "no target provided"}
