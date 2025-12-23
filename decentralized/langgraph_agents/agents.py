"""Decentralized agent implementations.

Each agent is autonomous, observes shared state, decides whether to act,
and publishes findings/actions to shared state. No agent acts as a central planner.

Agents coordinate through:
- Shared state messages
- Observable tool outputs
- Test results

Control flow emerges from agent decisions, not from a planner.
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
            }, timeout=180)  # Increased from 60 to 180 seconds
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
    """Base class for autonomous agents.
    
    Each agent:
    - Has its own system prompt
    - Observes shared state
    - Decides whether to act
    - Emits structured messages to shared state
    """
    def __init__(self, system_prompt: str, name: str = "agent", base_url: str = "http://localhost:11434/v1"):
        self.name = name
        self.system_prompt = system_prompt
        # Configure ChatOpenAI to use local OpenAI-compatible endpoint
        self.client = ChatOpenAI(model=LLM_SETTINGS["model"], temperature=LLM_SETTINGS["temperature"], openai_api_base=base_url)

    def call_model(self, user_prompt: str) -> str:
        """Call the LLM with retry logic for robustness."""
        messages = [SystemMessage(content=self.system_prompt), HumanMessage(content=user_prompt)]
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = self.client(messages)
                # LangChain ChatOpenAI returns a ChatGeneration; pick content
                try:
                    return resp.content
                except Exception:
                    # fallback
                    return str(resp)
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"[{self.name}] LLM call failed (attempt {attempt + 1}/{max_retries}): {e}")
                    import time
                    time.sleep(2)  # Wait 2 seconds before retry
                    continue
                else:
                    print(f"[{self.name}] LLM call failed after {max_retries} attempts: {e}")
                    return f"ERROR: LLM call failed - {e}"

    def should_act(self, state: Dict[str, Any]) -> bool:
        """Decide whether this agent should act based on current state.
        
        Override in subclasses to implement autonomous decision-making.
        """
        return True

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agent action and return state updates.
        
        Override in subclasses to implement agent behavior.
        """
        raise NotImplementedError


class CodeAnalyzerAgent(BaseAgent):
    """Analyzes codebase and publishes findings.
    
    Acts when:
    - No analysis has been performed yet
    - OR code has been modified and needs re-analysis
    
    Publishes:
    - List of relevant files
    - Code structure summary
    - Identified functions/classes
    """
    def __init__(self, base_url: str = "http://localhost:11434/v1"):
        super().__init__(
            system_prompt="""You are CodeAnalyzerAgent: an autonomous agent that analyzes codebases.

Your role:
1. Examine source and test files in the codebase
2. Identify key files, functions, and patterns
3. Publish structured findings to shared state

You are a peer agent - you don't plan or control others. You provide information.""",
            name="CodeAnalyzer",
            base_url=base_url
        )

    def should_act(self, state: Dict[str, Any]) -> bool:
        # Act if no analysis exists or if code was recently modified
        has_analysis = bool(state.get("code_analysis"))
        was_modified = bool(state.get("recently_modified_files"))
        needs_initial_analysis = state.get("iteration", 0) <= 1
        
        return needs_initial_analysis or (not has_analysis) or was_modified

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if not self.should_act(state):
            return {"agent_messages": state.get("agent_messages", []) + [
                {"agent": self.name, "action": "skip", "reason": "No analysis needed"}
            ]}

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
        priority = [f for f in py_files if '/tests/' in f or f.startswith('tests/') or f.endswith('_test.py') or f.endswith('test.py') or f.startswith('test_')]
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

        analysis = {
            "total_files": len(files),
            "python_files": len(py_files),
            "key_files": key_files,
            "snippets": snippets,
            "timestamp": state.get("iteration", 0)
        }

        return {
            "code_analysis": analysis,
            "recently_modified_files": None,  # Clear modification flag
            "agent_messages": state.get("agent_messages", []) + [
                {
                    "agent": self.name,
                    "action": "analyzed",
                    "summary": f"Found {len(py_files)} Python files, identified {len(key_files)} key files"
                }
            ]
        }


class TestRunnerAgent(BaseAgent):
    """Executes real test suite and publishes results.
    
    Acts when:
    - No test results exist yet
    - OR code has been modified since last test run
    
    Publishes:
    - Structured test results (total, passed, failed)
    - Failing test names
    - Raw test output
    """
    def __init__(self, base_url: str = "http://localhost:11434/v1"):
        super().__init__(
            system_prompt="""You are TestRunnerAgent: an autonomous agent that runs tests.

Your role:
1. Execute the real test suite via subprocess
2. Parse and structure test results
3. Publish results to shared state

You are a peer agent - you provide factual test results, not opinions.""",
            name="TestRunner",
            base_url=base_url
        )

    def should_act(self, state: Dict[str, Any]) -> bool:
        # Act if no test results exist or if code was modified
        has_results = state.get("test_results") is not None
        code_modified = state.get("code_modified", False)
        
        # Always run on first iteration
        first_run = state.get("iteration", 0) <= 1
        
        return first_run or (not has_results) or code_modified

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if not self.should_act(state):
            return {"agent_messages": state.get("agent_messages", []) + [
                {"agent": self.name, "action": "skip", "reason": "Tests already run"}
            ]}

        # Run tests using real subprocess
        cmd = state.get("test_command", "pytest -q")
        res = tools.run_tests(cmd)
        
        # Structure results
        test_results = {
            "total_tests": int(res.get("total_tests", 0)),
            "passed": int(res.get("passed", 0)),
            "failed": int(res.get("failed", 0)),
            "failed_test_names": res.get("failed_test_names", []),
            "raw": res.get("raw", ""),
            "returncode": res.get("returncode", 1),
            "timestamp": state.get("iteration", 0)
        }

        return {
            "test_results": test_results,
            "code_modified": False,  # Clear modification flag
            "agent_messages": state.get("agent_messages", []) + [
                {
                    "agent": self.name,
                    "action": "ran_tests",
                    "summary": f"Tests: {test_results['passed']}/{test_results['total_tests']} passed, {test_results['failed']} failed"
                }
            ]
        }


class DebuggerAgent(BaseAgent):
    """Diagnoses test failures and publishes suggested fixes.
    
    Acts when:
    - Tests are failing
    - AND no unprocessed fixes exist
    
    Publishes:
    - Root cause analysis
    - Specific fix suggestions with file/line/reason
    """
    def __init__(self, base_url: str = "http://localhost:11434/v1"):
        super().__init__(
            system_prompt="""You are DebuggerAgent: an autonomous agent that diagnoses bugs.

Your role:
1. Observe failing test output
2. Read relevant source files
3. Identify root causes
4. Publish specific fix suggestions

You are a peer agent - you suggest fixes, but don't implement them yourself.

Return JSON with:
{
  "analysis": "brief root cause description",
  "fix_suggestions": [
    {
      "file": "path/to/file.py",
      "reason": "Clear description of bug (e.g., 'Remove + 1.0 from subtotal calculation')",
      "line": 50
    }
  ]
}""",
            name="Debugger",
            base_url=base_url
        )

    def should_act(self, state: Dict[str, Any]) -> bool:
        # Act if tests are failing and no pending fixes
        test_results = state.get("test_results", {})
        has_failures = test_results.get("failed", 0) > 0
        has_pending_fixes = bool(state.get("pending_fixes"))
        
        return has_failures and not has_pending_fixes

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if not self.should_act(state):
            return {"agent_messages": state.get("agent_messages", []) + [
                {"agent": self.name, "action": "skip", "reason": "No failures or fixes pending"}
            ]}

        test_results = state.get("test_results", {})
        raw = test_results.get("raw", "")
        failed = test_results.get("failed", 0)
        
        if failed == 0:
            return {"agent_messages": state.get("agent_messages", []) + [
                {"agent": self.name, "action": "skip", "reason": "No failures"}
            ]}
        
        # Extract file paths from test output
        import re
        file_mentions = re.findall(r'File "?([^":\s]+\.py)"?, line (\d+)', raw)
        # Also look for simpler format: cart.py:14
        file_mentions += re.findall(r'(\w+\.py):(\d+):', raw)
        
        # Get unique files - separate source files and test files
        source_files = list(set(f[0] for f in file_mentions if not f[0].startswith('test_')))
        test_files = list(set(f[0] for f in file_mentions if f[0].startswith('test_')))
        
        # Read the source files AND test files (full content)
        file_contents = {}
        for fpath in source_files[:3]:  # Limit to 3 source files
            try:
                file_contents[fpath] = tools.read_file(fpath)
            except Exception:
                pass
        
        # Also read test files for context
        for fpath in test_files[:2]:  # Include up to 2 test files
            try:
                file_contents[fpath] = tools.read_file(fpath)
            except Exception:
                pass
        
        # Build prompt for LLM (more concise to reduce generation time)
        prompt = f"""Analyze test failures and identify bugs.

FAILED: {failed} tests
TEST OUTPUT (first 3000 chars):
{raw[:3000]}

SOURCE FILES:
"""
        for fpath, content in file_contents.items():
            # Limit content to 2000 chars per file to reduce prompt size
            prompt += f"\n=== {fpath} ===\n{content[:2000]}\n"
        
        prompt += """

Return JSON:
{
  "analysis": "brief root cause",
  "fix_suggestions": [
    {"file": "path.py", "reason": "what to fix", "line": 50}
  ]
}

Be specific about the bug (e.g., "Remove + 1.0 from line X", "Change < to <=").
"""
        
        # Call LLM
        response = self.call_model(prompt)
        
        # Parse JSON from response
        try:
            # Try to extract JSON from response
            if '{' in response:
                start = response.index('{')
                end = response.rindex('}') + 1
                result = json.loads(response[start:end])
            else:
                result = {"analysis": "Could not parse LLM response", "fix_suggestions": []}
        except Exception as e:
            result = {"analysis": f"Error parsing LLM response: {e}", "fix_suggestions": []}
        
        return {
            "pending_fixes": result.get("fix_suggestions", []),
            "debug_analysis": result.get("analysis", ""),
            "agent_messages": state.get("agent_messages", []) + [
                {
                    "agent": self.name,
                    "action": "diagnosed",
                    "summary": f"Found {len(result.get('fix_suggestions', []))} bugs: {result.get('analysis', '')[:100]}"
                }
            ]
        }


class CodeWriterAgent(BaseAgent):
    """Applies code changes to the real filesystem.
    
    Acts when:
    - Pending fixes exist from DebuggerAgent
    - OR optimization suggestions exist from PerformanceAnalyzerAgent
    
    Publishes:
    - List of modified files
    - Success/failure status
    """
    def __init__(self, base_url: str = "http://localhost:11434/v1"):
        super().__init__(
            system_prompt="""You are CodeWriterAgent: an autonomous agent that fixes code.

Your role:
1. Read pending fix suggestions from shared state
2. Read the source files
3. Apply fixes to the real filesystem
4. Publish modified file list

You are a peer agent - you implement fixes suggested by others.

Generate the COMPLETE corrected file content. Output ONLY the Python code, no explanations.""",
            name="CodeWriter",
            base_url=base_url
        )

    def should_act(self, state: Dict[str, Any]) -> bool:
        # Act if there are pending fixes or optimization suggestions
        pending_fixes = state.get("pending_fixes")
        optimization_suggestions = state.get("optimization_suggestions")
        
        has_pending_fixes = pending_fixes is not None and len(pending_fixes) > 0
        has_optimization = optimization_suggestions is not None and len(optimization_suggestions) > 0
        
        return has_pending_fixes or has_optimization

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if not self.should_act(state):
            return {"agent_messages": state.get("agent_messages", []) + [
                {"agent": self.name, "action": "skip", "reason": "No fixes to apply"}
            ]}

        # Get fix suggestions (safely handle None values)
        pending_fixes = state.get("pending_fixes") or []
        optimization_suggestions = state.get("optimization_suggestions") or []
        fix_suggestions = pending_fixes + optimization_suggestions
        
        if not fix_suggestions:
            return {"agent_messages": state.get("agent_messages", []) + [
                {"agent": self.name, "action": "skip", "reason": "No suggestions"}
            ]}
        
        # Group suggestions by file
        files_to_fix = {}
        for suggestion in fix_suggestions:
            filepath = suggestion.get("file")
            if filepath:
                if filepath not in files_to_fix:
                    files_to_fix[filepath] = []
                files_to_fix[filepath].append(suggestion)
        
        # Process each file
        patched = []
        notes = []
        
        for filepath, suggestions in files_to_fix.items():
            try:
                # Read current file content
                current_content = tools.read_file(filepath)
                
                # Build prompt for LLM (keep full content for accurate fixes)
                prompt = f"""Fix bugs in this file.

FILE: {filepath}
CURRENT CONTENT:
```python
{current_content}
```

FIXES NEEDED:
"""
                for idx, suggestion in enumerate(suggestions, 1):
                    prompt += f"\n{idx}. {suggestion.get('reason', 'Fix bug')}"
                    if 'line' in suggestion:
                        prompt += f" (line ~{suggestion.get('line')})"
                
                prompt += """

Output the COMPLETE corrected file. Python code only, no markdown fences.
"""
                
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
                    notes.append(f"Invalid content for {filepath}")
                    continue
                
                # Write the fixed file
                tools.apply_patch(filepath, fixed_content)
                patched.append(filepath)
                notes.append(f"✓ Fixed {filepath}")
            
            except Exception as e:
                notes.append(f"Error fixing {filepath}: {e}")
        
        return {
            "modified_files": state.get("modified_files", []) + patched,
            "code_modified": bool(patched),  # Signal that tests should re-run
            "pending_fixes": None,  # Clear pending fixes
            "optimization_suggestions": None,  # Clear optimization suggestions
            "agent_messages": state.get("agent_messages", []) + [
                {
                    "agent": self.name,
                    "action": "modified",
                    "summary": f"Modified {len(patched)} files: {', '.join(patched)}"
                }
            ]
        }


class PerformanceAnalyzerAgent(BaseAgent):
    """Analyzes performance and suggests optimizations.
    
    Acts when:
    - User request mentions performance/optimization
    - AND tests are passing
    - AND no pending optimization suggestions exist
    
    Publishes:
    - Performance analysis
    - Optimization suggestions
    """
    def __init__(self, base_url: str = "http://localhost:11434/v1"):
        super().__init__(
            system_prompt="""You are PerformanceAnalyzerAgent: an autonomous agent that optimizes code.

Your role:
1. Profile target functions when requested
2. Identify performance bottlenecks
3. Suggest optimizations that preserve correctness
4. Publish optimization suggestions

You are a peer agent - you analyze and suggest, not implement.""",
            name="PerformanceAnalyzer",
            base_url=base_url
        )

    def should_act(self, state: Dict[str, Any]) -> bool:
        # Act if performance mode requested and tests passing
        user_request = state.get("user_request", "").lower()
        is_perf_mode = "perf" in user_request or "optimi" in user_request
        
        test_results = state.get("test_results", {})
        tests_passing = test_results.get("failed", 0) == 0 and test_results.get("total_tests", 0) > 0
        
        has_pending = bool(state.get("optimization_suggestions"))
        
        return is_perf_mode and tests_passing and not has_pending

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if not self.should_act(state):
            return {"agent_messages": state.get("agent_messages", []) + [
                {"agent": self.name, "action": "skip", "reason": "Not in performance mode or tests failing"}
            ]}

        # Get target function from state
        target_path = state.get("profile_target_path")
        target_name = state.get("profile_target_name")
        
        if target_path and target_name:
            summary = tools.profile_function(target_path, target_name)
            
            # Create optimization suggestion
            suggestions = [{
                "file": target_path,
                "reason": f"Optimize {target_name} based on profiling: {summary[:200]}",
                "line": 0
            }]
            
            return {
                "optimization_suggestions": suggestions,
                "performance_profile": summary,
                "agent_messages": state.get("agent_messages", []) + [
                    {
                        "agent": self.name,
                        "action": "profiled",
                        "summary": f"Profiled {target_name}, found optimization opportunities"
                    }
                ]
            }
        
        return {"agent_messages": state.get("agent_messages", []) + [
            {"agent": self.name, "action": "skip", "reason": "No profile target specified"}
        ]}
