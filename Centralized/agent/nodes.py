"""Node implementations for the centralized planner architecture.

Nodes operate on a shared dictionary state. The planner instructs which
worker to call next by setting state['next_node'] or by returning structured
plans in state['plan'].
"""
from typing import Any, Dict, List, Optional
import json
import os
import subprocess
import sys
from .graph import Node
from .llm_client import LocalLLM


class PlannerNode(Node):
    def __init__(self, llm: LocalLLM):
        super().__init__("planner")
        self.llm = llm

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # Interpret user prompt and produce a JSON plan with next node
        prompt = state.get("user_prompt", "Make changes until tests pass.")
        system = (
            "You are a centralized planner that produces a JSON plan. "
            "Respond with a JSON object containing 'action' and 'target'."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        content = self.llm.chat(messages)
        # Try to parse JSON from LLM; if not possible, fallback to simple plan
        plan = None
        try:
            plan = json.loads(content)
        except Exception:
            plan = {"action": "analyze", "target": "code"}

        state["plan"] = plan
        # Map action to next node
        action = plan.get("action") if isinstance(plan, dict) else None
        if action == "analyze":
            state["next_node"] = "analyzer"
        elif action == "write":
            state["next_node"] = "writer"
        elif action == "test":
            state["next_node"] = "tester"
        elif action == "debug":
            state["next_node"] = "debugger"
        else:
            # default cycle
            state["next_node"] = "analyzer"

        return state


class CodeAnalyzerNode(Node):
    def __init__(self, repo_root: str):
        super().__init__("analyzer")
        self.repo_root = repo_root

    def _find_py_files(self) -> List[str]:
        py_files = []
        for root, dirs, files in os.walk(self.repo_root):
            # skip virtualenv site-packages
            if "site-packages" in root:
                continue
            for f in files:
                if f.endswith(".py"):
                    py_files.append(os.path.join(root, f))
        return py_files

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # Read tests and source files to summarize quickly
        files = self._find_py_files()
        summary = {"file_count": len(files), "sample_files": files[:20]}
        state.setdefault("analysis", {}).update(summary)
        # If tests failed, include failure snippets
        if state.get("test_results"):
            state.setdefault("analysis", {})["last_test_results"] = state["test_results"]

        # Decide next action: write or test
        # Simple heuristic: if no tests run yet, run tests.
        if not state.get("test_results"):
            state["next_node"] = "tester"
        else:
            # If tests failed, go to debugger
            if not state.get("test_results", {}).get("passed", False):
                state["next_node"] = "debugger"
            else:
                state["terminate"] = True

        return state


class CodeWriterNode(Node):
    def __init__(self, repo_root: str):
        super().__init__("writer")
        self.repo_root = repo_root

    def apply_patch_text(self, filepath: str, new_text: str) -> None:
        abs_path = os.path.join(self.repo_root, filepath) if not os.path.isabs(filepath) else filepath
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(new_text)

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # Expect state['plan'] to contain write instructions with 'files'
        plan = state.get("plan", {})
        files = plan.get("files") if isinstance(plan, dict) else None
        if not files:
            # Nothing to write — go to tester
            state["next_node"] = "tester"
            return state

        modified = []
        for fpath, content in files.items():
            self.apply_patch_text(fpath, content)
            modified.append(fpath)

        state.setdefault("modified_files", []).extend(modified)
        state["next_node"] = "tester"
        return state


class TestRunnerNode(Node):
    def __init__(self, repo_root: str, pytest_args: Optional[List[str]] = None):
        super().__init__("tester")
        self.repo_root = repo_root
        self.pytest_args = pytest_args or ["-q"]

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # Run pytest in subprocess and capture results
        cmd = [sys.executable, "-m", "pytest"] + self.pytest_args
        try:
            proc = subprocess.run(cmd, cwd=self.repo_root, capture_output=True, text=True, check=False)
            out = proc.stdout + "\n" + proc.stderr
            passed = proc.returncode == 0
            state["test_results"] = {"passed": passed, "returncode": proc.returncode, "output": out}
        except Exception as e:
            state["test_results"] = {"passed": False, "error": str(e)}

        # Next step: if passed, terminate; else debugger
        if state["test_results"].get("passed"):
            state["terminate"] = True
        else:
            state["next_node"] = "debugger"

        return state


class DebuggerNode(Node):
    def __init__(self, llm: LocalLLM):
        super().__init__("debugger")
        self.llm = llm

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # Use the LLM to analyze failing test output and propose a small patch
        tr = state.get("test_results", {})
        output = tr.get("output") or tr.get("error") or ""
        system = (
            "You are a helpful debugger. Given failing pytest output and a short "
            "summary of the repository, propose a JSON object with 'files' mapping "
            "relative file paths to new full contents to WRITE. Keep changes minimal."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": output},
        ]
        content = self.llm.chat(messages)
        try:
            suggestion = json.loads(content)
        except Exception:
            suggestion = {"message": content}

        # Attach suggestion to plan and instruct writer
        state["plan"] = {"action": "write", "files": suggestion.get("files") if isinstance(suggestion, dict) else None}
        if state["plan"].get("files"):
            state["next_node"] = "writer"
        else:
            # If the LLM didn't provide a files map, end with failure collected
            state.setdefault("errors", []).append({"debugger_output": content})
            state["terminate"] = True

        return state
