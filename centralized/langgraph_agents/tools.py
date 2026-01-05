"""Real side-effect tools used by agents.

All functions perform real filesystem or subprocess actions.
"""
import os
import subprocess
import json
from typing import List, Dict, Optional


def read_file(path: str) -> str:
    path = os.path.abspath(path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_file(path: str, content: str) -> None:
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def apply_patch(path: str, diff_or_new_content: str) -> None:
    """
    A pragmatic apply_patch implementation: if the provided second argument looks
    like a full file content, overwrite the file. If it's a unified diff (starts with
    '*** ' or '---'), try a simple fallback by writing the content after the diff header.

    Note: This tool provides real filesystem changes.
    """
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Heuristic: if diff looks like unified diff, extract new file content after a blank line
    if diff_or_new_content.strip().startswith("---") or diff_or_new_content.strip().startswith("***"):
        # try to find the first blank line following headers
        parts = diff_or_new_content.splitlines()
        # simplistic: write everything after a line that starts with '+++ ' or after the first blank line
        start = 0
        for i, line in enumerate(parts):
            if line.startswith('+++ '):
                start = i + 1
                break
        new_content = "\n".join(parts[start:])
    else:
        new_content = diff_or_new_content

    with open(path, "w", encoding="utf-8") as f:
        f.write(new_content)


def list_files(directory: str = ".") -> List[str]:
    directory = os.path.abspath(directory)
    files = []
    for root, dirs, filenames in os.walk(directory):
        for fn in filenames:
            files.append(os.path.relpath(os.path.join(root, fn), directory))
    return files


def run_tests(command: str = "pytest -q") -> Dict:
    """Run tests via subprocess and return structured results.

    Returns a dict:
      total_tests: int
      passed: int
      failed: int
      failed_test_names: List[str]
      raw: str
    """
    # Detect if we're in a virtualenv or if pytest should be run via python -m pytest
    # Try to find the project's virtualenv python binary
    # Look for env/bin/python relative to various locations
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # centralized/
    project_root = os.path.dirname(script_dir)  # different_multi_agent_planning/
    cwd = os.getcwd()  # current working directory (may be the target repo)
    
    venv_python = None
    # Search in project root, script_dir, and cwd
    search_dirs = [project_root, script_dir, cwd]
    for base_dir in search_dirs:
        for candidate in ['env/bin/python', '.venv/bin/python', 'venv/bin/python']:
            path = os.path.join(base_dir, candidate)
            if os.path.isfile(path):
                venv_python = path
                break
        if venv_python:
            break
    
    # If command starts with 'pytest', replace it with python -m pytest
    if command.startswith('pytest'):
        args = command.split()
        if venv_python:
            # Use virtualenv python with -m pytest
            cmd_parts = [venv_python, '-m'] + args
        else:
            # Fallback to python3 -m pytest
            cmd_parts = ['python3', '-m'] + args
        proc = subprocess.run(cmd_parts, capture_output=True, text=True)
    else:
        # Run command as-is for other test runners
        proc = subprocess.run(command.split(), capture_output=True, text=True)
    out = proc.stdout + "\n" + proc.stderr

    total = 0
    passed = 0
    failed = 0
    errors = 0
    failed_tests = []

    # Quick heuristics to parse pytest output
    # Look for lines like '== 3 passed, 1 failed in 0.12s ==' or '10 passed in 0.01s'
    import re
    
    # Find the LAST summary line with == markers that contains result counts
    # This avoids matching '== FAILURES ==' or '== short test summary info ==' etc.
    summary = None
    for line in reversed(out.splitlines()):
        # Look for lines like "=== 4 failed, 6 passed in 0.12s ===" or "== 10 passed in 0.01s =="
        if re.search(r"=+.*(\d+\s+(passed|failed|error)).*=+", line):
            # Extract the content between == markers
            m = re.search(r"=+\s*(.*?)\s*=+", line)
            if m:
                summary = m.group(1)
                break
    
    if summary:
        # split by comma
        parts = [p.strip() for p in summary.split(",")]
        for p in parts:
            if "passed" in p:
                try:
                    passed = int(re.search(r"(\d+)", p).group(1))
                except Exception:
                    pass
            if "failed" in p:
                try:
                    failed = int(re.search(r"(\d+)", p).group(1))
                except Exception:
                    pass
            if "error" in p:
                try:
                    errors = int(re.search(r"(\d+)", p).group(1))
                except Exception:
                    pass
        total = passed + failed + errors
    else:
        # Try simpler format: "10 passed in 0.01s"
        m_passed = re.search(r"(\d+)\s+passed", out)
        m_failed = re.search(r"(\d+)\s+failed", out)
        m_error = re.search(r"(\d+)\s+error", out)
        
        if m_passed:
            passed = int(m_passed.group(1))
        if m_failed:
            failed = int(m_failed.group(1))
        if m_error:
            errors = int(m_error.group(1))
        
        total = passed + failed + errors

    # Fallback: if pytest exit code is 0 and we didn't parse, assume all passed
    if proc.returncode == 0 and total == 0:
        # try to find number of tests run
        m2 = re.search(r"collected\s+(\d+)\s+items", out)
        if m2:
            total = int(m2.group(1))
            passed = total

    # Attempt to collect failing test names and errors
    # Find 'FAILED test_file.py::test_name' and 'ERROR test_file.py::test_name' patterns
    for line in out.splitlines():
        if line.strip().startswith("FAILED ") or line.strip().startswith("ERROR "):
            parts = line.strip().split()
            if len(parts) >= 2:
                failed_tests.append(parts[1])

    # If we have failed test names but failed count is 0, use the count from failed_tests
    # This handles cases where pytest shows errors but doesn't include them in the summary line
    if len(failed_tests) > 0 and (failed + errors) == 0:
        failed = len(failed_tests)
        total = passed + failed

    return {
        "total_tests": total,
        "passed": passed,
        "failed": failed + errors,  # Combine failures and errors as both indicate test issues
        "failed_test_names": failed_tests,
        "raw": out,
        "returncode": proc.returncode,
    }


def profile_function(function_path: str, function_name: str) -> str:
    """Profile a function by running a small Python subprocess that imports the module and runs cProfile.

    Returns a short profiling summary.
    """
    import tempfile
    import textwrap

    script = textwrap.dedent(f"""
    import cProfile, pstats, io
    import importlib.util, importlib

    spec = importlib.util.spec_from_file_location('mod', r'{function_path}')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    func = getattr(mod, '{function_name}')
    pr = cProfile.Profile()
    pr.enable()
    try:
        # Try calling without args; user should ensure function accepts none for profiling mode
        func()
    except Exception as e:
        print('ERROR', e)
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
    ps.print_stats(20)
    print(s.getvalue())
    """)

    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".py") as tmp:
        tmp.write(script)
        tmp_path = tmp.name

    proc = subprocess.run(["python3", tmp_path], capture_output=True, text=True)
    return proc.stdout + "\n" + proc.stderr
