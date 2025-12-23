# Decentralized Multi-Agent System

A truly decentralized LLM multi-agent system using LangGraph where autonomous agents coordinate without a central planner.

## Architecture

### Core Principles

1. **No Central Planner**: No single agent acts as a global controller
2. **Peer Autonomy**: All agents are autonomous peers with bounded responsibilities
3. **Emergent Coordination**: Control flow emerges from agent decisions based on shared state
4. **Real Side Effects**: All code changes and test executions are real (no simulation)

### Agents

Each agent is autonomous and decides when to act by observing shared state:

- **CodeAnalyzerAgent**: Inspects codebase and publishes file structure/findings
- **TestRunnerAgent**: Executes real test suite and publishes results
- **DebuggerAgent**: Diagnoses test failures and publishes fix suggestions
- **CodeWriterAgent**: Applies code changes to real filesystem based on suggestions
- **PerformanceAnalyzerAgent**: Profiles code and suggests optimizations

### Coordination Mechanism

Agents coordinate through:
- **Shared State**: Common state dictionary accessible to all agents
- **Message Passing**: Structured messages published to state
- **Observable Outputs**: Test results, file modifications, etc.

### Termination

The system terminates when:
- All tests pass (state convergence), OR
- Maximum iterations reached

Termination is detected by state observation, not by a planner decision.

## Usage

```bash
python3 run_system.py --user "Fix failing tests" --repo ./path/to/repo --max-iterations 20
```

### Arguments

- `--user`, `-u`: User request describing the task
- `--repo`, `-r`: Path to the target repository
- `--max-iterations`, `-m`: Maximum iterations before stopping (default: 20)

### Environment

- Local LLM endpoint: `http://localhost:11434/v1`
- Model: `qwen2.5:7b`

## Supported Modes

The system collectively supports three capabilities:

### (A) Debugging
Modify the codebase so all existing tests pass.

```bash
python3 run_system.py --repo ./agent_benchmark/repo_1_shopping_cart --user "Fix failing tests"
```

### (B) Feature Addition
Add new functionality so new tests pass.

```bash
python3 run_system.py --repo ./my_project --user "Add new feature to pass tests"
```

### (C) Performance Optimization
Optimize runtime performance while preserving correctness.

```bash
python3 run_system.py --repo ./my_project --user "Optimize performance of function X"
```

Mode selection emerges from agent interaction based on user request keywords.

## Output

### On Disk
- Modified source files
- `CHANGELOG_AGENT.md`: Detailed execution log including:
  - Actions taken by each agent
  - Files and functions modified
  - Rationale for changes
  - Test results

### Programmatic Output
- Number of tests passed
- Number of tests failed
- Names of failing tests (if any)
- Elapsed time
- Modified files list

## Requirements

See `requirements.txt` for dependencies.

## Key Features
 **Decentralized**: No central planner - agents are autonomous peers  
**Real Side Effects**: Actual file modifications and test execution  
**LangGraph**: Each node wraps exactly one agent  
**Local LLM**: Uses locally hosted model via OpenAI-compatible API  
**State-Based Coordination**: Agents communicate via shared state  
**Emergent Control Flow**: Decisions emerge from agent observations  

## Comparison with Centralized System

| Aspect | Centralized | Decentralized |
|--------|-------------|---------------|
| Planner | Single PlannerAgent controls flow | No planner - coordination emerges |
| Decision Making | Planner decides which agent runs | Each agent decides if it should act |
| Control Flow | Hardcoded by planner | Emerges from state conditions |
| Coordination | Via planner commands | Via shared state messages |
| Termination | Planner decides when to stop | State convergence detection |
