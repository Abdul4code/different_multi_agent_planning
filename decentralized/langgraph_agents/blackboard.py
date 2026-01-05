"""Blackboard: Shared communication medium for decentralized agents.

In a true decentralized system, there is no central controller.
Agents coordinate by:
1. Posting messages to a shared blackboard
2. Observing messages from other agents
3. Making independent decisions based on observations

The blackboard does NOT control execution - it's just a communication medium.
"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import threading
import json


@dataclass
class Message:
    """A message posted to the blackboard by an agent."""
    sender: str
    msg_type: str  # e.g., "task_available", "task_claimed", "result", "request"
    content: Dict[str, Any]
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    id: str = field(default_factory=lambda: f"{datetime.now().timestamp()}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sender": self.sender,
            "msg_type": self.msg_type,
            "content": self.content,
            "timestamp": self.timestamp,
            "id": self.id
        }


class Blackboard:
    """Shared blackboard for agent communication.
    
    Key properties:
    - NO control logic - just a message store
    - Agents post and read independently
    - No agent "owns" the blackboard
    - Supports concurrent access
    
    This enables truly decentralized coordination where:
    - Task ordering EMERGES from agent decisions
    - No global plan exists
    - Agents can generate new tasks
    """
    
    def __init__(self):
        self._messages: List[Message] = []
        self._lock = threading.Lock()
        self._subscribers: Dict[str, List[callable]] = {}
        
        # Shared observable state (not control state)
        self._shared_data: Dict[str, Any] = {
            "user_goal": "",
            "test_results": None,
            "code_analysis": None,
            "modified_files": [],
            "iteration": 0,
        }
    
    def post(self, message: Message) -> None:
        """Post a message to the blackboard."""
        with self._lock:
            self._messages.append(message)
            # Notify subscribers
            for callback in self._subscribers.get(message.msg_type, []):
                try:
                    callback(message)
                except Exception:
                    pass
    
    def get_messages(self, msg_type: Optional[str] = None, 
                     since_timestamp: Optional[float] = None,
                     sender: Optional[str] = None) -> List[Message]:
        """Get messages from the blackboard with optional filters."""
        with self._lock:
            result = self._messages.copy()
        
        if msg_type:
            result = [m for m in result if m.msg_type == msg_type]
        if since_timestamp:
            result = [m for m in result if m.timestamp > since_timestamp]
        if sender:
            result = [m for m in result if m.sender == sender]
        
        return result
    
    def get_latest(self, msg_type: str) -> Optional[Message]:
        """Get the most recent message of a given type."""
        messages = self.get_messages(msg_type=msg_type)
        return messages[-1] if messages else None
    
    def subscribe(self, msg_type: str, callback: callable) -> None:
        """Subscribe to messages of a specific type."""
        with self._lock:
            if msg_type not in self._subscribers:
                self._subscribers[msg_type] = []
            self._subscribers[msg_type].append(callback)
    
    def update_shared(self, key: str, value: Any) -> None:
        """Update shared observable data."""
        with self._lock:
            self._shared_data[key] = value
    
    def get_shared(self, key: str, default: Any = None) -> Any:
        """Get shared observable data."""
        with self._lock:
            return self._shared_data.get(key, default)
    
    def get_all_shared(self) -> Dict[str, Any]:
        """Get all shared data."""
        with self._lock:
            return self._shared_data.copy()
    
    def get_summary(self) -> str:
        """Get a text summary of blackboard state for LLM consumption."""
        with self._lock:
            lines = []
            lines.append(f"User Goal: {self._shared_data.get('user_goal', 'unknown')}")
            lines.append(f"Iteration: {self._shared_data.get('iteration', 0)}")
            
            # Test results
            tr = self._shared_data.get('test_results')
            if tr:
                lines.append(f"Tests: {tr.get('passed', 0)}/{tr.get('total_tests', 0)} passed, {tr.get('failed', 0)} failed")
            else:
                lines.append("Tests: not run yet")
            
            # Code analysis
            ca = self._shared_data.get('code_analysis')
            if ca:
                lines.append(f"Code Analysis: {ca.get('python_files', 0)} Python files found")
            else:
                lines.append("Code Analysis: not done")
            
            # Modified files
            mf = self._shared_data.get('modified_files', [])
            if mf:
                lines.append(f"Modified Files: {mf}")
            
            # Recent messages (last 10)
            recent = self._messages[-10:] if self._messages else []
            if recent:
                lines.append("\nRecent Messages:")
                for msg in recent:
                    lines.append(f"  [{msg.sender}] {msg.msg_type}: {str(msg.content)[:80]}")
            
            return "\n".join(lines)
    
    def clear(self) -> None:
        """Clear all messages (for testing)."""
        with self._lock:
            self._messages.clear()
