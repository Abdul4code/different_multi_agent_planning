"""Local OpenAI-compatible LLM client.

This client talks to a local OpenAI-compatible endpoint (no API key) that
implements the /v1/chat/completions endpoint. It returns the assistant
message content as a string.
"""
from typing import Any, Dict, List, Optional
import requests
import json


class LocalLLM:
    def __init__(self, api_base: str = "http://localhost:11434/v1", model: str = "qwen2.5:7b", timeout: int = 60):
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.timeout = timeout

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.2, max_tokens: int = 1024) -> str:
        url = f"{self.api_base}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=self.timeout)
        resp.raise_for_status()
        rj = resp.json()
        # Follow OpenAI-compatible shape: choices[0].message.content
        try:
            return rj["choices"][0]["message"]["content"]
        except Exception:
            # Fallback: return raw json
            return json.dumps(rj)
