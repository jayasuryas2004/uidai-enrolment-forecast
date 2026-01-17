# src/planning/planning_llm.py
"""
GitHub Copilot instructions for this file:

- This module wraps calls to a local Ollama model for the UIDAI Planning Assistant.
- The model is a formatter only: it receives a strict system prompt and a TOOL SUMMARY
  string built from internal planning tools.
- Do NOT add any logic that fetches raw DB data or PII here.
- Do NOT invent numbers; always rely on the tool_summary text produced by planning_tools.py.
- On any error, return None so the router can fall back to a safe message.

Planning LLM Wrapper
====================

Provides a safe wrapper for calling Ollama to format tool outputs.

The LLM is used ONLY to format responses from internal tools.
All business logic and numbers come from planning_tools.py.

╔════════════════════════════════════════════════════════════════════════════╗
║  IMPORTANT: The LLM must NOT invent numbers, districts, or data.          ║
║  It only formats the TOOL SUMMARY into natural language.                  ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

from typing import Optional

import requests

from .planning_assistant_spec import PLANNING_SYSTEM_PROMPT

# =============================================================================
# CONFIGURATION
# =============================================================================

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3.2:3b"


# =============================================================================
# LLM CALLER
# =============================================================================

def call_planning_llm(user_message: str, tool_summary: str) -> Optional[str]:
    """
    Call Ollama with strict system prompt + tool summary.
    Return plain text or None on error.
    
    Args:
        user_message: The user's original question.
        tool_summary: Pre-computed tool output from planning_tools.py.
    
    Returns:
        Formatted response string, or None on any error.
    
    Safety:
        - Always returns None on error (never crashes).
        - The LLM is constrained by PLANNING_SYSTEM_PROMPT.
        - All numbers/data come from tool_summary (not invented).
    """
    try:
        # Build user prompt with formatting instruction
        formatted_prompt = (
            f"User question: {user_message}\n\n"
            f"TOOL DATA (use these exact numbers):\n{tool_summary.strip()}\n\n"
            f"Format your response as bullet points with emojis. "
            f"Use the exact numbers from TOOL DATA above."
        )
        
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": PLANNING_SYSTEM_PROMPT},
                {"role": "user", "content": formatted_prompt},
            ],
            "stream": False,
        }
        resp = requests.post(OLLAMA_URL, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        content = data.get("message", {}).get("content", "")
        return content.strip() or None
    except Exception:
        return None


def check_ollama_available() -> bool:
    """
    Check if Ollama is running and accessible.
    
    Returns:
        True if Ollama is available, False otherwise.
    """
    try:
        response = requests.get(
            "http://localhost:11434/api/tags",
            timeout=5,
        )
        return response.status_code == 200
    except Exception:
        return False
