# src/planning/planning_llm.py
"""
GitHub Copilot instructions for this file:
- This module wraps calls to Groq API for the UIDAI Planning Assistant.
- The model is a formatter only: it receives a strict system prompt and a TOOL SUMMARY
  string built from internal planning tools.
- Do NOT add any logic that fetches raw DB data or PII here.
- Do NOT invent numbers; always rely on the tool_summary text produced by planning_tools.py.
- On any error, return None so the router can fall back to a safe message.
 
Planning LLM Wrapper
====================
Provides a safe wrapper for calling Groq to format tool outputs.
The LLM is used ONLY to format responses from internal tools.
All business logic and numbers come from planning_tools.py.
 
╔════════════════════════════════════════════════════════════════════════════╗
║  IMPORTANT: The LLM must NOT invent numbers, districts, or data.          ║
║  It only formats the TOOL SUMMARY into natural language.                  ║
╚════════════════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations
 
import os
import logging
from typing import Optional
 
from .planning_assistant_spec import PLANNING_SYSTEM_PROMPT
 
log = logging.getLogger(__name__)
 
# =============================================================================
# CONFIGURATION
# =============================================================================
 
MODEL_NAME = "llama-3.1-8b-instant"   # Free Groq model, replaces llama3.2:3b
 
# =============================================================================
# GROQ CLIENT (lazy-loaded)
# =============================================================================
 
_groq_client = None
 
def _get_client():
    """Return cached Groq client, or None if GROQ_API_KEY not set."""
    global _groq_client
    if _groq_client is not None:
        return _groq_client
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        from groq import Groq
        _groq_client = Groq(api_key=api_key)
        return _groq_client
    except Exception as e:
        log.error("Failed to create Groq client: %s", e)
        return None
 
# =============================================================================
# LLM CALLER  —  same signature as before, just Groq instead of Ollama
# =============================================================================
 
def call_planning_llm(user_message: str, tool_summary: str) -> Optional[str]:
    """
    Call Groq with strict system prompt + tool summary.
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
    client = _get_client()
    if client is None:
        log.warning("GROQ_API_KEY not set — returning None for fallback.")
        return None
 
    try:
        # Identical prompt structure to original Ollama version
        formatted_prompt = (
            f"User question: {user_message}\n\n"
            f"TOOL DATA (use these exact numbers):\n{tool_summary.strip()}\n\n"
            f"Format your response as bullet points with emojis. "
            f"Use the exact numbers from TOOL DATA above."
        )
 
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": PLANNING_SYSTEM_PROMPT},
                {"role": "user",   "content": formatted_prompt},
            ],
            temperature=0.2,
            max_tokens=1024,
        )
 
        content = response.choices[0].message.content
        return content.strip() or None
 
    except Exception as e:
        log.error("Groq API call failed: %s", e)
        return None   # Router falls back to safe message, same as before
 
 
# =============================================================================
# AVAILABILITY CHECK  —  same signature as before
# =============================================================================
 
def check_ollama_available() -> bool:
    """
    Originally checked if Ollama was running locally.
    Now checks if GROQ_API_KEY is set and Groq is reachable.
 
    Returns:
        True if Groq is ready, False otherwise.
    Called by planning_tab.py — signature unchanged.
    """
    client = _get_client()
    if client is None:
        return False
    try:
        client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=3,
        )
        return True
    except Exception as e:
        log.warning("Groq connectivity check failed: %s", e)
        return False
 
