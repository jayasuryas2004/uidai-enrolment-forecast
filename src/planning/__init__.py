# src/planning/__init__.py
"""
UIDAI Planning Assistant Module
===============================

Provides a safe, read-only planning assistant for resource planning
of Aadhaar enrolment centres.

Components:
    - planning_assistant_spec: System prompts and safety rules
    - planning_router: Intent detection and query routing
    - planning_tools: Data tools for capacity analysis
    - planning_llm: Ollama LLM wrapper (formatter only)
    - planning_tab: Streamlit UI component
"""

from .planning_assistant_spec import (
    PLANNING_ASSISTANT_NAME,
    PLANNING_ASSISTANT_DESCRIPTION,
    PLANNING_SYSTEM_PROMPT,
    PLANNING_ASSISTANT_CONTRACT,
)
from .planning_llm import (
    call_planning_llm,
    check_ollama_available,
    MODEL_NAME,
)
from .planning_router import (
    detect_intent,
    extract_month,
    extract_n_camps,
    handle_planning_query,
    handle_planning_query_raw,
    FALLBACK_LLM_ERROR,
)
from .planning_tab import planning_tab

__all__ = [
    # Spec
    "PLANNING_ASSISTANT_NAME",
    "PLANNING_ASSISTANT_DESCRIPTION",
    "PLANNING_SYSTEM_PROMPT",
    "PLANNING_ASSISTANT_CONTRACT",
    # LLM
    "call_planning_llm",
    "check_ollama_available",
    "MODEL_NAME",
    # Router
    "detect_intent",
    "extract_month",
    "extract_n_camps",
    "handle_planning_query",
    "handle_planning_query_raw",
    "FALLBACK_LLM_ERROR",
    # UI
    "planning_tab",
]
