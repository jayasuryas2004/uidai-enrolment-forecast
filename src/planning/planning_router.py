# src/planning/planning_router.py
"""
Planning Router
===============

Intent detection, query routing, and safety guardrails for the
UIDAI Planning Assistant.

This module provides:
    - detect_intent(): Classify user queries into allowed/blocked intents
    - extract_month(): Parse YYYY-MM month from query
    - extract_n_camps(): Parse number of extra centres/camps
    - handle_planning_query_raw(): Main routing logic (Phase 0, no LLM)

╔════════════════════════════════════════════════════════════════════════════╗
║  SAFETY: Personal data queries are blocked at the router level.           ║
║  All responses come from tools, never from LLM invention.                 ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import re
from typing import Literal, Optional, Tuple

from .planning_assistant_spec import (
    PLANNING_ASSISTANT_NAME,
    PERSONAL_DATA_RESPONSE,
    UNSUPPORTED_QUERY_RESPONSE,
)
from .planning_tools import (
    summarize_over_capacity,
    summarize_simulation,
)


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

Intent = Literal[
    "over_capacity",
    "simulate",
    "explain_dashboard",
    "unsafe_personal",
    "raw_export",
    "unsupported",
]


# =============================================================================
# PATTERNS FOR INTENT DETECTION
# =============================================================================

# Personal data / Aadhaar guardrail (BLOCKED)
PERSONAL_PATTERN = re.compile(
    r"(aadhaar|aadhar|uidai\s*number|id\s*number|enrol(?:l)?ment\s*id|\b\d{10,12}\b)",
    re.IGNORECASE,
)

# Month pattern: YYYY-MM (e.g., 2025-04)
MONTH_PATTERN = re.compile(r"\b(20[2-9]\d)-(0[1-9]|1[0-2])\b")

# Raw export patterns (BLOCKED)
RAW_EXPORT_PATTERN = re.compile(
    r"(export|download|dump|csv|excel|all\s+districts|full\s+table|show\s+all)",
    re.IGNORECASE,
)

# Over-capacity keywords
OVER_CAPACITY_KEYWORDS = [
    "over capacity",
    "over-capacity",
    "overcapacity",
    "capacity gap",
    "gap",
    "exceeding capacity",
    "highest demand",
    "top states",
    "top districts",
    "which states",
    "which districts",
]

# Simulation keywords
SIMULATION_KEYWORDS = [
    "extra centre",
    "extra center",
    "extra centres",
    "extra centers",
    "add centre",
    "add center",
    "add centres",
    "add centers",
    "camp",
    "camps",
    "if we add",
    "what if",
    "simulate",
    "simulation",
    "impact of adding",
]

# Dashboard explanation keywords
DASHBOARD_KEYWORDS = [
    "explain",
    "what does",
    "how is",
    "meaning of",
    "define",
    "dashboard",
]


# =============================================================================
# INTENT DETECTION
# =============================================================================

def detect_intent(message: str) -> Intent:
    """
    Classify user message into an intent category.
    
    Safety checks are performed FIRST (personal data, raw exports).
    Then allowed intents are matched.
    
    Args:
        message: User's input message
        
    Returns:
        Intent literal string
    """
    text = message.lower()
    
    # 1) SAFETY: Block personal data / Aadhaar queries
    if PERSONAL_PATTERN.search(message):
        return "unsafe_personal"
    
    # 2) SAFETY: Block raw export requests
    if RAW_EXPORT_PATTERN.search(text):
        return "raw_export"
    
    # 3) Over-capacity questions
    if any(kw in text for kw in OVER_CAPACITY_KEYWORDS):
        return "over_capacity"
    
    # 4) Simulation questions
    if any(kw in text for kw in SIMULATION_KEYWORDS):
        return "simulate"
    
    # 5) Dashboard explanation
    if any(kw in text for kw in DASHBOARD_KEYWORDS):
        return "explain_dashboard"
    
    # 6) Fallback: unsupported
    return "unsupported"


# =============================================================================
# EXTRACTION HELPERS
# =============================================================================

def extract_month(message: str) -> Optional[str]:
    """
    Extract YYYY-MM month from message.
    
    Args:
        message: User's input message
        
    Returns:
        Month string (e.g., "2025-04") or None if not found
    """
    match = MONTH_PATTERN.search(message)
    if match:
        return match.group(0)
    return None


def extract_n_camps(message: str) -> int:
    """
    Extract number of extra centres/camps from message.
    
    Uses simple heuristic: first integer in message, default 1.
    
    Args:
        message: User's input message
        
    Returns:
        Number of camps (minimum 1)
    """
    nums = re.findall(r"\b\d+\b", message)
    if not nums:
        return 1
    try:
        return max(1, int(nums[0]))
    except Exception:
        return 1


def extract_region(message: str) -> Optional[str]:
    """
    Extract state/district name from message (basic heuristic).
    
    This is a simple implementation that looks for common state names.
    In production, this would use a proper NER or lookup table.
    
    Args:
        message: User's input message
        
    Returns:
        Region name or None
    """
    # Common Indian states (subset for demo)
    states = [
        "Tamil Nadu", "Karnataka", "Maharashtra", "Uttar Pradesh",
        "Bihar", "Rajasthan", "Gujarat", "Madhya Pradesh",
        "Andhra Pradesh", "Telangana", "Kerala", "West Bengal",
        "Delhi", "Haryana", "Punjab",
    ]
    
    text_lower = message.lower()
    for state in states:
        if state.lower() in text_lower:
            return state
    
    return None


# =============================================================================
# RESPONSE BUILDERS
# =============================================================================

def build_personal_data_response() -> str:
    """Return the standard response for personal data queries."""
    return PERSONAL_DATA_RESPONSE


def build_unsupported_response() -> str:
    """Return the standard response for unsupported queries."""
    return UNSUPPORTED_QUERY_RESPONSE


def build_raw_export_response() -> str:
    """Return the standard response for raw export requests."""
    return (
        "This assistant cannot provide raw data exports or full table dumps. "
        "It can only summarize the top states/districts with capacity gaps.\n\n"
        "Try asking: 'Which states are over-capacity in 2025-04?'"
    )


def build_explain_response() -> str:
    """Return a brief explanation of dashboard metrics."""
    return (
        "The dashboard shows:\n"
        "- **Forecast**: Predicted enrolments based on the XGBoost v3 model.\n"
        "- **Capacity**: Estimated centre capacity (enrolments/month).\n"
        "- **Gap**: Forecast minus capacity. Positive = over-capacity.\n"
        "- **Extra centres needed**: How many additional centres to close the gap.\n\n"
        "Ask me about specific states or months for detailed analysis."
    )


# =============================================================================
# FALLBACK MESSAGES
# =============================================================================

FALLBACK_LLM_ERROR = (
    "Planning Assistant is temporarily unavailable. "
    "Please use the main dashboard filters for now."
)


# =============================================================================
# MAIN ROUTING HANDLER (Phase 0 - No LLM)
# =============================================================================

def handle_planning_query_raw(message: str) -> Tuple[str, Intent]:
    """
    Phase 0 handler: No LLM, only safety checks + dummy tool summaries.
    
    In Phase 1+, this will be replaced with LLM-based response generation
    using Ollama.
    
    Args:
        message: User's input message
        
    Returns:
        Tuple of (reply_text, intent)
    """
    intent = detect_intent(message)
    
    # 1) BLOCKED: Personal data
    if intent == "unsafe_personal":
        return build_personal_data_response(), intent
    
    # 2) BLOCKED: Raw exports
    if intent == "raw_export":
        return build_raw_export_response(), intent
    
    # 3) ALLOWED: Over-capacity analysis
    if intent == "over_capacity":
        month = extract_month(message) or "2025-04"
        tool_summary = summarize_over_capacity(month)
        reply = (
            "Here is a data-driven summary for over-capacity states:\n\n"
            + tool_summary
        )
        return reply, intent
    
    # 4) ALLOWED: Simulation
    if intent == "simulate":
        region = extract_region(message)
        n_camps = extract_n_camps(message)
        month = extract_month(message)
        tool_summary = summarize_simulation(
            user_message=message,
            region=region,
            n_camps=n_camps,
            month=month,
        )
        reply = (
            "Here is a simulation summary based on your request:\n\n"
            + tool_summary
        )
        return reply, intent
    
    # 5) ALLOWED: Dashboard explanation
    if intent == "explain_dashboard":
        return build_explain_response(), intent
    
    # 6) FALLBACK: Unsupported
    return build_unsupported_response(), "unsupported"


# =============================================================================
# MAIN ROUTING HANDLER (Phase 1 - With LLM)
# =============================================================================

def handle_planning_query(message: str) -> Tuple[str, Intent]:
    """
    Phase 1 handler: safety + tools + LLM formatter.
    
    The LLM is a formatter only - all numbers and data come from tools.
    Returns graceful fallback if LLM is unavailable.
    
    Args:
        message: User's input message
        
    Returns:
        Tuple of (reply_text, intent)
        
    Safety:
        - Personal data queries are blocked BEFORE any tool/LLM call.
        - All numeric data comes from planning_tools.py.
        - LLM failures are handled gracefully with fallback.
    """
    from .planning_llm import call_planning_llm
    
    intent = detect_intent(message)
    
    # 1) BLOCKED: Personal data (handled FIRST, no LLM needed)
    if intent == "unsafe_personal":
        return build_personal_data_response(), intent
    
    # 2) BLOCKED: Raw exports (handled FIRST, no LLM needed)
    if intent == "raw_export":
        return build_raw_export_response(), intent
    
    # 3) Unsupported queries (no tool data to format)
    if intent == "unsupported":
        return build_unsupported_response(), intent
    
    # Build TOOL SUMMARY (from tools / dummy data)
    if intent == "over_capacity":
        month = extract_month(message) or "2025-04"
        tool_summary = summarize_over_capacity(month)
    
    elif intent == "simulate":
        region = extract_region(message)
        n_camps = extract_n_camps(message)
        month = extract_month(message)
        tool_summary = summarize_simulation(
            user_message=message,
            region=region,
            n_camps=n_camps,
            month=month,
        )
    
    elif intent == "explain_dashboard":
        # Static response, no LLM needed
        return build_explain_response(), intent
    
    else:
        return build_unsupported_response(), "unsupported"
    
    # Call LLM as formatter ONLY
    llm_reply = call_planning_llm(message, tool_summary)
    
    if llm_reply is None:
        # Graceful fallback: tool_summary is already formatted with bullets
        return tool_summary, intent
    
    return llm_reply, intent
