# src/planning/planning_tab.py
"""
Planning Tab UI
===============

Streamlit component for the UIDAI Planning Assistant tab.

This module provides:
    - planning_tab(): Main Streamlit UI for the planning assistant
    - Chat interface with message history
    - Suggested prompts for quick access
    - Safety information sidebar

╔════════════════════════════════════════════════════════════════════════════╗
║  PHASE 1: Uses handle_planning_query() with Ollama LLM formatter.         ║
║  All numbers come from internal tools, LLM only formats the output.       ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import streamlit as st

from .planning_assistant_spec import (
    PLANNING_ASSISTANT_NAME,
    PLANNING_ASSISTANT_DESCRIPTION,
)
from .planning_llm import check_ollama_available, MODEL_NAME
from .planning_router import handle_planning_query


# =============================================================================
# SUGGESTED PROMPTS
# =============================================================================

SUGGESTED_PROMPTS = [
    {
        "label": "Top over-capacity states in 2025-04",
        "query": "Which states are over capacity in 2025-04?",
    },
    {
        "label": "Impact of adding 2 centres in Tamil Nadu",
        "query": "If we add 2 extra centres in Tamil Nadu in 2025-04, what happens?",
    },
    {
        "label": "Capacity gaps in 2025-03",
        "query": "Show me the top states with capacity gaps in 2025-03",
    },
    {
        "label": "Simulate 3 centres in Karnataka",
        "query": "What if we add 3 camps in Karnataka next month?",
    },
]


# =============================================================================
# CSS STYLING
# =============================================================================

PLANNING_TAB_CSS = """
<style>
/* Chat message styling */
.planning-chat-container {
    max-height: 500px;
    overflow-y: auto;
    padding: 1rem;
    background-color: #f9fafb;
    border-radius: 8px;
    margin-bottom: 1rem;
}

.planning-user-msg {
    background-color: #e0f2fe;
    padding: 0.75rem 1rem;
    border-radius: 12px;
    margin-bottom: 0.5rem;
    max-width: 80%;
    margin-left: auto;
}

.planning-assistant-msg {
    background-color: #ffffff;
    padding: 0.75rem 1rem;
    border-radius: 12px;
    margin-bottom: 0.5rem;
    max-width: 80%;
    border: 1px solid #e2e8f0;
}

/* Suggested prompts */
.suggested-prompt-btn {
    background-color: #f1f5f9;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-size: 0.85rem;
    cursor: pointer;
    transition: all 0.2s;
}

.suggested-prompt-btn:hover {
    background-color: #e0f2fe;
    border-color: #0076A8;
}

/* Safety box */
.planning-safety-box {
    background-color: #fef3c7;
    border-left: 4px solid #f59e0b;
    padding: 0.75rem 1rem;
    border-radius: 4px;
    font-size: 0.85rem;
}

/* Info box */
.planning-info-box {
    background-color: #e0f2fe;
    border-left: 4px solid #0076A8;
    padding: 0.75rem 1rem;
    border-radius: 4px;
    font-size: 0.85rem;
}
</style>
"""


# =============================================================================
# SESSION STATE MANAGEMENT
# =============================================================================

def init_planning_session():
    """Initialize session state for planning assistant."""
    if "planning_messages" not in st.session_state:
        st.session_state.planning_messages = []
    if "planning_input_key" not in st.session_state:
        st.session_state.planning_input_key = 0
    if "planning_use_llm" not in st.session_state:
        st.session_state.planning_use_llm = True
    if "ollama_available" not in st.session_state:
        st.session_state.ollama_available = None  # Will be checked on first load


def add_message(role: str, content: str):
    """Add a message to the planning chat history."""
    st.session_state.planning_messages.append({"role": role, "content": content})


def clear_messages():
    """Clear all planning chat messages."""
    st.session_state.planning_messages = []


# =============================================================================
# MESSAGE PROCESSING
# =============================================================================

def process_user_message(message: str):
    """
    Process a user message and generate assistant response.
    
    Uses Phase 1 handler with LLM formatting.
    Falls back gracefully if Ollama is unavailable.
    
    Args:
        message: User's input message
    """
    # Get response from router (Phase 1 with LLM formatter)
    reply, intent = handle_planning_query(message)
    
    # Add both messages to history using += (not append twice)
    st.session_state.planning_messages += [
        {"role": "user", "content": message},
        {"role": "assistant", "content": reply},
    ]


# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_chat_history():
    """Render the chat message history."""
    for msg in st.session_state.planning_messages:
        role = msg["role"]
        content = msg["content"]
        
        with st.chat_message(role):
            st.markdown(content)


def render_suggested_prompts():
    """Render suggested prompt buttons."""
    st.markdown("**💡 Try one of these:**")
    
    cols = st.columns(2)
    for i, prompt in enumerate(SUGGESTED_PROMPTS):
        col_idx = i % 2
        with cols[col_idx]:
            if st.button(prompt["label"], key=f"prompt_{i}", use_container_width=True):
                process_user_message(prompt["query"])
                st.rerun()


def render_safety_sidebar():
    """Render safety information in sidebar."""
    st.markdown(
        """
        <div class="planning-info-box">
            <strong>📋 What you can ask:</strong>
            <ul style="margin: 0.5rem 0 0 1rem; padding: 0;">
                <li>Over-capacity states for a month (YYYY-MM)</li>
                <li>Top districts with highest capacity gap</li>
                <li>Impact of adding extra centres/camps</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown("")
    
    st.markdown(
        """
        <div class="planning-safety-box">
            <strong>🔒 Safety rules:</strong>
            <ul style="margin: 0.5rem 0 0 1rem; padding: 0;">
                <li>No Aadhaar or citizen-level queries</li>
                <li>No raw data exports</li>
                <li>Read-only, planning recommendations only</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_llm_status():
    """Render LLM status and toggle in sidebar."""
    # Check Ollama status (cache in session state)
    if st.session_state.ollama_available is None:
        st.session_state.ollama_available = check_ollama_available()
    
    ollama_ok = st.session_state.ollama_available
    
    # LLM toggle
    st.session_state.planning_use_llm = st.toggle(
        "Use LLM formatting",
        value=st.session_state.planning_use_llm,
        help="When enabled, responses are formatted by Ollama LLM. When disabled, raw tool summaries are shown.",
    )
    
    # Status indicator
    if ollama_ok:
        status_color = "#22c55e"  # green
        status_text = "Connected"
        status_icon = "🟢"
    else:
        status_color = "#ef4444"  # red
        status_text = "Unavailable"
        status_icon = "🔴"
    
    st.markdown(
        f"""
        <div style="font-size: 0.85rem; margin-top: 0.5rem;">
            <strong>Ollama:</strong> {status_icon} {status_text}<br>
            <span style="color: #64748b;">Model: {MODEL_NAME}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Refresh button
    if st.button("🔄 Check Ollama", use_container_width=True):
        st.session_state.ollama_available = check_ollama_available()
        st.rerun()


# =============================================================================
# MAIN PLANNING TAB
# =============================================================================

def planning_tab():
    """
    Render the Planning Assistant tab.
    
    This is the main entry point for the planning assistant UI.
    """
    # Initialize session state
    init_planning_session()
    
    # Inject CSS
    st.markdown(PLANNING_TAB_CSS, unsafe_allow_html=True)
    
    # Spacer to push content below Streamlit header
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    
    # Header
    st.markdown(
        f"""
        <div class="uidai-page-header">🤖 {PLANNING_ASSISTANT_NAME}</div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Resource-planning helper for enrolment centres (Phase 1 – LLM formatter only).")
    
    # Layout: main chat area + sidebar
    col_main, col_sidebar = st.columns([2.5, 1])
    
    with col_sidebar:
        render_safety_sidebar()
        
        st.markdown("---")
        
        # LLM status and toggle
        st.markdown("**🤖 LLM Settings**")
        render_llm_status()
        
        st.markdown("---")
        
        # Clear chat button
        if st.button("🗑️ Clear chat", use_container_width=True):
            clear_messages()
            st.rerun()
        
        st.markdown("---")
        
        # Phase indicator
        st.markdown(
            """
            <div style="font-size: 0.8rem; color: #94a3b8;">
                <strong>Phase:</strong> 1 (LLM Formatter)<br>
                <strong>Mode:</strong> Tool → LLM → Response<br>
                <strong>Data:</strong> Demo/placeholder
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with col_main:
        # Status info for Phase 1
        st.info("Status: Phase 1 – LLM formats tool outputs. All numbers still come from internal tools.")
        
        # Chat container
        st.markdown('<div class="uidai-card">', unsafe_allow_html=True)
        
        # Show chat history
        if st.session_state.planning_messages:
            render_chat_history()
        else:
            # Welcome message
            st.markdown(
                """
                👋 **Welcome!** I'm here to help with enrolment centre planning.
                
                Ask me about:
                - Which states are over-capacity
                - Impact of adding extra centres
                - Capacity gaps for specific months
                
                *Note: This is a beta version using demo data.*
                """
            )
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Suggested prompts (if no messages yet)
        if not st.session_state.planning_messages:
            render_suggested_prompts()
        
        # Chat input
        user_input = st.chat_input(
            "Ask a planning question...",
            key=f"planning_input_{st.session_state.planning_input_key}",
        )
        
        if user_input:
            with st.spinner('🤖 Analyzing capacity data...'):
                process_user_message(user_input)
            st.rerun()
