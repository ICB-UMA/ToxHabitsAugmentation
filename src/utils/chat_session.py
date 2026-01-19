#!/usr/bin/env python3
"""
Chat Session Data Structure for LLM Interactions.

This module defines a structured object to maintain the state of a chat session
with a Large Language Model (LLM), including model details, generation options,
chat history, and session status.
"""

# --- Third-Party Imports ---
from dataclasses import dataclass

# --- Local Imports ---
from ..imports import (
    Dict,
    List,
    Any
)


@dataclass
class LLMChatSession:
    """Structured object to track LLM chat session state"""
    model_name: str
    generation_options: Dict[str, Any]
    chat_history: List[Dict[str, str]]
    is_active: bool = True
    openai_client: Any = None
    openai_context: List[Dict[str, str]] = []
    gemini_client: Any = None
    gemini_chat: Any = None
    api_key: str = ""
