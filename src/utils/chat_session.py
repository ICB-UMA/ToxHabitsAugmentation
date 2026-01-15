#!/usr/bin/env python3
"""
...
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
    """Structured object to track Ollama chat session state"""
    model_name: str
    generation_options: Dict[str, Any]
    chat_history: List[Dict[str, str]]
    is_active: bool = True
