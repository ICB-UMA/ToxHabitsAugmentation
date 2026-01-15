#!/usr/bin/env python3
"""
...
"""

# --- Third-Party Imports ---
from requests.exceptions import ConnectionError as OllamaConnectionError

# --- Local Imports ---
from ..utils.logging import logger
from ..utils.chat_session import LLMChatSession
from ..imports import (
    ollama,
    Dict,
    Optional,
    Any,
    RANDOM_SEED
)


def create_ollama_chat_session(
    model_name: str,
    generation_options: Optional[Dict[str, Any]] = None,
    system_prompt: Optional[str] = None
) -> LLMChatSession:
    """
    Creates a persistent Ollama chat session with configured parameters.

    Args:
        model_name: Name of the Ollama model to use (e.g. "llama3", "mistral")
        generation_options: Dictionary of Ollama generation parameters (temperature, max_tokens, etc.)
        system_prompt: Optional system prompt to define the model's behavior

    Returns:
        Initialized LLMChatSession object
    """
    # Set default generation options if none provided
    default_options = {
        "temperature": 0.7,
        "max_tokens": 1024,
        "top_p": 0.9,
        "seed": RANDOM_SEED
    }
    final_options = {**default_options, **(generation_options or {})}

    # Initialize chat history
    chat_history = []
    if system_prompt:
        chat_history.append({"role": "system", "content": system_prompt})

    return LLMChatSession(
        model_name=model_name,
        generation_options=final_options,
        chat_history=chat_history
    )


def send_ollama_message(
    session: LLMChatSession,
    user_message: str,
    stream: bool = False
) -> str:
    """
    Sends a message to the Ollama model and returns the response.
    Maintains chat history for context-aware conversations.

    Args:
        session: Active LLMChatSession object
        user_message: User's input message
        stream: If True, returns incremental responses (default: False)

    Returns:
        Full assistant response text
    """
    if not session.is_active:
        logger.error("Cannot send message: Ollama session is closed")
        raise RuntimeError("Cannot send message: Ollama session is closed")

    # Add user message to chat history
    session.chat_history.append({"role": "user", "content": user_message})

    try:
        # Send request to Ollama server
        response = ollama.chat(
            model=session.model_name,
            messages=session.chat_history,
            options=session.generation_options,
            stream=stream
        )

        # Handle streaming responses
        full_response = ""
        if stream:
            for chunk in response:
                chunk_content = chunk["message"]["content"]
                full_response += chunk_content
                logger.debug(chunk_content, end="", flush=True)
        else:
            full_response = response["message"]["content"]

        # Add assistant response to chat history
        session.chat_history.append({"role": "assistant", "content": full_response})

        return full_response

    except OllamaConnectionError:
        logger.error("Could not connect to Ollama server. Ensure `ollama serve` is running.")
        raise ConnectionError("Could not connect to Ollama server. Ensure `ollama serve` is running.")


def close_ollama_session(
    session: LLMChatSession
) -> None:
    """
    Cleans up and closes the Ollama chat session.
    The Ollama server runs independently, so this only cleans up local state.
    """
    session.is_active = False
    session.chat_history.clear()  # Free up memory
    logger.info("Ollama session closed successfully.")
