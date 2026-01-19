#!/usr/bin/env python3
"""
OpenAI LLM Integration Module

This module provides functions to create and manage chat sessions with
OpenAI's language models, including sending messages and handling
PDF/document attachments.
"""

# --- Third-Party Imports ---
import openai

# --- Local Imports ---
from ..utils.logging import logger
from ..utils.chat_session import LLMChatSession
from ..imports import (
    Dict,
    Optional,
    Any,
    Path,
    OPENAI_API_KEY
)


def create_openai_chat_session(
    model_name: str = "gpt-4.1-nano-2025-04-14",
    generation_options: Optional[Dict[str, Any]] = None,
    system_prompt: Optional[str] = None,
    api_key: Optional[str] = OPENAI_API_KEY
) -> LLMChatSession:
    """
    Creates a persistent OpenAI chat session with configured parameters.

    Args:
        model_name: OpenAI model to use (e.g., "gpt-4", "gpt-3.5-turbo")
        generation_options: Dictionary of OpenAI generation parameters (temperature, max_tokens, etc.)
        system_prompt: Optional system prompt to define the model's behavior
        api_key: OpenAI API key. If not provided, will fall back to OPENAI_API_KEY env var

    Returns:
        Initialized LLMChatSession object
    """
    if not api_key:
        logger.error("No OpenAI API key provided and OPENAI_API_KEY env var not set")
        raise ValueError("API key required for OpenAI. Set via parameter or OPENAI_API_KEY env var")

    # Initialize the OpenAI Client
    openai.api_key = api_key
    client = openai.OpenAI()

    # Set default generation options if none provided
    default_options = {
        "temperature": 0.7,
        "max_tokens": 1024,
        "top_p": 0.9
    }
    final_options = {**default_options, **(generation_options or {})}

    # Create LLMChatSession with OpenAI-specific attributes
    session = LLMChatSession(
        model_name=model_name,
        generation_options=final_options,
        chat_history=[],
        openai_context=[],
        pdf_prompt="",
        openai_client = client,
        api_key=api_key,
    )

    # Store system prompt in history for reference
    if system_prompt:
        context_message = {"role": "system", "content": system_prompt}
        session.chat_history.append(context_message)
        session.openai_context.append(context_message)

    logger.info(f"OpenAI chat session created with model: {model_name}")
    return session


def send_openai_message(
    session: LLMChatSession,
    user_message: str,
    stream: bool = False
) -> str:
    """
    Sends a message to the OpenAI model and returns the response.
    Maintains chat history for context-aware conversations.

    Args:
        session: Active LLMChatSession object
        user_message: User's input message
        stream: If True, returns incremental responses (default: False)

    Returns:
        Full assistant response text
    """
    if not session.is_active:
        logger.error("Cannot send message: OpenAI session is closed")
        raise RuntimeError("Cannot send message: OpenAI session is closed")

    if not hasattr(session, 'openai_client'):
        logger.error("Invalid session: Missing OpenAI client instance")
        raise ValueError("Session was not created with openai.OpenAI() client")

    # Add user message to our wrapper history
    user_message = {"role": "user", "content": user_message}
    session.chat_history.append(user_message)
    user_message = session.openai_context + [user_message]

    try:
        if stream:
            # Streaming response
            response_stream = session.openai_client.chat.completions.create(
                model=session.model_name,
                messages=user_message,
                stream=True,
                **session.generation_options
            )
            full_response = ""
            for chunk in response_stream:
                # chunk is typically a dict with choices[0].delta.content
                if not isinstance(chunk, dict):
                    continue
                choices = chunk.get("choices")
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                text = delta.get("content")
                if text:
                    full_response += text
                    logger.debug(chunk.text, end="", flush=True)
        else:
            # Non-streaming response
            response = session.openai_client.chat.completions.create(
                model=session.model_name,
                messages={"role": "user", "content": user_message},
                **session.generation_options
            )
            full_response = response.choices[0].message.content if response.choices else ""
        
        # Add assistant response to history
        session.chat_history.append({"role": "assistant", "content": full_response})
        
        logger.debug(f"OpenAI response received: {len(full_response)} characters")
        return full_response
    
    except Exception as e:
        logger.error(f"Error calling OpenAI API: {e}")
        raise RuntimeError(f"Failed to get response from OpenAI: {e}")


def send_openai_message_with_pdf(
    session: LLMChatSession,
    user_message: str,
    pdf_path: str,
    stream: bool = False
) -> str:
    """
    Sends a message along with a PDF document to the OpenAI model.
    
    Args:
        session: Active LLMChatSession object with OpenAI attributes
        user_message: User's input message about the PDF
        pdf_path: Path to the PDF file to analyze
        stream: If True, returns incremental responses (default: False)
    
    Returns:
        Full assistant response text
    """
    if not session.is_active:
        raise RuntimeError("Cannot send message: OpenAI session is closed")

    if not hasattr(session, 'openai_client'):
        raise ValueError("Session was not created with openai.OpenAI() client")

    # Validate PDF file exists
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        logger.error(f"PDF file not found: {pdf_path}")
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    try:
        # Upload the PDF file to OpenAI
        logger.info(f"Uploading PDF file: {pdf_path}")
        
        # Open file in binary mode
        with open(pdf_file, "rb") as f:
            uploaded_file = session.openai_client.files.create(
                file=f,
                purpose="user_data"
            )

        logger.info(f"PDF uploaded successfully: {uploaded_file.name}")

        # Prepare message with attachment reference
        user_input_with_attachment = session.openai_context + [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_file",
                        "file_id": uploaded_file.id,
                    },
                    {
                        "type": "input_text",
                        "text": user_message,
                    },
                ]
            }
        ]
        
        if stream:
            # Streaming response
            response_stream = session.openai_client.chat.completions.create(
                model=session.model_name,
                input=user_input_with_attachment,
                stream=True,
                **session.generation_options
            )
            full_response = ""
            for chunk in response_stream:
                # chunk is typically a dict with choices[0].delta.content
                if not isinstance(chunk, dict):
                    continue
                choices = chunk.get("choices")
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                text = delta.get("content")
                if text:
                    full_response += text
                    logger.debug(chunk.text, end="", flush=True)
        else:
            # Non-streaming response
            response = session.openai_client.chat.completions.create(
                model=session.model_name,
                input=user_input_with_attachment,
                **session.generation_options
            )
            full_response = response.output_text

        # Update wrapper history
        session.chat_history.append({
            "role": "user", 
            "content": user_message,
            "attachments": [{"type": "pdf", "path": pdf_path, "file_id": uploaded_file.name}]
        })
        session.chat_history.append({"role": "assistant", "content": full_response})
        
        return full_response
        
    except Exception as e:
        logger.error(f"Error processing PDF with OpenAI: {e}")
        raise RuntimeError(f"Failed to process PDF with OpenAI: {e}")


def close_openai_session(
    session: LLMChatSession
) -> None:
    """
    Cleans up and closes the OpenAI chat session.
    """
    session.is_active = False
    
    # Clear OpenAI-specific attributes
    if hasattr(session, 'openai_client'):
        delattr(session, 'openai_client')
    if hasattr(session, 'openai_context'):
        delattr(session, 'openai_context')
    if hasattr(session, 'api_key'):
        delattr(session, 'api_key')

    # Free up memory
    session.chat_history.clear()
    logger.info("OpenAI session closed successfully.")
