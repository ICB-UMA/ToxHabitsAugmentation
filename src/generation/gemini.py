#!/usr/bin/env python3
"""
Gemini LLM Integration Module

This module provides functions to create and manage chat sessions with
Google's Gemini language models, including sending messages and handling
PDF/document attachments.
"""

# --- Third-Party Imports ---
import mimetypes
from google import genai
from google.genai import types

# --- Local Imports ---
from ..utils.logging import logger
from ..utils.chat_session import LLMChatSession
from ..imports import (
    time,
    Dict,
    Optional,
    Any,
    List,
    Path,
    GOOGLE_API_KEY
)


def create_gemini_chat_session(
    model_name: str = "gemini-1.5-flash",
    generation_options: Optional[Dict[str, Any]] = None,
    system_prompt: Optional[str] = None,
    api_key: str = GOOGLE_API_KEY
) -> LLMChatSession:
    """
    Creates a persistent Gemini chat session with configured parameters.

    Args:
        model_name: Name of the Gemini model to use (e.g. "gemini-1.5-flash", "gemini-1.5-pro")
        generation_options: Dictionary of Gemini generation parameters (temperature, max_tokens, etc.)
        system_prompt: Optional system prompt to define the model's behavior
        api_key: Google AI API key (if not provided, will look for GOOGLE_API_KEY env var)

    Returns:
        Initialized LLMChatSession object with Gemini-specific attributes
    """
    if not api_key:
        logger.error("No API key provided and GOOGLE_API_KEY env var not set")
        raise ValueError("API key required for Gemini. Set via parameter or GOOGLE_API_KEY env var")

    # Initialize the v1 Client
    client = genai.Client(api_key=api_key)

    # Set default generation options if none provided
    default_options = {
        "temperature": 0.7,
        "max_output_tokens": 1024,
        "top_p": 0.9,
        "top_k": 40,
    }
    final_options = {**default_options, **(generation_options or {})}

    # Create the chat object
    chat = client.chats.create(
        model=model_name,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=final_options.get("temperature"),
            max_output_tokens=final_options.get("max_output_tokens"),
            top_p=final_options.get("top_p"),
            top_k=final_options.get("top_k"),
        ),
        history=[]
    )

    # Create LLMChatSession with Gemini-specific attributes
    session = LLMChatSession(
        model_name=model_name,
        generation_options=final_options,
        chat_history=[],
        gemini_client = client,
        gemini_chat = chat,
        api_key = api_key
    )

    # Store system prompt in history for reference
    if system_prompt:
        session.chat_history.append({"role": "system", "content": system_prompt})

    logger.info(f"Gemini chat session created with model: {model_name}")
    return session


def send_gemini_message(
    session: LLMChatSession,
    user_message: str,
    stream: bool = False
) -> str:
    """
    Sends a message to the Gemini model and returns the response.
    Maintains chat history for context-aware conversations.

    Args:
        session: Active LLMChatSession object with Gemini attributes
        user_message: User's input message
        stream: If True, returns incremental responses (default: False)

    Returns:
        Full assistant response text
    """
    if not session.is_active:
        logger.error("Cannot send message: Gemini session is closed")
        raise RuntimeError("Cannot send message: Gemini session is closed")

    if not hasattr(session, 'gemini_chat'):
        logger.error("Invalid session: Missing Gemini chat instance")
        raise ValueError("Session was not created with client.chats.create")

    # Add user message to our wrapper history
    session.chat_history.append({"role": "user", "content": user_message})

    try:
        if stream:
            # The new SDK returns a generator for stream=True
            response_stream = session.gemini_chat.send_message_stream(user_message)
            full_response = ""
            for chunk in response_stream:
                if chunk.text:
                    full_response += chunk.text
                    logger.debug(chunk.text, end="", flush=True)
        else:
            response = session.gemini_chat.send_message(user_message)
            full_response = response.text
        
        # Add assistant response to history
        session.chat_history.append({"role": "assistant", "content": full_response})
        
        logger.debug(f"Gemini response received: {len(full_response)} characters")
        return full_response
    
    except Exception as e:
        logger.error(f"Error calling Gemini API: {e}")
        raise RuntimeError(f"Failed to get response from Gemini: {e}")


def send_gemini_message_with_pdf(
    session: LLMChatSession,
    user_message: str,
    pdf_path: str,
    stream: bool = False
) -> str:
    """
    Sends a message along with a PDF document to the Gemini model.
    
    Args:
        session: Active LLMChatSession object with Gemini attributes
        user_message: User's input message about the PDF
        pdf_path: Path to the PDF file to analyze
        stream: If True, returns incremental responses (default: False)
    
    Returns:
        Full assistant response text
    """
    if not session.is_active:
        raise RuntimeError("Cannot send message: Gemini session is closed")

    if not hasattr(session, 'gemini_client'):
        raise ValueError("Session was not created with genai.Client")

    # Validate PDF file exists
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        logger.error(f"PDF file not found: {pdf_path}")
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    try:
        # Upload the PDF file to Gemini
        logger.info(f"Uploading PDF file: {pdf_path}")
        
        # Open file in binary mode
        with open(pdf_file, "rb") as f:
            uploaded_file = session.gemini_client.files.upload(
                file=f,
                config=types.UploadFileConfig(
                    display_name=pdf_file.name,
                    mime_type="application/pdf"
                )
            )

        # Wait for file to be processed
        while uploaded_file.state == "PROCESSING":
            time.sleep(2)
            uploaded_file = session.gemini_client.files.get(name=uploaded_file.name)
        
        if uploaded_file.state == "FAILED":
            raise RuntimeError("Failed to process PDF file")
        
        logger.info(f"PDF uploaded successfully: {uploaded_file.name}")
        
        # In the new SDK, pass content as a list of Parts or similar structure
        # Note: The Chat object in SDK v1 maintains history. To insert a file
        # into the flow, we send it as part of the message content
        
        message_content = [
            types.Part.from_uri(
                file_uri=uploaded_file.uri,
                mime_type=uploaded_file.mime_type
            ),
            types.Part.from_text(text=user_message)
        ]

        if stream:
            # The new SDK returns a generator for stream=True
            response_stream = session.gemini_chat.send_message_stream(message_content)
            full_response = ""
            for chunk in response_stream:
                if chunk.text:
                    full_response += chunk.text
                    logger.debug(chunk.text, end="", flush=True)
        else:
            response = session.gemini_chat.send_message(message_content)
            full_response = response.text

        # Update wrapper history
        session.chat_history.append({
            "role": "user", 
            "content": user_message,
            "attachments": [{"type": "pdf", "path": pdf_path, "file_id": uploaded_file.name}]
        })
        session.chat_history.append({"role": "assistant", "content": full_response})
        
        return full_response
        
    except Exception as e:
        logger.error(f"Error processing PDF with Gemini: {e}")
        raise RuntimeError(f"Failed to process PDF with Gemini: {e}")


def send_gemini_message_with_documents(
    session: LLMChatSession,
    user_message: str,
    document_paths: List[str],
    stream: bool = False
) -> str:
    """
    Sends a message along with multiple documents (PDFs, images, etc.) to Gemini.
    
    Args:
        session: Active LLMChatSession object with Gemini attributes
        user_message: User's input message about the documents
        document_paths: List of paths to documents to analyze
        stream: If True, returns incremental responses (default: False)
    
    Returns:
        Full assistant response text
    """
    if not session.is_active:
        raise RuntimeError("Cannot send message: Gemini session is closed")

    message_parts = []
    uploaded_files_meta = []

    try:
        # Upload all documents
        for doc_path in document_paths:
            doc_file = Path(doc_path)
            if not doc_file.exists():
                logger.warning(f"Document not found, skipping: {doc_path}")
                continue
            
            # Determine mime type
            mime_type, _ = mimetypes.guess_type(doc_path)
            if not mime_type:
                mime_type = "application/octet-stream"

            with open(doc_file, "rb") as f:
                uploaded_file = session.gemini_client.files.upload(
                    file=f,
                    config=types.UploadFileConfig(
                        display_name=doc_file.name,
                        mime_type=mime_type
                    )
                )

            # Wait for processing
            while uploaded_file.state == "PROCESSING":
                time.sleep(1)
                uploaded_file = session.gemini_client.files.get(name=uploaded_file.name)
            if uploaded_file.state == "FAILED":
                logger.error(f"Upload failed for {doc_file.name}")
                continue
            
            # Add to message parts
            message_parts.append(
                types.Part.from_uri(
                    file_uri=uploaded_file.uri,
                    mime_type=uploaded_file.mime_type
                )
            )
            uploaded_files_meta.append({"path": doc_path, "file_id": uploaded_file.name})
            logger.info(f"Successfully uploaded: {doc_file.name}")

        if not message_parts and document_paths:
             logger.warning("No documents were successfully uploaded")
        
        # Add text prompt to parts
        message_parts.append(types.Part.from_text(text=user_message))

        # Send message with all documents
        if stream:
            response_stream = session.gemini_chat.send_message_stream(message_parts)
            full_response = ""
            for chunk in response_stream:
                if chunk.text:
                    full_response += chunk.text
                    logger.debug(chunk.text, end="", flush=True)
        else:
            response = session.gemini_chat.send_message(message_parts)
            full_response = response.text

        # Update history
        session.chat_history.append({
            "role": "user",
            "content": user_message,
            "attachments": uploaded_files_meta
        })
        session.chat_history.append({"role": "assistant", "content": full_response})
        
        return full_response
        
    except Exception as e:
        logger.error(f"Error processing documents with Gemini: {e}")
        raise RuntimeError(f"Failed to process documents: {e}")


def close_gemini_session(
    session: LLMChatSession
) -> None:
    """
    Cleans up and closes the Gemini chat session.
    """
    session.is_active = False
    
    # Clear Gemini-specific attributes
    if hasattr(session, 'gemini_chat'):
        delattr(session, 'gemini_chat')
    if hasattr(session, 'gemini_client'):
        delattr(session, 'gemini_client')
    if hasattr(session, 'api_key'):
        delattr(session, 'api_key')

    # Free up memory
    session.chat_history.clear()
    logger.info("Gemini session closed successfully.")
