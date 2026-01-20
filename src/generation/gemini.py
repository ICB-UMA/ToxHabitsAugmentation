#!/usr/bin/env python3
"""
Gemini LLM Integration Module

This module provides functions to create and manage chat sessions with
Google's Gemini language models, including sending messages and handling
PDF/document attachments.
"""

# --- Third-Party Imports ---
from http import client
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
    developer_prompt: Optional[str] = None,
    pdf_path: Optional[str] = None,
    api_key: str = GOOGLE_API_KEY
) -> LLMChatSession:
    """
    Creates a persistent Gemini chat session with configured parameters.

    Args:
        model_name: Name of the Gemini model to use (e.g. "gemini-1.5-flash", "gemini-1.5-pro")
        generation_options: Dictionary of Gemini generation parameters (temperature, max_tokens, etc.)
        system_prompt: Optional system prompt to define the model's behavior
        developer_prompt: Optional developer prompt to set initial context
        pdf_path: Optional path to a PDF document to upload and reference
        api_key: Google AI API key (if not provided, will look for GOOGLE_API_KEY env var)

    Returns:
        Initialized LLMChatSession object with Gemini-specific attributes
    """
    if not api_key:
        logger.error("No API key provided and GOOGLE_API_KEY env var not set")
        raise ValueError("API key required for Gemini. Set via parameter or GOOGLE_API_KEY env var")

    # Initialize the v1 Client
    client = genai.Client(api_key=api_key)
    cache = None
    chat = None

    # Set default generation options if none provided
    default_options = {
        "temperature": 0.7,
        "max_output_tokens": 1024,
        "top_p": 0.9,
        "top_k": 40,
    }
    final_options = {**default_options, **(generation_options or {})}

    # Create LLMChatSession with Gemini-specific attributes
    session = LLMChatSession(
        model_name=model_name,
        generation_options=final_options,
        chat_history=[],
        gemini_client = client,
        gemini_cache = None,
        api_key = api_key
    )

    # Prepare Gemini client if developer prompt provided
    if developer_prompt:
        if pdf_path:
            try:
                # Validate PDF file exists
                pdf_file = Path(pdf_path)
                if not pdf_file.exists():
                    logger.error(f"PDF file not found: {pdf_path}")
                    raise FileNotFoundError(f"PDF file not found: {pdf_path}")

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
        
                # Create a cached content object
                cache = client.caches.create(
                    model=model_name,
                    config=types.GenerateContentConfig(
                        system_instruction=developer_prompt,
                        contents=[uploaded_file],
                        temperature=final_options.get("temperature"),
                        max_output_tokens=final_options.get("max_output_tokens"),
                        top_p=final_options.get("top_p"),
                        top_k=final_options.get("top_k"),
                    )
                )

            except Exception as e:
                logger.error(f"Error processing PDF with Gemini: {e}")
                raise RuntimeError(f"Failed to process PDF with Gemini: {e}")
        else:
            try:
                # Create a cached content object without PDF
                cache = client.caches.create(
                    model=model_name,
                    config=types.GenerateContentConfig(
                        system_instruction=developer_prompt,
                        temperature=final_options.get("temperature"),
                        max_output_tokens=final_options.get("max_output_tokens"),
                        top_p=final_options.get("top_p"),
                        top_k=final_options.get("top_k"),
                    )
                )
            except genai.errors.ClientError as e:
                logger.error(f"Error creating Gemini cache: {e}")

                # Fallback: Create chat without cache
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

        # Store cache and chat in session
        session.gemini_cache = cache
        session.gemini_chat = chat
        session.chat_history.append({"role": "system", "content": developer_prompt})

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

    if not hasattr(session, 'gemini_client'):
        logger.error("Invalid session: Missing Gemini client instance")
        raise ValueError("Session was not created with genai.Client")

    if not hasattr(session, 'gemini_cache'):
        logger.error("Invalid session: Missing Gemini cache instance")
        raise ValueError("Session was not created with client.caches.create")

    # Add user message to our wrapper history
    session.chat_history.append({"role": "user", "content": user_message})

    try:
        if stream:
            # The new SDK returns a generator for stream=True
            response_stream = session.gemini_client.send_message_stream(user_message)
            full_response = ""
            for chunk in response_stream:
                if chunk.text:
                    full_response += chunk.text
                    logger.debug(chunk.text, end="", flush=True)
        else:
            if session.gemini_cache:
                # Use cache instance if available
                response = session.gemini_client.models.generate_content(
                    model=session.model_name,
                    contents=user_message,
                    config=types.GenerateContentConfig(
                        cached_content=session.gemini_cache.name,
                        temperature=session.final_options.get("temperature"),
                        max_output_tokens=session.final_options.get("max_output_tokens"),
                        top_p=session.final_options.get("top_p"),
                        top_k=session.final_options.get("top_k")
                    )
                )
            else:
                # Fallback to chat instance
                response = session.gemini_chat.send_message(user_message)
            full_response = response.text
        
        # Add assistant response to history
        session.chat_history.append({"role": "assistant", "content": full_response})
        
        logger.debug(f"Gemini response received: {len(full_response)} characters")
        time.sleep(30)  # To avoid rate limits
        return full_response
    
    except Exception as e:
        logger.error(f"Error calling Gemini API: {e}")
        raise RuntimeError(f"Failed to get response from Gemini: {e}")


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
