#!/usr/bin/env python3
"""
Hugging Face Generation Utilities

This module provides functions to create and manage chat sessions
with Hugging Face models, send messages, and handle responses.
"""

# --- Third-Party Imports ---
from threading import Thread
from transformers import (
    AutoModel,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    TextIteratorStreamer
)

# --- Local Imports ---
from ..utils.logging import logger
from ..utils.chat_session import LLMChatSession
from ..imports import (
    torch,
    Dict,
    Optional,
    Any,
    RANDOM_SEED
)


def create_huggingface_chat_session(
    model_name: str,
    generation_options: Optional[Dict[str, Any]] = None,
    system_prompt: Optional[str] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> LLMChatSession:
    """
    Creates a persistent Hugging Face chat session with configured parameters.

    Args:
        model_name: Name of the Hugging Face model to use (e.g. "llama3", "mistral")
        generation_options: Dictionary of Hugging Face generation parameters (temperature, max_tokens, etc.)
        system_prompt: Optional system prompt to define the model's behavior
        device: Device to load the model on ("cuda" or "cpu")

    Returns:
        Initialized LLMChatSession object
    """
    # Set default generation options adapted for Hugging Face generate()
    default_options = {
        "temperature": 0.7,
        "max_new_tokens": 1024,
        "top_p": 0.9,
        "do_sample": True, # Required in HF when using temperature/top_p
        "pad_token_id": 2  # Default for Mistral architectures
    }
    final_options = {**default_options, **(generation_options or {})}
    
    if "seed" in final_options:
        torch.manual_seed(final_options.pop("seed"))
    else:
        torch.manual_seed(RANDOM_SEED)

    try:
        # Load Tokenizer
        logger.info(f"Loading tokenizer and model for {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        if model_name in ["HiTZ/Medical-mT5-large"]:
            # This model is an encoder-decoder architecture, so we use AutoModelForSeq2SeqLM
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                device_map=device,
                trust_remote_code=True,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
        else:
            # For decoder-only models (like LLaMA/Mistral), we use AutoModel
            model = AutoModel.from_pretrained(
                model_name,
                device_map=device,
                trust_remote_code=True,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )

    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise RuntimeError(f"Failed to load model {model_name}: {str(e)}")

    # Initialize chat history
    chat_history = []
    if system_prompt:
        chat_history.append({"role": "system", "content": system_prompt})

    return LLMChatSession(
        model_name=model_name,
        generation_options=final_options,
        chat_history=chat_history,
        model=model,
        tokenizer=tokenizer
    )


def send_huggingface_message(
    session: LLMChatSession,
    user_message: str,
    stream: bool = False
) -> str:
    """
    Sends a message to the Hugging Face model and returns the response.
    Maintains chat history for context-aware conversations.

    Args:
        session: Active LLMChatSession object
        user_message: User's input message
        stream: If True, returns incremental responses (default: False)

    Returns:
        Full assistant response text
    """
    if not session.is_active or session.model is None:
        logger.error("Cannot send message: Hugging Face session is closed")
        raise RuntimeError("Cannot send message: Hugging Face session is closed")

    # Add user message to chat history
    session.chat_history.append({"role": "user", "content": user_message})

    try:
        # Format the chat history using the model's built-in chat template
        input_ids = session.tokenizer.apply_chat_template(
            session.chat_history,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(session.model.device)

        full_response = ""

        # Handle streaming responses using TextIteratorStreamer and threading
        if stream:
            streamer = TextIteratorStreamer(
                session.tokenizer, 
                skip_prompt=True, 
                skip_special_tokens=True
            )
            generation_kwargs = dict(
                input_ids=input_ids,
                streamer=streamer,
                **session.generation_options
            )
            
            # Generation must run in a background thread to yield chunks to the streamer
            thread = Thread(target=session.model.generate, kwargs=generation_kwargs)
            thread.start()

            for chunk in streamer:
                full_response += chunk
                logger.debug(chunk, end="", flush=True) 
            logger.debug() # Print newline when stream finishes

        # Handle standard blocking response
        else:
            outputs = session.model.generate(
                input_ids,
                **session.generation_options
            )
            
            # Slice the output to only get the newly generated tokens (ignore the prompt)
            new_tokens = outputs[0][input_ids.shape[1]:]
            full_response = session.tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Add assistant response to chat history
        session.chat_history.append({"role": "assistant", "content": full_response})

        return full_response

    except Exception as e:
        logger.error(f"Error during Hugging Face generation: {str(e)}")
        raise RuntimeError(f"Generation failed: {str(e)}")


def close_huggingface_session(
    session: LLMChatSession
) -> None:
    """
    Cleans up and closes the Hugging Face chat session.
    Frees up the large amount of RAM/VRAM consumed by the 7B model.
    """
    session.is_active = False
    session.chat_history.clear()
    
    # Delete model and tokenizer references
    del session.model
    del session.tokenizer
    session.model = None
    session.tokenizer = None
    
    # Clear CUDA cache if GPU was used
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    logger.info("Hugging Face session closed successfully and VRAM freed.")
