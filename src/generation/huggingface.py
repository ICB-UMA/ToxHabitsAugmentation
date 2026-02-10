#!/usr/bin/env python3
"""
Hugging Face Generation Utilities

This module provides functions to create and manage chat sessions
with Hugging Face models, send messages, and handle responses.
"""

# --- Third-Party Imports ---
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer
)

# --- Local Imports ---
from ..utils.logging import logger
from ..utils.chat_session import LLMChatSession
from ..imports import (
    torch,
    Dict,
    Optional,
    Any,
    RANDOM_SEED,
    DEVICE
)


def create_huggingface_chat_session(
    model_name: str,
    generation_options: Optional[Dict[str, Any]] = None,
    system_prompt: Optional[str] = None,
) -> LLMChatSession:
    """
    Creates a persistent Hugging Face chat session with configured parameters.

    Args:
        model_name: Name of the Hugging Face model to use (e.g. "llama3", "mistral")
        generation_options: Dictionary of Hugging Face generation parameters (temperature, max_tokens, etc.)
        system_prompt: Optional system prompt to define the model's behavior

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
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_safetensors=False
        )

        if model_name in ["HiTZ/Medical-mT5-large"]:
            # This model is an encoder-decoder architecture, so we use AutoModelForSeq2SeqLM
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                device_map=DEVICE,
                trust_remote_code=True,
                use_safetensors=False,
                torch_dtype=torch.float16 if DEVICE != "cpu" else torch.float32
            )
        else:
            # For decoder-only models (like LLaMA/Mistral), we use AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=DEVICE,
                trust_remote_code=True,
                use_safetensors=False,
                torch_dtype=torch.float16 if DEVICE != "cpu" else torch.float32
            )

    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise RuntimeError(f"Failed to load model {model_name}: {str(e)}")

    # Initialize chat history
    chat_history = []

    return LLMChatSession(
        model_name=model_name,
        generation_options=final_options,
        chat_history=chat_history,
        hf_model=model,
        hf_tokenizer=tokenizer
    )


def send_huggingface_message(
    session: LLMChatSession,
    user_message: str,
    first_message: bool = False,
) -> str:
    """
    Sends a message to the Hugging Face model and returns the response.
    Maintains chat history for context-aware conversations.

    Args:
        session: Active LLMChatSession object
        user_message: User's input message
        first_message: Indicates if this is the first message in the session

    Returns:
        Full assistant response text
    """
    if not session.is_active or session.hf_model is None:
        logger.error("Cannot send message: Hugging Face session is closed")
        raise RuntimeError("Cannot send message: Hugging Face session is closed")

    # Add user message to chat history
    session.chat_history.append({"role": "user", "content": user_message})

    if session.model_name in ["HiTZ/Medical-mT5-large"]:
        try:
            # Format the chat history into a plain string
            formatted_prompt = session.hf_tokenizer.apply_chat_template(
                session.chat_history,
                add_generation_prompt=True,
                tokenize=False  # Get the string, not tokens
            )

            # Tokenize the formatted string properly
            inputs = session.hf_tokenizer(
                formatted_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True
            )

            # Move tensors to the correct device
            input_ids = inputs["input_ids"].to(session.hf_model.device)
            attention_mask = inputs["attention_mask"].to(session.hf_model.device)

            # Use torch.no_grad() for inference efficiency
            with torch.no_grad():
                outputs = session.hf_model.generate(
                    input_ids=input_ids,           # Pass as keyword argument
                    attention_mask=attention_mask, # Required for T5/mT5
                    **session.generation_options
                )

            # Decode: For Encoder-Decoder models like T5/mT5, the output IS the
            # complete generated sequence. No need to slice off the prompt.
            full_response = session.hf_tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )

            # Add assistant response to chat history
            session.chat_history.append({"role": "assistant", "content": full_response})

            return full_response

        except Exception as e:
            logger.error(f"Error during Hugging Face generation: {str(e)}")
            # It's good practice to include the original exception for debugging
            raise RuntimeError(f"Generation failed: {str(e)}") from e
    else:
        try:
            # Format the chat history using the model's built-in chat template
            prompt = session.hf_tokenizer.apply_chat_template(
                session.chat_history,
                tokenize=False,
                add_generation_prompt=True
            )

            # Tokenize - returns dict with 'input_ids', 'attention_mask', etc.
            inputs_ids = session.hf_tokenizer(prompt, return_tensors="pt")
            
            # Move to model's device
            inputs_ids = {k: v.to(session.hf_model.device) for k, v in inputs_ids.items()}
            
            # Get input length for slicing later
            input_length = inputs_ids["input_ids"].shape[1]

            # Generate
            with torch.no_grad():
                outputs = session.hf_model.generate(
                    **inputs_ids,
                    **session.generation_options
                )
                
            # Slice to get only new tokens - FIX: use input_length, not input_ids.shape[1]
            new_tokens = outputs[0][input_length:]
            full_response = session.hf_tokenizer.decode(new_tokens, skip_special_tokens=True)

            if first_message and not session.chat_history:
                # Add assistant response to chat history
                session.chat_history.append({"role": "assistant", "content": full_response})
            else:
                # Flush chat history after each message to save memory
                session.chat_history = []

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
    del session.hf_model
    del session.hf_tokenizer
    session.hf_model = None
    session.hf_tokenizer = None
    
    # Clear CUDA cache if GPU was used
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    logger.info("Hugging Face session closed successfully and VRAM freed.")
