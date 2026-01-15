#!/usr/bin/env python3
"""
...
"""

# --- Local Imports ---
from .gemini import (
    create_gemini_chat_session,
    send_gemini_message
)
from .huggingface import (
    create_huggingface_chat_session,
    send_huggingface_message
)
from .openai import (
    create_openai_chat_session,
    send_openai_message
)
from .ollama import (
    close_ollama_session,
    create_ollama_chat_session,
    send_ollama_message
)
from ..utils.logging import logger
from ..utils.file_utils import load_params_from_yaml
from ..utils.text_utils import get_document_with_entities
from ..ner.entity_postprocessing import (
    extract_entities_and_clean_text,
    generate_augmented_files,
    save_generated_text
)
from ..imports import (
    os,
    json,
    Path,
    Optional,
    GEMINI_MODELS,
    OLLAMA_MODELS,
    OPENAI_MODELS,
    HUGGINGFACE_MODELS
)


def run_generation_workflow(
    prompt_type,
    model,
    base_path: Optional[str] = None,
    original_path: Optional[str] = None,
    data_path: Optional[str] = None, 
    output_path: Optional[str] = None,
    prompts_path: Optional[str] = None,
    yaml_file_path: Optional[str] = None,
    ledger_path: Optional[str] = None,
    temporal_path: Optional[str] = None
):
    # Determine base path
    base = Path(base_path) if base_path else Path.home()

    # Convert to Path objects
    original_path = Path(base / "data" / "raw" / "train") if not original_path else Path(original_path)
    ledger_path = Path(base / "data" / "processed" / "mapping_ledger.json") if not data_path else Path(data_path) / "mapping_ledger.json"
    data_path = Path(base / "data" / "processed" / "extracted_sentences") if not data_path else Path(data_path)
    output_path = Path(base / "data" / "augmented") if not output_path else Path(output_path)
    prompts_path = Path(base / "config" / "prompts") if not prompts_path else Path(prompts_path)
    yaml_file_path = Path(base / "config" / "parameters") if not yaml_file_path else Path(yaml_file_path)
    
    # Load prompt
    prompt_file = os.path.join(prompts_path, f"{prompt_type}.txt")
    if not os.path.exists(prompt_file):
        raise FileNotFoundError(f"Prompt file {prompt_file} not found.")
    with open(prompt_file, 'r', encoding='utf-8') as f:
        base_prompt = f.read()

    # Sanitize model name for output path
    model_name = "".join(x for x in model if str(x).isalnum() or str(x) in ['-', '_'])

    # Create output path
    output_path = os.path.join(output_path, prompt_type, model_name)

    # Load parameters from YAML
    yaml_file = os.path.join(yaml_file_path, f"{model}.yml")
    model_params = load_params_from_yaml(yaml_file)

    # Load and create LLM session
    if model in OPENAI_MODELS:
        session = create_openai_chat_session(
        model_name=model_params["model_name"],
        generation_options=model_params["generation_options"],
        system_prompt=model_params["system_prompt"]
    )
    elif model in GEMINI_MODELS:
        session = create_gemini_chat_session(
        model_name=model_params["model_name"],
        generation_options=model_params["generation_options"],
        system_prompt=model_params["system_prompt"]
    )
    elif model in OLLAMA_MODELS:
        session = create_ollama_chat_session(
        model_name=model_params["model_name"],
        generation_options=model_params["generation_options"],
        system_prompt=model_params["system_prompt"]
    )
    elif model in HUGGINGFACE_MODELS:
        session = create_huggingface_chat_session(
        model_name=model_params["model_name"],
        generation_options=model_params["generation_options"],
        system_prompt=model_params["system_prompt"]
    )

    try:
        # Read types dictionary from json
        with open(ledger_path, 'r', encoding='utf-8') as f:
            ledger = dict(json.load(f))
    except Exception as e:
        logger.error(f"Error loading mapping ledger: {e}")

    # For each original document
    for doc_name, extracted_sentences in ledger.items():
        # Retrieve the original text
        txt_path = os.path.join(original_path, doc_name)
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read()
        logger.debug(f"Data Augmentation process for document {doc_name}...")
        
        # Initialize entities and displacement
        entities = []
        displacement = 0

        # For each extracted sentence
        for ledger_entry in extracted_sentences:
            # Extract sentence file name
            sentence_name = os.path.splitext(os.path.basename(ledger_entry["extracted_file"]))[0]

            # Get new prompt with text + entities from txt/ann
            if temporal_path:
                # Use a temp file for the auxiliary prompt
                aux_prompt_path = os.path.join(temporal_path, f"{sentence_name}_aux_prompt.txt")
                get_document_with_entities(data_path, sentence_name, output_file=aux_prompt_path)
                
                with open(aux_prompt_path, 'r', encoding='utf-8') as f:
                    text_prompt = f.read()
            else:
                # Get new prompt directly
                text_prompt = get_document_with_entities(data_path, sentence_name)

            # Combine with base prompt
            full_prompt = f"{base_prompt}\n\nInput Text to Process: {text_prompt}"

            # Send to model
            logger.debug(f"  Sending prompt for document {sentence_name}...")
            if model in OPENAI_MODELS:
                response = send_openai_message(session, full_prompt)
            elif model in GEMINI_MODELS:
                response = send_gemini_message(session, full_prompt)
            elif model in OLLAMA_MODELS:
                response = send_ollama_message(session, full_prompt)
            elif model in HUGGINGFACE_MODELS:
                response = send_huggingface_message(session, full_prompt)

            # Clean the response text and extract the new entities
            generated_text, generated_entities = extract_entities_and_clean_text(response)
            logger.debug(f"  The response obtained was: {generated_text}")
            logger.debug(f"  The cleaned text is: {generated_text}")
            logger.debug(f"  The entities obtained are: {generated_entities}")

            # Process response and save as txt/ann
            text, entities, displacement = generate_augmented_files(
                text, 
                entities, 
                generated_text, 
                generated_entities,
                displacement,
                ledger_entry
            )

            # Clean up temp file
            if temporal_path:
                os.remove(aux_prompt_path)

        # Save the generated text based on the original document
        doc_name = os.path.splitext(os.path.basename(doc_name))[0]
        save_generated_text(text, entities, doc_name, output_path)

    # Close the model session
    if model in OLLAMA_MODELS:
        close_ollama_session(session)
