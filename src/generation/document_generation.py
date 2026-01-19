#!/usr/bin/env python3
"""
...
"""

# --- Local Imports ---
from .gemini import (
    close_gemini_session,
    create_gemini_chat_session,
    send_gemini_message,
    send_gemini_message_with_pdf
)
from .huggingface import (
    create_huggingface_chat_session,
    send_huggingface_message
)
from .openai import (
    close_openai_session,
    create_openai_chat_session,
    send_openai_message,
    send_openai_message_with_pdf
)
from .ollama import (
    close_ollama_session,
    create_ollama_chat_session,
    send_ollama_message
)
from ..utils.logging import logger
from ..utils.file_utils import (
    extract_filename_without_extension,
    load_params_from_yaml
)
from ..utils.text_utils import (
    get_document_with_entities
)
from ..ner.entity_postprocessing import (
    extract_entities_and_clean_text,
    generate_augmented_files,
    save_generated_text
)
from ..imports import (
    os,
    json,
    Dict,
    List,
    Tuple,
    Path,
    Any,
    Optional,
    GEMINI_MODELS,
    OLLAMA_MODELS,
    OPENAI_MODELS,
    HUGGINGFACE_MODELS,
    PROMPT_ANNOTATION_GUIDE,
    SENTENCE_BY_SENTENCE_METHODS,
    ALL_TOGETHER_METHODS
)

def build_sentence_prompt(
    data_path: str,
    sentence_name: str,
    temporal_path: Optional[str] = None
) -> str:
    """
    Builds a prompt from a sentence with its entities.
    Optionally uses a temporary file.
    
    Args:
        data_path: Path to data directory
        sentence_name: Name of sentence file
        temporal_path: Optional temporary directory path
        
    Returns:
        Formatted prompt text
    """
    if temporal_path:
        # Use a temp file for the auxiliary prompt
        aux_prompt_path = os.path.join(temporal_path, f"{sentence_name}_aux_prompt.txt")
        get_document_with_entities(data_path, sentence_name, output_file=aux_prompt_path)
        
        with open(aux_prompt_path, 'r', encoding='utf-8') as f:
            text_prompt = f.read()
        
        # Clean up temp file
        os.remove(aux_prompt_path)
    else:
        text_prompt = get_document_with_entities(data_path, sentence_name)
    
    return text_prompt


def build_full_prompt(
    base_prompt: str,
    text_prompt: str,
    label: str = "Input Text to Process"
) -> str:
    """Combines base prompt with text prompt."""
    return f"{base_prompt}\n\n{label}: {text_prompt}"


def process_model_response(
    response: str
) -> Tuple[str, List[Dict]]:
    """
    Processes and logs model response.
    
    Args:
        response: Raw response from model
        
    Returns:
        Tuple of (cleaned_text, entities)
    """
    # Clean the response text and extract the new entities
    generated_text, generated_entities = extract_entities_and_clean_text(response)
    
    logger.debug(f"  Raw response: {response}")
    logger.debug(f"  Cleaned text: {generated_text}")
    logger.debug(f"  Extracted entities: {generated_entities}")
    
    return generated_text, generated_entities


def process_sentence_by_sentence(
    session: Any,
    model: str,
    base_prompt: str,
    extracted_sentences: List[Dict],
    data_path: str,
    temporal_path: Optional[str],
    original_text: str,
    original_entities: List[Dict],
    prompt_type: str,
    pdf_path: Optional[str] = None
) -> Tuple[str, List[Dict]]:
    """
    Processes each sentence individually and replaces in original text.
    Used by method_1_var and method_3_guides_inj.
    
    Returns:
        Tuple of (modified_text, combined_entities)
    """
    text = original_text
    entities = original_entities
    displacement = 0
    
    for ledger_entry in extracted_sentences:
        # Extract sentence name
        sentence_name = extract_filename_without_extension(ledger_entry["extracted_file"])
        
        # Build prompt for this sentence
        text_prompt = build_sentence_prompt(data_path, sentence_name, temporal_path)
        full_prompt = build_full_prompt(base_prompt, text_prompt)
        
        # Send to model
        logger.debug(f"  Sending prompt for sentence: {sentence_name}")
        response = send_message_to_model(session, model, full_prompt, prompt_type, pdf_path)
        
        # Process response
        generated_text, generated_entities = process_model_response(response)
        
        # Update text with replacement
        text, entities, displacement = generate_augmented_files(
            text,
            entities,
            generated_text,
            generated_entities,
            displacement,
            ledger_entry
        )
    
    return text, entities


def process_all_sentences_together(
    session: Any,
    model: str,
    base_prompt: str,
    extracted_sentences: List[Dict],
    data_path: str,
    prompt_type: str,
    original_text: str = None,
    pdf_path: Optional[str] = None
) -> Tuple[str, List[Dict]]:
    """
    Processes all sentences together in a single prompt.
    Used by method_2_context and method_4_full_gen.
    
    Returns:
        Tuple of (generated_text, generated_entities)
    """
    # Build combined prompt
    if prompt_type == "method_4_full_gen":
        # Use original text as example
        full_prompt = build_full_prompt(base_prompt, original_text, label="Input Text of Example")
    else:  # method_2_context
        # Combine all sentences with separator
        sentence_prompts = [""]
        for ledger_entry in extracted_sentences:
            sentence_name = extract_filename_without_extension(ledger_entry["extracted_file"])
            sentence_prompt = get_document_with_entities(data_path, sentence_name)
            sentence_prompts.append(sentence_prompt)
        
        # Recombine the text prompt
        sentence_prompts.append("")
        text_prompt = "[CREATE TEXT] ".join(sentence_prompts)
        full_prompt = build_full_prompt(base_prompt, text_prompt)
    
    # Send to model
    logger.debug(f"  Sending combined prompt to model")
    response = send_message_to_model(session, model, full_prompt, prompt_type, pdf_path)
    
    # Process and return response
    return process_model_response(response)


def run_generation_workflow(
    prompt_type: str,
    model: str,
    base_path: Optional[str] = None,
    original_path: Optional[str] = None,
    data_path: Optional[str] = None, 
    output_path: Optional[str] = None,
    prompts_path: Optional[str] = None,
    yaml_file_path: Optional[str] = None,
    ledger_path: Optional[str] = None,
    pdf_path: Optional[str] = None,
    temporal_path: Optional[str] = None
):
    # Determine base path
    base = Path(base_path) if base_path else Path.home()

    # Determine PDF extension based on model
    extension = "txt" if model in OLLAMA_MODELS or model in HUGGINGFACE_MODELS else "pdf"

    # Convert to Path objects
    original_path = Path(base / "data" / "raw" / "train") if not original_path else Path(original_path)
    pdf_path = Path(base / "config" / "annotation_guide" / "toxhabits_annotationguide_v8." + extension) if not pdf_path else Path(pdf_path)
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
    
    # Send annotation guide if needed
    if ["method_3_guides_inj", "method_4_full_gen"]:
        send_message_to_model(
            session=session,
            model=model,
            prompt=PROMPT_ANNOTATION_GUIDE,
            prompt_type=prompt_type,
            pdf_path=pdf_path
        )

    try:
        # Read types dictionary from json
        with open(ledger_path, 'r', encoding='utf-8') as f:
            ledger = dict(json.load(f))
    except Exception as e:
        logger.error(f"Error loading mapping ledger: {e}")

    # Process each original document
    for doc_name, extracted_sentences in ledger.items():
        # Load original document
        txt_path = os.path.join(original_path, doc_name)
        with open(txt_path, 'r', encoding='utf-8') as f:
            original_text = f.read()
        
        logger.debug(f"Data Augmentation process for document {doc_name}...")

        # Route to appropriate processing method
        if prompt_type in SENTENCE_BY_SENTENCE_METHODS:
            # Process sentence by sentence with text replacement
            final_text, final_entities = process_sentence_by_sentence(
                session=session,
                model=model,
                base_prompt=base_prompt,
                extracted_sentences=extracted_sentences,
                data_path=data_path,
                temporal_path=temporal_path,
                original_text=original_text,
                original_entities=[],
                prompt_type=prompt_type,
                pdf_path=None
            )
        
        elif prompt_type in ALL_TOGETHER_METHODS:
            # Process all sentences together
            final_text, final_entities = process_all_sentences_together(
                session=session,
                model=model,
                base_prompt=base_prompt,
                extracted_sentences=extracted_sentences,
                data_path=data_path,
                prompt_type=prompt_type,
                original_text=original_text,
                pdf_path=pdf_path
            )
        
        else:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
        
        # Save generated document
        doc_basename = os.path.splitext(os.path.basename(doc_name))[0]
        save_generated_text(final_text, final_entities, doc_basename, output_path)

    # Close the model session
    if model in OPENAI_MODELS:
        close_openai_session(session)
    elif model in GEMINI_MODELS:
        close_gemini_session(session)
    elif model in OLLAMA_MODELS:
        close_ollama_session(session)


def send_message_to_model(
    session: Any,
    model: str,
    prompt: str,
    prompt_type: str,
    pdf_path: Optional[str] = None
) -> str:
    """
    Universal function to send messages to any supported model.
    Handles model-specific logic and PDF attachments.
    
    Args:
        session: Active chat session
        model: Model identifier string
        prompt: The prompt to send
        prompt_type: Type of prompt method being used
        pdf_path: Optional PDF path for methods that support it
        
    Returns:
        Model response text
    """
    # Determine if PDF should be used
    use_pdf = prompt_type in ["method_3_guides_inj", "method_4_full_gen"] and pdf_path
    
    # Route to appropriate model
    if model in OPENAI_MODELS:
        if use_pdf:
            return send_openai_message_with_pdf(session, prompt, pdf_path)
        return send_openai_message(session, prompt)
    elif model in GEMINI_MODELS:
        if use_pdf:
            return send_gemini_message_with_pdf(session, prompt, pdf_path)
        return send_gemini_message(session, prompt)
    elif model in OLLAMA_MODELS:
        return send_ollama_message(session, prompt)
    elif model in HUGGINGFACE_MODELS:
        return send_huggingface_message(session, prompt)
    else:
        raise ValueError(f"Unsupported model: {model}")
