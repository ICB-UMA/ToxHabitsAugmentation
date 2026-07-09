#!/usr/bin/env python3
"""
Entity-aware synonym replacement using a masked language model.

This module creates one augmented document per source document by replacing
entity mentions with context-compatible alternatives proposed by a Spanish
 clinical masked language model.
"""

# --- Third-Party Imports ---
from transformers import pipeline, set_seed

# --- Local Imports ---
from ..utils.logging import logger
from ..utils.text_utils import parse_ann_file
from ..utils.dictionary_similarity import (
    build_label_dictionary_similarity,
    infer_most_likely_label,
)
from ..ner.entity_postprocessing import save_generated_text
from ..imports import (
    json,
    re,
    torch,
    Dict,
    List,
    Tuple,
    Optional,
    Path,
    RANDOM_SEED,
)

# --- Constants ---

DEFAULT_MLM_MODEL = "PlanTL-GOB-ES/bsc-bio-ehr-es"
STANDARD_MODE = "standard"
DICTIONARY_RELABEL_MODE = "dictionary_relabel"

# --- Helper Functions ---

def get_sentence_span(
    text: str,
    start: int,
    end: int
) -> Tuple[int, int]:
    """
    Return sentence-like boundaries around an entity span.
    
    Args:
        text: The full text of the document.
        start: Start index of the entity mention.
        end: End index of the entity mention.
        
    Returns:
        A tuple (sentence_start, sentence_end) representing the boundaries of the sentence containing the entity.
    """
    # Find the nearest sentence-ending punctuation before the entity span.
    left = text.rfind(".", 0, start)
    left_q = text.rfind("?", 0, start)
    left_e = text.rfind("!", 0, start)
    left_nl = text.rfind("\n", 0, start)
    sentence_start = max(left, left_q, left_e, left_nl) + 1

    # Find the nearest sentence-ending punctuation after the entity span.
    right_candidates = [
        pos for pos in (
            text.find(".", end),
            text.find("?", end),
            text.find("!", end),
            text.find("\n", end),
        ) if pos != -1
    ]
    sentence_end = min(right_candidates) if right_candidates else len(text)
    return sentence_start, sentence_end


def mask_entity_in_sentence(
    sentence: str,
    local_start: int,
    local_end: int
) -> str:
    """
    Mask one entity mention in its sentence context.
    
    Args:
        sentence: The sentence containing the entity.
        local_start: Start index of the entity within the sentence.
        local_end: End index of the entity within the sentence.
        
    Returns:
        The sentence with the entity mention replaced by a <mask> token.
    """
    return f"{sentence[:local_start]}<mask>{sentence[local_end:]}"


def is_valid_candidate(
    token: str,
    original_text: str
) -> bool:
    """
    Apply conservative filters to MLM candidates.
    
    Args:
        token: The candidate token string from the MLM.
        original_text: The original entity mention text.
    
    Returns:
        True if the candidate is valid for replacement; False otherwise.
    """
    candidate = token.strip()
    if not candidate:
        return False
    if candidate.lower() == original_text.strip().lower():
        return False
    if candidate.startswith("#"):
        return False
    if any(char.isdigit() for char in candidate):
        return False
    if len(candidate) <= 1:
        return False
    if not re.match(r"^[A-Za-zÁÉÍÓÚÜÑáéíóúüñ-]+$", candidate):
        return False
    return True


def choose_replacement(
    fill_mask,
    sentence: str,
    local_start: int,
    local_end: int,
    original_entity_text: str,
    min_score: float = 0.01,
) -> Optional[str]:
    """
    Choose the first valid replacement from top MLM predictions.
    
    Args:
        fill_mask: The Hugging Face fill-mask pipeline instance.
        sentence: The sentence containing the entity.
        local_start: Start index of the entity within the sentence.
        local_end: End index of the entity within the sentence.
        original_entity_text: The original entity mention text.
        min_score: Minimum confidence score for candidate acceptance.
        
    Returns:
        The first valid replacement token, or None if no valid candidate is found.
    """
    # Mask the entity mention in the sentence for MLM prediction.
    masked_sentence = mask_entity_in_sentence(sentence, local_start, local_end)

    try:
        # Get top predictions from the masked language model.
        predictions = fill_mask(masked_sentence)
    except Exception as exc:
        logger.warning("MLM prediction failed for sentence fragment: %s", exc)
        return None

    # Filter and return the first valid candidate above the minimum score threshold.
    if not isinstance(predictions, list):
        predictions = [predictions]

    for candidate in predictions:
        # Extract the score and token string from the prediction.
        score = float(candidate.get("score", 0.0))
        token = str(candidate.get("token_str", "")).strip()
        if score < min_score:
            continue
        if is_valid_candidate(token, original_entity_text):
            return token

    return None


def update_entity_offsets(
    entities: List[Dict],
    start_idx: int,
    delta: int
) -> List[Dict]:
    """
    Shift offsets for entities after a replaced span.
    
    Args:
        entities: List of entity dictionaries with 'start' and 'end' keys.
        start_idx: The index in the text where the replacement occurred.
        delta: The change in length due to the replacement (new_length - old_length).
    
    Returns:
        A new list of entities with updated offsets.
    """
    updated = []
    for entity in entities:
        entity_copy = entity.copy()
        # Only update entities that occur after the replaced span.
        if entity_copy["start"] > start_idx:
            entity_copy["start"] += delta
            entity_copy["end"] += delta
        updated.append(entity_copy)
    return updated


def run_mlm_synonym_replacement(
    original_path: str,
    output_path: str,
    logger_file: Optional[str] = None,
    model_name: str = DEFAULT_MLM_MODEL,
    top_k: int = 20,
    min_score: float = 0.01,
    replacement_mode: str = STANDARD_MODE,
    dictionary_dataset_path: Optional[str] = None,
    dictionary_similarity_threshold: float = 0.72,
) -> Dict:
    """
    Generate augmented documents by replacing entity mentions with MLM synonyms.

    Args:
        original_path: Path to original corpus (.txt/.ann pairs).
        output_path: Destination directory for augmented files.
        logger_file: Optional JSON log file path.
        model_name: Hugging Face masked language model identifier.
        top_k: Number of candidates evaluated from fill-mask output.
        min_score: Minimum candidate confidence accepted.
        replacement_mode: "standard" (MLM only) or "dictionary_relabel".
        dictionary_dataset_path: Dataset path used to build dictionary similarity; defaults to original_path.
        dictionary_similarity_threshold: Similarity threshold for cross-label dictionary candidates.

    Returns:
        Generation summary.
    """
    # Validate input paths and prepare output directory.
    set_seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    source_dir = Path(original_path)
    target_dir = Path(output_path)
    target_dir.mkdir(parents=True, exist_ok=True)

    if replacement_mode not in [STANDARD_MODE, DICTIONARY_RELABEL_MODE]:
        raise ValueError(
            f"Unknown replacement_mode '{replacement_mode}'. "
            f"Expected '{STANDARD_MODE}' or '{DICTIONARY_RELABEL_MODE}'."
        )

    # Gather all .txt files and ensure corresponding .ann files exist.
    txt_files = sorted(source_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {source_dir}")

    device = 0 if torch.cuda.is_available() else -1

    fill_mask = None
    dictionary_similarity = None

    # Initialize the Hugging Face fill-mask pipeline with the specified model.
    fill_mask = pipeline(
        "fill-mask",
        model=model_name,
        tokenizer=model_name,
        top_k=top_k,
        device=device,
    )

    if replacement_mode == DICTIONARY_RELABEL_MODE:
        # Build dictionary similarity mapping for cross-label replacements.
        dictionary_similarity = build_label_dictionary_similarity(
            dataset_path=dictionary_dataset_path or original_path,
            min_similarity=dictionary_similarity_threshold,
        )

    logger_json = {
        "model": model_name,
        "replacement_mode": replacement_mode,
        "generated_documents": [],
        "failed_documents": [],
    }

    if dictionary_similarity is not None:
        logger_json["dictionary_similarity_summary"] = {
            "labels": sorted(dictionary_similarity.get("label_to_terms", {}).keys()),
            "similarity_threshold": dictionary_similarity_threshold,
        }

    for txt_path in txt_files:
        # Ensure the corresponding .ann file exists for the .txt file.
        ann_path = txt_path.with_suffix(".ann")
        if not ann_path.exists():
            logger.warning("Skipping %s: missing .ann pair", txt_path.name)
            continue

        # Read the text and parse entities from the .ann file.
        doc_name = txt_path.stem
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()
        entities, _ = parse_ann_file(ann_path)

        if not entities:
            # If no entities are found, save the original text and log the event.
            save_generated_text(text, [], doc_name, str(target_dir))
            logger_json["generated_documents"].append({
                "sentence": doc_name,
                "status": "success_no_entities",
                "replacements": 0,
            })
            continue

        # Initialize the updated text and entity list for processing.
        updated_text = text
        updated_entities = [entity.copy() for entity in entities]
        replacements = 0

        # Replace entities from left to right while maintaining offsets.
        for idx in range(len(updated_entities)):
            entity = updated_entities[idx]
            entity_text = entity["text"]

            # Multi-token entities are skipped to keep one-mask substitution stable.
            if len(entity_text.split()) != 1:
                continue

            # Determine the sentence span containing the entity and extract it.
            start = entity["start"]
            end = entity["end"]
            sent_start, sent_end = get_sentence_span(updated_text, start, end)
            sentence = updated_text[sent_start:sent_end]

            # Calculate local indices of the entity within the sentence and attempt to find a replacement.
            local_start = start - sent_start
            local_end = end - sent_start

            # Choose a replacement using the MLM or dictionary-based method.
            replacement = choose_replacement(
                fill_mask=fill_mask,
                sentence=sentence,
                local_start=local_start,
                local_end=local_end,
                original_entity_text=entity_text,
                min_score=min_score,
            )

            if not replacement:
                # Log a warning if no valid replacement is found and skip to the next entity.
                logger.warning(
                    "No valid replacement found for entity '%s' in document '%s'",
                    entity_text, doc_name
                )
                continue

            if replacement_mode == STANDARD_MODE:
                # In standard mode, keep the original label.
                new_label = entity["type"]
            else:
                # In dictionary relabel mode, infer label from the MLM-generated replacement term.
                new_label = infer_most_likely_label(
                    term=replacement,
                    dictionary_similarity=dictionary_similarity,
                    min_similarity=dictionary_similarity_threshold,
                )

                if not new_label:
                    # If no label is inferred, skip this replacement to avoid mis-labeling.
                    logger.warning(
                        "No valid label inferred for replacement '%s' in document '%s'",
                        replacement, doc_name
                    )
                    continue

            # Update the text with the replacement and adjust entity offsets accordingly.
            updated_text = f"{updated_text[:start]}{replacement}{updated_text[end:]}"

            # Calculate the change in length due to the replacement and update the entity's text and end offset.
            delta = len(replacement) - (end - start)
            updated_entities[idx]["text"] = replacement
            updated_entities[idx]["end"] = start + len(replacement)
            updated_entities[idx]["type"] = new_label if new_label else updated_entities[idx]["type"]

            if delta != 0:
                # Shift the offsets of all subsequent entities to account for the change in text length.
                trailing = updated_entities[idx + 1:]
                shifted = update_entity_offsets(trailing, start, delta)
                updated_entities = updated_entities[:idx + 1] + shifted

            replacements += 1

        # Save the updated text and entities to the output directory, and log the results.
        if not updated_entities:
            logger_json["failed_documents"].append({
                "sentence": doc_name,
                "status": "failed_no_entities",
            })
            continue

        # Save the augmented document and its updated entity annotations.
        save_generated_text(updated_text, updated_entities, doc_name, str(target_dir))
        logger_json["generated_documents"].append({
            "sentence": doc_name,
            "status": "success",
            "replacements": replacements,
        })

    if logger_file:
        with open(logger_file, "w", encoding="utf-8") as f:
            json.dump(logger_json, f, indent=4)

    return logger_json
