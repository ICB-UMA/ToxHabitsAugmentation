#!/usr/bin/env python3
"""
...
"""

# --- Local Imports ---
from ..utils.logging import logger
from ..utils.text_utils import (
    adjust_entity_offsets,
    format_ann_content
)
from ..imports import (
    re,
    Path,
    Dict,
    List,
    Tuple,
    Any
)


def extract_entities_and_clean_text(
    raw_text: str
) -> Tuple[str, List[Dict]]:
    """
    Extracts all ``[entity_text:entity_type]`` markers from the raw text.

    Returns
        The original text with the markers stripped, but the captured
        entity_text kept in place.
        One entity instance per marker, containing its id, type,
        text and character offsets in the *original* raw_text.
    """
    # Regex pattern that matches [entity_text:entity_type]
    pattern = r'\[([^:]+):([^]]+)\]'

    # Compile with the POSIX flag so that '.' does NOT match new‑lines
    regex = re.compile(pattern)

    # Initialize the entities and cleaned text lists
    entities = []
    cleaned_parts = []          
    last_index = 0
    displacement = 0

    for match in regex.finditer(raw_text):
        # Append the part *before* the current marker
        cleaned_parts.append(raw_text[last_index:match.start()])

        # Extract captured groups
        entity_text = match.group(1)
        entity_type = match.group(2)

        # Record the entity with offsets of the *whole* marker
        marker_start = match.start() - displacement
        marker_end   = marker_start + len(entity_text)

        entities.append({
            "id": f"T{len(entities)+1}",
            "type": entity_type,
            "text": entity_text,
            "start": marker_start,
            "end": marker_end
        })

        # Append the entity_text itself to the cleaned output
        cleaned_parts.append(entity_text)

        # Move the cursor past this marker so the next loop iteration starts after it
        last_index = match.end()

        # Calculate the new offset when removing the entity type and the colon characters [:]
        displacement = displacement + 3 + len(entity_type)

    # Append the final tail (everything after the last marker)
    cleaned_parts.append(raw_text[last_index:])

    # Join everything into the final clean string
    clean_text = ''.join(cleaned_parts)

    return clean_text, entities


def generate_augmented_files(
    original_text: str,
    original_entities: list,
    generated_text: str,
    generated_entities: list,
    displacement: int,
    ledger: dict
) -> tuple[str, list, int]:
    """
    Replaces a segment of original text with generated text, managing entity positions
    and tracking displacement for future replacements.
    
    Args:
        original_text: The full original text to be modified
        original_entities: List of existing entities in the original text
        generated_text: New text that will replace a segment of the original
        generated_entities: Entities found in the generated text
        displacement: Accumulated displacement from previous replacements
        ledger: Dictionary containing replacement position information
        
    Returns:
        tuple containing:
        - modified_text: The original text with the replacement applied
        - combined_entities: All entities with updated positions
        - new_displacement: Updated displacement value for future replacements
    """
    
    # Extract replacement positions from ledger
    original_start = ledger["original_start"]
    original_end = ledger["original_end"]
    
    # Apply displacement to get actual current positions in the modified text
    adjusted_start = original_start + displacement
    adjusted_end = original_end + displacement
    
    # Replace the text segment
    modified_text = (
        original_text[:adjusted_start] + 
        generated_text + 
        original_text[adjusted_end:]
    )
    
    # Calculate the length difference for displacement update
    original_length = original_end - original_start
    generated_length = len(generated_text)
    length_difference = generated_length - original_length
    new_displacement = displacement + length_difference
    
    # Adjust entity offsets for the extracted sentence
    adjusted_generated_entities = [
        adjust_entity_offsets(entity, adjusted_start) 
        for entity in generated_entities
    ]
    
    # Adjust original entities that come after the replacement point
    adjusted_original_entities = []
    for entity in original_entities:
        adjusted_entity = entity.copy()
        
        # Entities before the replacement remain unchanged
        if entity["end"] <= original_start:
            adjusted_original_entities.append(adjusted_entity)
        
        # Entities after the replacement need position adjustment
        elif entity["start"] >= original_end:
            adjusted_entity["start"] = entity["start"] + new_displacement
            adjusted_entity["end"] = entity["end"] + new_displacement
            adjusted_original_entities.append(adjusted_entity)
        else:
            # Skip overlapping entities
            logger.info(f"Warning: Entity {entity['id']} overlaps with replacement region and will be removed")
            continue
    
    # Combine entities and reassign IDs to avoid duplicates
    combined_entities = []
    entity_counter = 1
    
    # First add all adjusted original entities
    for entity in adjusted_original_entities:
        entity_copy = entity.copy()
        entity_copy["id"] = f"T{entity_counter}"
        combined_entities.append(entity_copy)
        entity_counter += 1
    
    # Then add all adjusted generated entities
    for entity in adjusted_generated_entities:
        entity_copy = entity.copy()
        entity_copy["id"] = f"T{entity_counter}"
        combined_entities.append(entity_copy)
        entity_counter += 1
    
    # Sort entities by start position for better readability
    combined_entities.sort(key=lambda x: x["start"])
    
    # Validation: Verify all entities are within text bounds
    text_length = len(modified_text)
    for entity in combined_entities:
        if entity["end"] > text_length:
            logger.warning(f"Warning: Entity {entity['id']} extends beyond text length")
            
        # Verify the entity text matches the actual text at that position
        extracted_text = modified_text[entity["start"]:entity["end"]]
        if extracted_text != entity["text"]:
            logger.warning(f"Warning: Entity {entity['id']} text mismatch. Expected '{entity['text']}', got '{extracted_text}'")
    
    return modified_text, combined_entities, new_displacement


def save_generated_text(
    generated_text: str,
    generated_entities: List[Dict[str, Any]],
    filename: str,
    output_path: str
) -> None:
    """
    Save a generated text and its corresponding annotation to disk.

    Args:
        generated_text: The text content to save to a .txt file
        generated_entities: A list of annotation dictionaries to convert to a .ann content
            using the existing format_ann_content function
            filename: Base name (without extension) for the output files
        output_path: Directory path where the files will be written

    Returns:
        txt file containing the generated_text and ann file containing the annotation content 
        produced by format_ann_content
    """
    # Ensure output directory exists
    path = Path(output_path)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Unable to ensure output directory '{path}': {e}")
        raise

    # Create the paths to the new files
    generated_txt_path = path / f"{filename}.txt"
    generated_ann_path = path / f"{filename}.ann"

    try:
        # Save generated sentence text
        with generated_txt_path.open("w", encoding="utf-8") as f_txt:
            f_txt.write(generated_text)

        # Save adjusted annotation file
        ann_content = format_ann_content(generated_entities)
        with generated_ann_path.open("w", encoding="utf-8") as f_ann:
            f_ann.write(ann_content)

        logger.info(f"Created: {generated_ann_path} ({len(generated_text)} chars, {len(generated_entities)} entities)")
    except Exception as e:
        logger.error(f"Error saving files for '{filename}': {e}")
        raise
