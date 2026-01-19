#!/usr/bin/env python3
"""
Text Utilities for BRAT Annotations and phrase extraction.

This module provides:
- Functions to parse BRAT .ann files and extract entity annotations.
- Functions to find sentences containing entities.
- Functions to format entities back into BRAT .ann file format.
"""

# --- Local Imports ---
from .logging import logger
from ..imports import (
    os,
    Dict,
    List,
    Path,
    Optional,
    SENTENCE_SPLITTER
)


def adjust_entity_offsets(
    entity: Dict,
    sentence_start: int
) -> Dict:
    """
    Adjust entity offsets relative to the extracted sentence.

    Args:
        entity: Original entity dict with absolute offsets
        sentence_start: Start offset of the sentence in the original text

    Returns:
        New entity dict with offsets relative to the sentence start
    """
    return {
        'id': entity['id'],
        'type': entity['type'],
        'start': entity['start'] + sentence_start,
        'end': entity['end'] + sentence_start,
        'text': entity['text']
    }


def find_sentences_with_entities(
    text: str,
    entities: List[Dict]
) -> List[Dict]:
    """
    Find sentences that contain entities and return sentence information.
    
    Args:
        text: Full text from which sentences are extracted
        entities: List of entity dicts with 'start' and 'end' offsets
    
    Returns:
        List of dicts with:
        - start: sentence start offset in original text
        - end: sentence end offset in original text  
        - text: sentence text
        - entities: entities contained in this sentence
    """
    sentences = []

    # Split Spanish text into sentences using spaCy
    document = SENTENCE_SPLITTER(text)
    sentence_list = document.sents

    # Reconstruct sentences with their original offsets
    for sentence_text in sentence_list:
        if not sentence_text.text.strip():
            # Skip empty sentences
            continue

        # Calculate sentence offsets
        sentence_start = sentence_text.start_char
        sentence_end = sentence_text.end_char
        
        # Find entities that overlap with this sentence
        sentence_entities = []
        for entity in entities:
            # Check if entity overlaps with sentence (entity start < sentence end AND entity end > sentence start)
            if entity['start'] < sentence_end and entity['end'] > sentence_start:
                sentence_entities.append(entity)
        
        if sentence_entities:
            sentences.append({
                'start': sentence_start,
                'end': sentence_end,
                'text': sentence_text.text,
                'entities': sentence_entities
            })
    
    return sentences


def format_ann_content(
    entities: List[Dict]
) -> str:
    """
    Format entities into BRAT .ann file content.

    Args:
        entities: List of entity dicts

    Returns:
        String content formatted for a BRAT .ann file
    """
    lines = []
    for entity in entities:
        # Create BRAT annotation line
        line = f"{entity['id']}\t{entity['type']} {entity['start']} {entity['end']}\t{entity['text']}"
        lines.append(line)
    return '\n'.join(lines) + '\n' if lines else ''

    
def get_document_with_entities(
    base_path: str,
    filename: str,
    output_file: Optional[str] = None
) -> str:
    """
    Given a base filename (without extension), reads .txt and .ann files from `base_path`,
    and returns (or writes) a file/string with the text modified so that entities are 
    marked as [entity_text:entity_type].
    
    Args:
        base_path: Directory containing the .txt and .ann files
        filename: Base filename without extension
        output_file: Optional path to write the output to a file
        
    Returns:
        String with marked entities in the format [entity_text:entity_type]
    """
    # Convert to Path objects
    txt_path = Path(os.path.join(base_path, f"{filename}.txt"))
    ann_path = Path(os.path.join(base_path, f"{filename}.ann"))

    # Read the text from the txt file
    with open(txt_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Parse the ann file and return a list of entities
    entities = parse_ann_file(ann_path)
    
    # Sort entities by start position in descending order
    # This prevents index shifting when we replace text from the end backwards
    entities_sorted = sorted(entities, key=lambda x: x['start'], reverse=True)
    
    # Replace entities in reverse order to maintain correct indices
    marked_text = text
    for entity in entities_sorted:
        start = entity['start']
        end = entity['end']
        entity_text = entity['text']
        entity_type = entity['type']
        
        # Create the marked entity
        marked_entity = f"[{entity_text}:{entity_type}]"
        
        # Replace the original entity text with the marked version
        marked_text = marked_text[:start] + marked_entity + marked_text[end:]
    
    # Write to file if output_file is provided, otherwise return the string
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(marked_text)
    
    return marked_text


def parse_ann_file(
    ann_path: Path
) -> List[Dict]:
    """
    Parse BRAT .ann file and return list of entity annotations.
    
    Each entity dict contains:
    - id: entity ID (e.g., 'T1')
    - type: entity type (e.g., 'Person')
    - start: start character offset
    - end: end character offset  
    - text: actual text of the entity

    Args:
        ann_path: Path to the .ann file

    Returns:
        List of entity dicts associated with the .ann file
    """
    entities = []

    # Check if file exists
    if not ann_path.exists():
        return entities

    with open(ann_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            # Skip empty lines and comments
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Parse entity lines (T1 Person 128 138 John Doe) and ignore others (Attributes, Events and Relations)
            if line.startswith('T'):
                try:
                    # Split line into components
                    parts = line.split('\t')
                    if len(parts) < 3:
                        continue

                    # Extract entity details
                    entity_id = parts[0]
                    annotation_info = parts[1].split()
                    text = parts[2] if len(parts) > 2 else ""
                    
                    if len(annotation_info) >= 3:
                        # Handle multi-span annotations (e.g., T1 Person 128 138;140 145)
                        entity_type = annotation_info[0]
                        start = int(annotation_info[1])
                        end = int(annotation_info[2])

                        # Create entity dict for multi-span
                        entities.append({
                            'id': entity_id,
                            'type': entity_type,
                            'start': start,
                            'end': end,
                            'text': text
                        })
                except (ValueError, IndexError) as e:
                    logger.warning(f"Warning: Could not parse line {line_num} in {ann_path}: {line}")
                    continue

    # Sort entities by start offset
    return sorted(entities, key=lambda x: x['start'])
