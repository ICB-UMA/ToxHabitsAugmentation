#!/usr/bin/env python3
"""
Text Utilities for BRAT Annotations and phrase extraction.

This module provides:
- Functions to parse BRAT .ann files.
- Functions to extract sentences containing entities.
- Functions to adjust entity offsets relative to sentences.
"""

# --- Third-Party Imports ---
from urllib3.util import Retry
from requests.adapters import HTTPAdapter

# --- Local Imports ---
from .logging import logger
from ..imports import (
    Dict,
    List,
    Path,
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
        'start': entity['start'] - sentence_start,
        'end': entity['end'] - sentence_start,
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
