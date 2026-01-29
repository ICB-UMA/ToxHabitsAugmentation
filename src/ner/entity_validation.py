#!/usr/bin/env python3
"""
Validation utilities for NER entities in the ToxHabitsAugmentation project.

This module provides functions to validate that entity texts match the
original text at their specified offsets, handling discontinuous entities
by concatenating spans with spaces.
"""

# --- Local Imports ---
from ..utils.logging import logger
from ..imports import (
    os,
    Dict,
    List,
    Optional,
    defaultdict
)


def print_validation_report(
    errors: Dict,
    output_path: Optional[str] = None,
    output_file: Optional[str] = "entity_validation_report.txt"
):
    """
    Generate and optionally save a validation error report.
    Just prints to console if no output_path is provided.
    
    Args:
        errors: A dictionary mapping document names to lists of error details.
        output_path: If provided, the path to save the report.
    """
    if not errors:
        # No errors found
        report = "✓ Validation passed! All entities match their text offsets correctly."
        logger.info(report)
        return
    
    # Generate report
    lines = []
    total_errors = sum(len(v) for v in errors.values())
    lines.append(f"✗ Validation Errors Found: {total_errors} across {len(errors)} documents")
    lines.append("=" * 60)
    
    for doc_name, doc_errors in errors.items():
        # Document-level errors
        lines.append(f"\nDocument: {doc_name}")
        for err in doc_errors:
            # Entity-level errors
            ent = err["entity"]
            lines.append(f"\n  Entity {ent['id']} ({ent['type']}) at line {ent['line']}")
            lines.append(f"    Spans: {ent['spans']}")
            
            # Detail the error
            if err["error_type"] == "text_mismatch":
                lines.append(f"    Expected: \"{err['expected']}\"")
                lines.append(f"    Found:    \"{err['found']}\"")
            else:
                lines.append(f"    Error: {err['message']}")
    
    # Combine lines into final report
    report = "\n".join(lines)
    logger.info(report)
    
    if output_path:
        # Save report to file
        output_file = os.path.join(output_path, output_file)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"Report of validation saved to {output_file}")


def validate_entities(
    corpus: Dict
) -> Dict[str, List[Dict]]:
    """
    Validates that entity texts match the original text at specified offsets.
    Handles discontinuous entities by concatenating spans with spaces.
    
    Args:
        corpus: A dictionary where keys are document names and values are
                       dictionaries with "text" and "entities". Each entity is a
                       dictionary with "text" and "spans" (list of [start, end]).
    Returns:
        errors: A dictionary mapping document names to lists of
                error details for mismatched entities.
    """
    # Initialize errors dictionary
    errors = defaultdict(list)
    
    for doc_name, data in corpus.items():
        # Get the full text of the document
        text = data["text"]
        
        for entity in data["entities"]:
            # Extract text from all spans (for discontinuous entities)
            extracted_parts = []
            valid_spans = True
            start = entity["start"]
            end = entity["end"]
            
            # Check span validity
            if start < 0 or end > len(text) or start > end:
                # Log invalid span error
                errors[doc_name].append({
                    "entity": entity,
                    "error_type": "invalid_span",
                    "expected": entity["text"],
                    "found": None,
                    "message": f"Invalid span [{start}, {end}] for text length {len(text)}"
                })
                valid_spans = False
                break
            
            # Extract text for the current span
            extracted_parts.append(text[start:end])
            
            if not valid_spans:
                # Skip further validation for this entity due to invalid spans
                continue
            
            # Reconstruct text from spans
            # Note: BRAT typically uses space to join discontinuous parts
            extracted_text = " ".join(extracted_parts) if len(extracted_parts) > 1 else extracted_parts[0]
            
            # Validate match (exact match required, but you can modify for whitespace tolerance)
            if extracted_text != entity["text"]:
                # Log text mismatch error
                errors[doc_name].append({
                    "entity": entity,
                    "error_type": "text_mismatch",
                    "expected": entity["text"],
                    "found": extracted_text,
                    "spans": [(start, end)],
                    "message": f"Text mismatch at {[(start, end)]}"
                })
    
    return dict(errors)
