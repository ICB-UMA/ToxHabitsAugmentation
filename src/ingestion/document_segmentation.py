#!/usr/bin/env python3
"""
Document Segmentation for ToxHabitsAugmentation

This module processes text documents and their BRAT annotations to extract 
sentences containing entities, adjusting entity offsets accordingly. It also
maintains a mapping ledger to track the relationship between original documents
and extracted sentences.
"""

# --- Local Imports ---
from ..utils.logging import logger
from ..utils.text_utils import (
    adjust_entity_offsets,
    find_sentences_with_entities,
    format_ann_content,
    parse_ann_file
)
from ..imports import (
    json,
    Path,
    Dict,
    Optional
)


def reconstruct_from_ledger(
    original_txt_path: str,
    ledger_path: str
) -> Dict[str, str]:
    """
    Reconstruct original text segments from extracted sentences using the ledger.
    Useful for verification.

    Args:
        original_txt_path: Path to the original .txt file
        ledger_path: Path to the mapping_ledger.json file

    Returns:
        Dictionary mapping extracted filenames to their original text segments
    """
    # Convert to Path objects
    original_txt_path = Path(original_txt_path)
    ledger_path = Path(ledger_path)
    
    if not ledger_path.exists():
        logger.error(f"Ledger not found at {ledger_path}")
        raise FileNotFoundError(f"Ledger not found: {ledger_path}")
    
    # Load ledger
    with open(ledger_path, 'r', encoding='utf-8') as f:
        ledger = json.load(f)
    
    reconstructed = {}
    
    for original_filename, extractions in ledger.items():
        # Only process the specified original file
        if original_filename != original_txt_path.name:
            continue
        
        # Read original text
        with open(original_txt_path, 'r', encoding='utf-8') as f:
            original_text = f.read()
        
        for extraction in extractions:
            # Get original text segment
            orig_start = extraction['original_start']
            orig_end = extraction['original_end']
            extracted_text = original_text[orig_start:orig_end]
            reconstructed[extraction['extracted_file']] = extracted_text
    
    return reconstructed


def segmentate_toxhabits_documents(
    base_path: Optional[str] = None,
    input_path: Optional[str] = None,
    output_path: Optional[str] = None
) -> None:
    """
    Process all .txt/.ann pairs in input_path.
    
    Args:
        base_path: base directory to store datasets; defaults to the user's home directory
        input_path: Path containing original .txt and .ann files
        output_path: Path to save extracted sentences and adjusted .ann files
    """
    # Determine base path
    base = Path(base_path) if base_path else Path.home()

    # Convert to Path objects
    input_path = Path(base / "data" / "raw" / "train") if not input_path else Path(input_path)
    ledger_path = Path(base / "data" / "processed" / "mapping_ledger.json") if not output_path else Path(output_path) / "mapping_ledger.json"
    output_path = Path(base / "data" / "processed" / "extracted_sentences") if not output_path else Path(output_path)
    
    # Create output directories
    output_path.mkdir(parents=True, exist_ok=True)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing ledger or create new one
    if ledger_path.exists():
        with open(ledger_path, 'r', encoding='utf-8') as f:
            ledger = json.load(f)
    else:
        ledger = {}
    
    # Find all .txt files and process corresponding .ann files
    txt_files = list(input_path.glob("*.txt"))
    
    logger.info(f"Found {len(txt_files)} text files to process")
    
    for txt_file in txt_files:
        ann_file = txt_file.with_suffix('.ann')
        
        logger.info(f"Processing: {txt_file.name}")
        
        # Read text file
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            logger.warning(f"Error reading {txt_file}: {e}")
            continue
        
        # Parse annotation file
        entities = parse_ann_file(ann_file)
        
        if not entities:
            logger.info(f"No entities found in {ann_file.name}, skipping")
            continue
        
        # Find sentences containing entities
        sentences_with_entities = find_sentences_with_entities(text, entities)
        
        if not sentences_with_entities:
            logger.info(f"No sentences with entities found for {txt_file.name}, skipping")
            continue
        
        # Process each sentence with entities
        for i, sentence_info in enumerate(sentences_with_entities, 1):
            sentence_start = sentence_info['start']
            sentence_end = sentence_info['end']
            sentence_text = sentence_info['text']
            sentence_entities = sentence_info['entities']
            
            # Create unique filename for extracted sentence
            # Format: original_filename_sentence{i}.txt
            base_name = txt_file.stem
            extracted_txt_name = f"{base_name}_sentence{i}.txt"
            extracted_ann_name = f"{base_name}_sentence{i}.ann"
            
            extracted_txt_path = output_path / extracted_txt_name
            extracted_ann_path = output_path / extracted_ann_name
            
            # Adjust entity offsets for the extracted sentence
            adjusted_entities = [
                adjust_entity_offsets(entity, -sentence_start) 
                for entity in sentence_entities
            ]
            
            # Save extracted sentence text
            try:
                with open(extracted_txt_path, 'w', encoding='utf-8') as f:
                    f.write(sentence_text)
                
                # Save adjusted annotation file
                ann_content = format_ann_content(adjusted_entities)
                with open(extracted_ann_path, 'w', encoding='utf-8') as f:
                    f.write(ann_content)
                
                logger.info(f"  Created: {extracted_txt_name} ({len(sentence_text)} chars, {len(adjusted_entities)} entities)")
                
            except Exception as e:
                logger.error(f"Error saving files for {extracted_txt_name}: {e}")
                continue
            
            # Update ledger
            if txt_file.name not in ledger:
                ledger[txt_file.name] = []
            
            # Append extraction info to ledger
            ledger[txt_file.name].append({
                'extracted_file': extracted_txt_name,
                'original_start': sentence_start,
                'original_end': sentence_end,
                'extracted_start': 0,
                'extracted_end': len(sentence_text),
                'entities_count': len(adjusted_entities),
                'entities': [
                    {
                        'id': entity['id'],
                        'type': entity['type'],
                        'original_start': entity['start'] + sentence_start,
                        'original_end': entity['end'] + sentence_start,
                        'extracted_start': entity['start'],
                        'extracted_end': entity['end']
                    }
                    for entity in adjusted_entities
                ]
            })
    
    # Save updated ledger
    try:
        with open(ledger_path, 'w', encoding='utf-8') as f:
            json.dump(ledger, f, indent=2, ensure_ascii=False)
        logger.info(f"\nMapping ledger saved to: {ledger_path}")
    except Exception as e:
        logger.error(f"Error saving mapping ledger: {e}")
