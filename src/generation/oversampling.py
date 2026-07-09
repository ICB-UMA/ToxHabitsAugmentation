#!/usr/bin/env python3
"""
Entity-aware oversampling utilities.

This module creates an augmented dataset by duplicating original documents,
prioritizing files that contain the least frequent entity types.
"""

# --- Local Imports ---
from ..utils.logging import logger
from ..utils.text_utils import parse_ann_file
from ..imports import (
    json,
    shutil,
    Counter,
    Dict,
    List,
    Path,
    Optional,
)


def entity_type_counter_from_ann(
    ann_path: Path
) -> Counter:
    """
    Build a counter of entity types from a BRAT .ann file.
    
    Args:
        ann_path: Path to the .ann file.
    
    Returns:
        Counter object mapping entity types to their counts in the document.
    """
    entities, _ = parse_ann_file(ann_path)
    return Counter(entity["type"] for entity in entities)


def compute_selection_score(
    deficits: Dict[str, int],
    doc_counter: Counter
) -> int:
    """
    Compute how much a document helps reduce current entity deficits.

    The score is the capped gain per entity type to avoid over-crediting one
    heavily represented type in a candidate document.

    Args:
        deficits: Dictionary of entity types to their current deficit counts.
        doc_counter: Counter of entity types in the candidate document.
    
    Returns:
        Integer score representing the document's contribution to reducing deficits.
    """
    score = 0

    # For each entity type, add the minimum of the document's count and the remaining deficit.
    for entity_type, missing_count in deficits.items():
        if missing_count <= 0:
            continue
        # Add the minimum of the missing count and the document's count for this entity type.
        score += min(missing_count, doc_counter.get(entity_type, 0))
    return score


def safe_copy_pair(
    src_txt: Path,
    src_ann: Path,
    dst_txt: Path,
    dst_ann: Path
) -> None:
    """
    Copy .txt/.ann pair if both source files exist.
    
    Args:
        src_txt: Source .txt file path.
        src_ann: Source .ann file path.
        dst_txt: Destination .txt file path.
        dst_ann: Destination .ann file path.
    
    Raises:
        FileNotFoundError: If either source file does not exist.
    """
    # Ensure both source files exist before copying.
    if not src_txt.exists() or not src_ann.exists():
        raise FileNotFoundError(f"Missing pair for source document: {src_txt.stem}")

    # Ensure destination directory exists and copy files.
    dst_txt.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_txt, dst_txt)
    shutil.copy2(src_ann, dst_ann)


def run_entity_aware_oversampling(
    original_path: str,
    output_path: str,
    logger_file: Optional[str] = None,
    max_duplication_ratio: float = 1.0,
) -> Dict:
    """
    Build an oversampled corpus by duplicating documents with rare entities.

    Args:
        original_path: Path to the original BRAT corpus (.txt/.ann pairs).
        output_path: Destination directory where oversampled corpus is written.
        logger_file: Optional JSON log path.
        max_duplication_ratio: Maximum extra copies relative to original size.
            A value of 1.0 means at most `N` extra files for `N` originals.

    Returns:
        Dictionary with oversampling statistics.
    """
    # Validate input paths and prepare output directory.
    source_dir = Path(original_path)
    target_dir = Path(output_path)
    target_dir.mkdir(parents=True, exist_ok=True)

    # Gather all .txt files and ensure corresponding .ann files exist.
    txt_files = sorted(source_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {source_dir}")

    documents: List[Dict] = []
    global_entity_counter = Counter()

    # Process each .txt file and its corresponding .ann file to build the corpus and entity counts.
    for txt_file in txt_files:
        ann_file = txt_file.with_suffix(".ann")
        if not ann_file.exists():
            logger.warning(f"Skipping {txt_file.name}: missing .ann file")
            continue

        # Count entity types in the document and update global counts.
        doc_counter = entity_type_counter_from_ann(ann_file)
        global_entity_counter.update(doc_counter)
        documents.append({
            "stem": txt_file.stem,
            "txt": txt_file,
            "ann": ann_file,
            "entity_counter": doc_counter,
        })

    # Validate that we have documents to process.
    if not documents:
        raise RuntimeError("No valid .txt/.ann pairs found for oversampling")

    # Always include original corpus in output.
    for doc in documents:
        safe_copy_pair(
            doc["txt"],
            doc["ann"],
            target_dir / f"{doc['stem']}.txt",
            target_dir / f"{doc['stem']}.ann",
        )

    # If there are no entities, we cannot oversample, so we log and return.
    if not global_entity_counter:
        logger.info("No entities found in corpus. Oversampling will keep original copies only.")
        result = {
            "original_documents": len(documents),
            "added_documents": 0,
            "final_documents": len(documents),
            "initial_entity_distribution": {},
            "final_entity_distribution": {},
            "selected_documents": {},
        }
        if logger_file:
            with open(logger_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4)
        return result

    # Determine the target number of documents per entity type and the maximum number of additional documents allowed.
    initial_distribution = dict(global_entity_counter)
    target_per_entity = max(global_entity_counter.values())
    max_added_documents = int(round(len(documents) * max_duplication_ratio))

    selected_counts = Counter()
    current_distribution = Counter(global_entity_counter)

    # Helper function to compute deficits for each entity type.
    def _deficits(counter: Counter) -> Dict[str, int]:
        return {
            entity_type: max(0, target_per_entity - count)
            for entity_type, count in counter.items()
        }

    # Compute initial deficits and start the oversampling loop.
    deficits = _deficits(current_distribution)
    additions = 0

    # Loop until we reach the maximum number of added documents or there are no more deficits.
    while additions < max_added_documents and any(v > 0 for v in deficits.values()):
        best_doc = None
        best_score = 0

        for doc in documents:
            # Compute how much this document helps reduce the current deficits.
            score = compute_selection_score(deficits, doc["entity_counter"])
            if score > best_score:
                best_score = score
                best_doc = doc

        # If no document contributes to reducing deficits, we break the loop.
        if best_doc is None or best_score <= 0:
            break

        # Update counts and deficits based on the selected document.
        selected_counts[best_doc["stem"]] += 1
        current_distribution.update(best_doc["entity_counter"])
        additions += 1
        deficits = _deficits(current_distribution)

    # Materialize duplicated files with deterministic names.
    for stem, n_copies in selected_counts.items():
        original = next(doc for doc in documents if doc["stem"] == stem)
        for copy_idx in range(1, n_copies + 1):
            # Create a new stem for the duplicated document to avoid overwriting.
            duplicated_stem = f"{stem}_oversample_{copy_idx}"
            safe_copy_pair(
                original["txt"],
                original["ann"],
                target_dir / f"{duplicated_stem}.txt",
                target_dir / f"{duplicated_stem}.ann",
            )

    # Prepare the result dictionary with oversampling statistics.
    result = {
        "original_documents": len(documents),
        "added_documents": additions,
        "final_documents": len(documents) + additions,
        "target_per_entity": target_per_entity,
        "initial_entity_distribution": dict(initial_distribution),
        "final_entity_distribution": dict(current_distribution),
        "selected_documents": dict(selected_counts),
    }

    # Log the result to a JSON file if a logger_file path is provided.
    if logger_file:
        with open(logger_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4)

    logger.info(
        "Oversampling finished: %s originals + %s copies = %s documents",
        result["original_documents"],
        result["added_documents"],
        result["final_documents"],
    )
    return result
