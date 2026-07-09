#!/usr/bin/env python3
"""
Dictionary and similarity utilities for entity-aware replacement.

This module parses BRAT corpora (.txt/.ann) to build label dictionaries and
cross-label similarity candidates that can be used for relabeling replacements.
"""

# --- Standard Library Imports ---
from difflib import SequenceMatcher

# --- Local Imports ---
from .logging import logger
from .text_utils import parse_ann_file
from ..imports import (
    re,
    Counter,
    Dict,
    List,
    Optional,
    Path,
)


def normalize_term(
    term: str
) -> str:
    """
    Build a lightweight normalized representation for dictionary matching.

    Args:
        term: The input term to normalize.
    
    Returns:
        A normalized string representation of the term, suitable for similarity comparisons.
    """
    # Basic normalization: lowercase, strip accents, remove punctuation, and collapse whitespace.
    text = term.strip().lower()
    text = text.replace("á", "a").replace("é", "e").replace("í", "i")
    text = text.replace("ó", "o").replace("ú", "u").replace("ü", "u")
    text = re.sub(r"[^a-zñ\s-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Handle frequent gender/plural morphology (fumador/fumadora/fumadores...).
    for suffix in ["adoras", "adores", "adora", "ador", "doras", "dores", "dora", "dor"]:
        if text.endswith(suffix) and len(text) > len(suffix) + 2:
            return text[:-len(suffix)] + "dor"

    # Handle common verbal inflections (bebe, bebia, bebera, bebiendo...).
    for suffix in [
        "ariamos", "eriamos", "iriamos", "abamos", "iamos", "aremos", "eremos", "iremos",
        "arian", "erian", "irian", "aban", "ian", "aran", "eran", "iran",
        "ando", "iendo", "ado", "ada", "ados", "adas", "ido", "ida", "idos", "idas",
        "are", "ere", "ire", "aba", "ia", "ara", "era", "ira", "an", "en", "es", "as", "a", "e", "o",
    ]:
        if text.endswith(suffix) and len(text) > len(suffix) + 2:
            return text[:-len(suffix)]

    return text


def are_regex_family_related(
    term_a: str,
    term_b: str
) -> bool:
    """
    Check if two terms look like morphological variants using regex-like heuristics.
    
    Args:
        term_a: First term to compare.
        term_b: Second term to compare.
    
    Returns:
        True if the terms are likely related by common morphological patterns, False otherwise.
    """
    # Normalize both terms for comparison.
    a = normalize_term(term_a)
    b = normalize_term(term_b)
    if not a or not b:
        return False

    # Check for exact match after normalization.
    if a == b:
        return True

    # Gender/plural families from base root.
    root_match = re.match(r"^(.{3,})dor$", a)
    if root_match:
        root = root_match.group(1)
        if re.match(rf"^{re.escape(root)}do(r|ra|res|ras)?$", b):
            return True

    # Verb families from stem + frequent endings.
    for source, target in [(a, b), (b, a)]:
        if len(source) >= 4:
            stem = source[:4]
            if target.startswith(stem):
                return True

    return False


def compute_term_similarity(
    term_a: str,
    term_b: str
) -> float:
    """
    Compute similarity score combining normalization, regex families and sequence ratio.
    
    Args:
        term_a: First term to compare.
        term_b: Second term to compare.

    Returns:
        A float representing the similarity score between the two terms.
    """
    # Normalize both terms for comparison.
    norm_a = normalize_term(term_a)
    norm_b = normalize_term(term_b)
    if not norm_a or not norm_b:
        return 0.0
    if norm_a == norm_b:
        return 1.0

    # Compute the sequence similarity ratio using difflib's SequenceMatcher.
    ratio = SequenceMatcher(None, norm_a, norm_b).ratio()
    if are_regex_family_related(term_a, term_b):
        ratio = max(ratio, 0.85)
    return ratio


def build_label_dictionary_similarity(
    dataset_path: str,
    min_similarity: float = 0.72
) -> Dict:
    """
    Parse a BRAT corpus (.txt/.ann) and build dictionaries + cross-label similarity.

    Args:
        dataset_path: Path to the dataset directory containing .txt and .ann files.
        min_similarity: Minimum similarity threshold for considering terms as related.

    Returns:
        Dictionary with:
        - label_to_terms: terms observed per label with frequencies.
        - cross_label_similar_terms: candidate replacements that can change labels.
        - term_label_frequencies: per-term label counts for noisy-label disambiguation.
        - term_total_frequencies: per-term total counts.
    """
    # Initialize the base directory and list of annotation files.
    base_dir = Path(dataset_path)
    ann_files = sorted(base_dir.glob("*.ann"))
    if not ann_files:
        raise FileNotFoundError(f"No .ann files found in {base_dir}")

    # Build counters from annotation files.
    label_term_counter: Dict[str, Counter] = {}
    term_label_counter: Dict[str, Counter] = {}
    term_total_counter = Counter()

    for ann_file in ann_files:
        txt_file = ann_file.with_suffix(".txt")
        if not txt_file.exists():
            logger.warning("Skipping %s because %s is missing", ann_file.name, txt_file.name)
            continue

        # Parse the .ann file to extract entities and their labels, updating the label-term frequency counter.
        entities, _ = parse_ann_file(ann_file)
        for entity in entities:
            label = entity["type"]
            term = entity["text"].strip()
            if not term:
                continue

            term_key = term.lower()
            if label not in label_term_counter:
                label_term_counter[label] = Counter()
            label_term_counter[label][term] += 1

            if term_key not in term_label_counter:
                term_label_counter[term_key] = Counter()
            term_label_counter[term_key][label] += 1
            term_total_counter[term_key] += 1

    # If no terms were found, return empty structures.
    if not label_term_counter:
        return {
            "label_to_terms": {},
            "cross_label_similar_terms": {},
            "term_label_frequencies": {},
            "term_total_frequencies": {},
        }

    # Build cross-label similarity candidates for each term in each label.
    cross_label_similar_terms: Dict[str, Dict[str, List[Dict]]] = {}
    labels = sorted(label_term_counter.keys())

    for source_label in labels:
        # Initialize the dictionary for the source label.
        cross_label_similar_terms[source_label] = {}
        for source_term, source_freq in label_term_counter[source_label].items():
            candidates = []
            for target_label in labels:
                # Skip self-comparison; we only want cross-label candidates.
                if target_label == source_label:
                    continue
                for target_term, target_freq in label_term_counter[target_label].items():
                    # Compute similarity score between the source term and target term.
                    score = compute_term_similarity(source_term, target_term)
                    if score >= min_similarity:
                        candidates.append({
                            "term": target_term,
                            "label": target_label,
                            "score": round(score, 4),
                            "frequency": int(target_freq),
                            "source_frequency": int(source_freq),
                        })

            # Sort candidates by score, frequency, term, and label for consistent selection.
            candidates.sort(key=lambda x: (-x["score"], -x["frequency"], x["term"], x["label"]))
            if candidates:
                # Store the sorted candidates for the source term under the source label.
                cross_label_similar_terms[source_label][source_term.lower()] = candidates

    # Build the label-to-terms dictionary with frequencies for output.
    label_to_terms = {
        label: [{"term": term, "frequency": int(freq)} for term, freq in counter.most_common()]
        for label, counter in label_term_counter.items()
    }

    return {
        "label_to_terms": label_to_terms,
        "cross_label_similar_terms": cross_label_similar_terms,
        "term_label_frequencies": {
            term: {label: int(freq) for label, freq in label_counter.items()}
            for term, label_counter in term_label_counter.items()
        },
        "term_total_frequencies": {term: int(freq) for term, freq in term_total_counter.items()},
    }


def infer_most_likely_label(
    term: str,
    dictionary_similarity: Dict,
    min_similarity: float = 0.72,
) -> Optional[str]:
    """
    Infer label for a term using majority frequency and similarity fallback.

    Strategy:
    1) If term exists exactly in dictionary, pick the most frequent label.
    2) Otherwise, find most similar dictionary terms and pick the one with
       highest total frequency when scores tie, then pick that term's majority label.
    """
    # Normalize the input term for consistent comparison.
    term_key = term.strip().lower()
    if not term_key:
        return None

    # Retrieve the term-label frequency and total frequency dictionaries from the provided dictionary_similarity.
    term_label_frequencies = dictionary_similarity.get("term_label_frequencies", {})
    term_total_frequencies = dictionary_similarity.get("term_total_frequencies", {})

    # Exact match: choose majority label for this term.
    if term_key in term_label_frequencies:
        label_counter = term_label_frequencies[term_key]
        max_freq = max(label_counter.values())
        best_labels = [label for label, freq in label_counter.items() if freq == max_freq]
        return sorted(best_labels)[0]

    # Similarity fallback over all known terms.
    best_similarity = 0.0
    best_terms = []

    for known_term in term_label_frequencies.keys():
        # Compute similarity score between the input term and known term.
        sim = compute_term_similarity(term_key, known_term)
        if sim > best_similarity:
            best_similarity = sim
            best_terms = [known_term]
        elif sim == best_similarity:
            best_terms.append(known_term)

    if not best_terms or best_similarity < min_similarity:
        return None

    # Tie-break by most frequent term in training data.
    best_term = sorted(
        best_terms,
        key=lambda t: (-term_total_frequencies.get(t, 0), t)
    )[0]

    # Return the majority label for the best similar term.
    label_counter = term_label_frequencies.get(best_term, {})
    if not label_counter:
        return None

    # Return the most frequent label for the best similar term.
    max_freq = max(label_counter.values())
    best_labels = [label for label, freq in label_counter.items() if freq == max_freq]
    return sorted(best_labels)[0]
