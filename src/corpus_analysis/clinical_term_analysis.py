#!/usr/bin/env python3
"""
Clinical Term Analysis Module

This module provides functions to analyze the frequency of clinical terms
within a document corpus, considering parent-child relationships in entity types.
It also includes functionality to validate entities and generate detailed
reports of the analysis.
"""

# --- Local Imports ---
from ..ner.entity_validation import validate_entities
from ..utils.logging import logger
from ..imports import (
    os,
    pd,
    plt,
    json,
    Dict,
    List,
    Path,
    Counter,
    Optional,
    defaultdict
)


def analyze_frequencies(
    corpus: Dict
) -> Dict:
    """
    Analyzes the frequency of each entity within the document corpus.
    Considers parent-child relationships in entity types.

    Args:
        corpus: A dictionary where keys are document names and values
                contain entity data for each document.
    
    Returns:
        A dictionary with frequency counts and statistics.
    """
    # Global counters
    text_freq = Counter()
    type_freq = Counter()
    text_type_freq = Counter()
    total_errors = 0
    total_documents_with_errors = 0

    # Parent entity counters
    parent_freq = Counter() 
    parent_children = defaultdict(Counter)
    docs_per_parent = Counter()
    parent_subchildren = {}
    
    # Track document frequency (how many docs contain each entity)
    docs_per_text = Counter()
    docs_per_entity = Counter()
    total_texts = 0
    total_entities = 0
    
    for _, data in corpus.items():
        doc_texts = set()  # Track unique texts in this document
        doc_entities = set()  # Track unique entities in this document
        doc_parents = set()  # Track unique parents in this document
        
        # Count annotation errors
        total_errors += data.get("entity_errors", 0)
        if data.get("entity_errors", 0) > 0:
            total_documents_with_errors += 1
        
        for entity in data["entities"]:
            # Normalize text to lowercase for frequency counting
            text = entity["text"].lower()
            etype = entity["type"]
            
            # Update global frequencies
            text_freq[text] += 1
            type_freq[etype] += 1
            text_type_freq[(text, etype)] += 1
            
            # Track unique texts in this document
            doc_texts.add(text)
            total_texts += 1

            # Track unique entities in this document
            doc_entities.add(etype)
            total_entities += 1

            # Determine parent entity
            if "-" in etype:
                parent = etype.split("-")[0]
            else:
                parent = etype  # Standalone entity is its own parent

            # Aggregate under parent
            parent_freq[parent] += 1
            parent_children[parent][etype] += 1
            doc_parents.add(parent)

            # Track subchildren frequencies
            if parent not in parent_subchildren:
                parent_subchildren[parent] = defaultdict(Counter)
            parent_subchildren[parent][etype][text] += 1
        
        # Update document frequencies for texts
        for text in doc_texts:
            docs_per_text[text] += 1

        # Update document frequencies for entities
        for etype in doc_entities:
            docs_per_entity[etype] += 1

        # Update document frequencies for parents
        for parent in doc_parents:
            docs_per_parent[parent] += 1
    
    # Prepare final results
    num_docs = len(corpus)
    results = {
        "by_text": text_freq,
        "by_type": type_freq,
        "by_text_type": text_type_freq,
        "documents_per_text": docs_per_text,
        "documents_per_entity": docs_per_entity,
        "by_parent": parent_freq,
        "parent_children": dict(parent_children),
        "parent_subchildren": {k: dict(v) for k, v in parent_subchildren.items()},
        "documents_per_parent": docs_per_parent,
        "stats": {
            "total_entities": total_entities,
            "unique_texts": len(text_freq),
            "unique_types": len(type_freq),
            "unique_parents": len(parent_freq),
            "avg_entities_per_doc": total_entities / num_docs if num_docs > 0 else 0,
            "num_documents": num_docs,
            "total_errors": total_errors,
            "documents_with_errors": total_documents_with_errors
        }
    }

    return results


def export_parent_hierarchy(
    freqs: Dict,
    output_path: str,
    output_file: Optional[str] = "parent_hierarchy.json",
):
    """
    Export parent-child relationships to JSON for external analysis.

    Args:
        freqs: Frequency analysis results.
        output_path: Directory to save the results.
        output_file: Filename for the saved JSON file.
    """
    # Build hierarchy dictionary
    hierarchy = {}
    for parent, children in freqs["parent_children"].items():
        hierarchy[parent] = {
            "total_count": freqs["by_parent"][parent],
            "document_count": freqs["documents_per_parent"][parent],
            "children": dict(children.most_common())
        }
    
    # Save to JSON file
    output_file = os.path.join(output_path, output_file)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(hierarchy, f, indent=2, ensure_ascii=False)
    
    print(f"Parent hierarchy exported to {output_file}")


def plot_entity_frequencies(
    freqs: Dict,
    output_path: Optional[str] = None,
    output_file: str = "entity_frequencies.jpg",
    top_n: int = 10
) -> None:
    """
    Plots the top N entity types by frequency as a horizontal bar chart.

    Args:
        freqs: Frequency analysis results from analyze_frequencies function.
        output_path: Optional path to save the plot image.
        output_file: Filename for the saved plot image.
        top_n: Number of top entity types to display.
    """
    # Prepare data for plotting
    entity_types = []
    counts = []
    for etype, count in freqs["by_type"].most_common(top_n):
        entity_types.append(etype)
        counts.append(count)
    
    # Create horizontal bar chart
    plt.figure(figsize=(10, 6))
    plt.barh(entity_types[::-1], counts[::-1], color='skyblue')
    plt.xlabel('Frequency')
    plt.title(f'Top {top_n} Entity Types by Frequency')
    plt.tight_layout()
    
    # Save or show plot
    if output_path:
        output_file = os.path.join(output_path, output_file)
        plt.savefig(output_file)
        logger.info(f"Entity frequency plot saved to {output_file}")
    else:
        plt.show()
    
    plt.close()


def print_frequency_corpus_report(
    freqs: Dict,
    lines: Optional[list] = [],
    output_path: Optional[str] = None
):
    """ 
    Print frequency analysis of entities by parent type with hierarchical display.

    Args:
        freqs: Frequency analysis results from analyze_frequencies function.
        lines: List to append report lines to.
        output_path: Optional path to save the report as a text file.

    Returns:
        lines: List of strings representing the report lines.
    """
    # Get overall stats
    stats = freqs["stats"]
    
    # Create DataFrame
    stats_df = pd.DataFrame({
        "Metric": [
            "Documents",
            "Total entities",
            "Unique texts",
            "Unique types",
            "Avg entities/doc",
            "Annotation errors",
            "Documents with errors"
        ],
        "Value": [
            stats["num_documents"],
            stats["total_entities"],
            stats["unique_texts"],
            stats["unique_types"],
            f"{stats['avg_entities_per_doc']:.2f}",
            stats["total_errors"],
            stats["documents_with_errors"]
        ]
    })

    if output_path:
        # Save corpus stats to CSV
        stats_df.to_csv(os.path.join(output_path, "frequency_corpus_report.csv"), index=False, sep=";")
    
    # Generate report
    lines.append("CORPUS ENTITIES STATISTICS")
    lines.append("-" * 30)
    lines.append(stats_df.to_markdown(index=False, tablefmt="pipe", colalign=["left", "right"]))
    lines.append("-" * 30)

    return lines


def print_frequency_entities_report(
    freqs: Dict,
    lines: Optional[list] = [], 
    show_children: bool = True,
    show_subchildren: bool = True
):
    """ 
    Print frequency analysis of entities by parent type with hierarchical display.

    Args:
        freqs: Frequency analysis results from analyze_frequencies function.
        lines: List to append report lines to.
        show_children: Whether to display child entity types.
        show_subchildren: Whether to display sub-child entity types.

    Returns:
        lines: List of strings representing the report lines.
    """
    # Get overall stats
    stats = freqs["stats"]

    # Prepare data for frequency by text type
    parents_data = []
    for parent, count in freqs["by_parent"].most_common():
        pct_entities = (count / stats["total_entities"]) * 100
        
        # Calculate document frequency for types (count docs with ≥1 entity of this type)
        type_docs = freqs["documents_per_parent"][parent]
        pct_docs = (type_docs / stats["num_documents"]) * 100 if stats["num_documents"] > 0 else 0
        
        # Build parent row
        parent_info = {
            "Count": count,
            "% Entities": f"{pct_entities:.1f}",
            "Docs": type_docs,
            "% Docs": f"{pct_docs:.1f}",
            "Parent": parent
        }

        if show_children:
            # Add placeholder columns for children
            parent_info["Children"] = ""
            parent_info["Count C."] = ""
            parent_info["% C."] = ""

            if show_subchildren:
                # Add placeholder columns for sub-children
                parent_info["Sub-Children"] = ""
                parent_info["Count SC."] = ""
                parent_info["% SC."] = ""

        # Append parent row
        parents_data.append(parent_info)
        
        if show_children and parent in freqs["parent_children"]:
            # Add child rows
            children = freqs["parent_children"][parent]
            
            for child, child_count in children.most_common():
                pct = (child_count / count) * 100 if count > 0 else 0

                # Build child row
                children_info = {
                    "Count": "",
                    "% Entities": "",
                    "Docs": "",
                    "% Docs": "",
                    "Parent": "",
                    "Children": f"└─ {child}",
                    "Count C.": f"{child_count}",
                    "% C.": f"{pct:.1f}"
                }
                
                if show_subchildren:
                    # Add placeholder columns for sub-children
                    children_info["Sub-Children"] = ""
                    children_info["Count SC."] = ""
                    children_info["% SC."] = ""
                
                # Append child row
                parents_data.append(children_info)
            
                if show_subchildren and parent in freqs["parent_subchildren"]:
                    if child in freqs["parent_subchildren"][parent]:
                        subchildren = freqs["parent_subchildren"][parent][child]

                        for subchild, subchild_count in subchildren.most_common():
                            pct_sc = (subchild_count / child_count) * 100 if child_count > 0 else 0
                            
                            # Build sub-child row
                            subchildren_info = {
                                "Count": "",
                                "% Entities": "",
                                "Docs": "",
                                "% Docs": "",
                                "Parent": "",
                                "Children": "",
                                "Count C.": "",
                                "% C.": "",
                                "Sub-Children": f"└─ {subchild}",
                                "Count SC.": f"{subchild_count}",
                                "% SC.": f"{pct_sc:.1f}"
                            }
                            
                            # Append sub-child row
                            parents_data.append(subchildren_info)
    
    # Create DataFrame
    parents_df = pd.DataFrame(parents_data)

    # Determine left-aligned columns
    align_columns = []
    for col in parents_df.columns:
        if col in ["Parent", "Children", "Sub-Children"]:
            align_columns.append("left")
        else:
            align_columns.append("right")

    # Prepare stats title
    title = "FREQUENCY OF PARENT ENTITIES"
    separator = "-" * 57
    if show_children:
        title += " (with Children"
        separator = "-" * 63
        if show_subchildren:
            title += " and Sub-Children"
        title += ")"
        separator = "-" * 140
    
    # Generate report
    if lines:
        lines.append("\n")
    lines.append(f"{title}")
    lines.append(separator)
    lines.append(parents_df.to_markdown(index=False, tablefmt="pipe", colalign=align_columns))
    lines.append(separator)

    return lines


def print_frequency_texts_report(
    freqs: Dict,
    lines: Optional[list] = [],
    top_n: int = 10,
    output_path: Optional[str] = None
):
    """ 
    Print frequency analysis results for texts with high-quality Jupyter Notebook tables.

    Args:
        freqs: Frequency analysis results from analyze_frequencies function.
        lines: List to append report lines to.
        top_n: Number of top texts to display.
        output_path: Optional path to save the report as a CSV file.

    Returns:
        lines: List of strings representing the report lines.
    """
    # Get overall stats
    stats = freqs["stats"]

    # Prepare data for frequency by text type
    top_texts_data = []
    for text, count in freqs["by_text"].most_common(top_n):
        pct_text = (count / stats["unique_texts"]) * 100
        
        # Calculate document frequency for types (count docs with ≥1 text of this type)
        text_docs = freqs["documents_per_text"][text]
        pct_docs = (text_docs / stats["num_documents"]) * 100 if stats["num_documents"] > 0 else 0
        
        top_texts_data.append({
            "Count": count,
            "% Texts": f"{pct_text:.1f}",
            "Docs": text_docs,
            "% Docs": f"{pct_docs:.1f}",
            "Entity": f"{text}"
        })
    
    # Create DataFrame
    texts_df = pd.DataFrame(top_texts_data)
    
    if output_path:
        # Save corpus stats to CSV
        texts_df.to_csv(os.path.join(output_path, "frequency_texts_report.csv"), index=False, sep=";")
    
    # Determine left-aligned columns
    align_columns = []
    for col in texts_df.columns:
        if col in ["Entity"]:
            align_columns.append("left")
        else:
            align_columns.append("right")
    
    # Generate report
    if lines:
        lines.append("\n")
    lines.append(f"TOP {top_n} MOST FREQUENT ENTITY TEXTS")
    lines.append("-" * 67)
    lines.append(texts_df.to_markdown(index=False, tablefmt="pipe", colalign=align_columns))
    lines.append("-" * 67)

    return lines


def print_simple_frequency_entities_report(
    freqs: Dict,
    lines: Optional[list] = [],
    output_path: Optional[str] = None
):
    """ 
    Print frequency analysis results with high-quality Jupyter Notebook tables.

    Args:
        freqs: Frequency analysis results from analyze_frequencies function.
        lines: List to append report lines to.
        output_path: Optional path to save the report as a CSV file.
    
    Returns:
        lines: List of strings representing the report lines.
    """
    # Get overall stats
    stats = freqs["stats"]

    # Prepare data for frequency by entity type
    type_data = []
    for etype, count in freqs["by_type"].most_common():
        pct_entities = (count / stats["total_entities"]) * 100
        
        # Calculate document frequency for types (count docs with ≥1 entity of this type)
        type_docs = freqs["documents_per_entity"][etype]
        pct_docs = (type_docs / stats["num_documents"]) * 100 if stats["num_documents"] > 0 else 0
        
        # Build type row
        type_data.append({
            "Count": count,
            "% Entities": f"{pct_entities:.1f}",
            "Docs": type_docs,
            "% Docs": f"{pct_docs:.1f}",
            "Entity Type": etype
        })
    
    # Create DataFrame
    types_df = pd.DataFrame(type_data)
    
    if output_path:
        # Save corpus stats to CSV
        types_df.to_csv(os.path.join(output_path, "frequency_entities_report.csv"), index=False, sep=";")
    
    # Determine left-aligned columns
    align_columns = []
    for col in types_df.columns:
        if col in ["Entity Type"]:
            align_columns.append("left")
        else:
            align_columns.append("right")
    
    # Generate report
    if lines:
        lines.append("\n")
    lines.append("FREQUENCY BY ENTITY TYPE")
    lines.append("-" * 62)
    lines.append(types_df.to_markdown(index=False, tablefmt="pipe", colalign=align_columns))
    lines.append("-" * 62)

    return lines


def print_clinical_term_report(
    corpus: Dict[str, Dict],
    top_n: int = 10,
    output_path: Optional[str] = None
):
    """
    Perform clinical term frequency analysis and validation, then print reports.

    Args:
        corpus: A dictionary where keys are document names and values
                contain entity data for each document.
        top_n: Number of top items to display for frequency-based metrics.
        output_path: If provided, the path to save the report.
    """
    # Analyze frequencies
    entities_frequencies = analyze_frequencies(corpus)
    
    # Validate entities
    errors = validate_entities(corpus)

    # Print frequency reports
    lines = print_frequency_corpus_report(entities_frequencies, output_path=output_path)
    lines = print_simple_frequency_entities_report(entities_frequencies, lines, output_path=output_path)
    lines = print_frequency_texts_report(entities_frequencies, lines, top_n=top_n, output_path=output_path)

    # Print hierarchical frequency reports
    for option in [(False, False), (True, False), (True, True)]:
        show_children, show_subchildren = option
        lines = print_frequency_entities_report(entities_frequencies, lines, show_children=show_children, show_subchildren=show_subchildren)
    
    # Combine lines into final report
    report = "\n".join(lines)

    if output_path:
        output_path = Path(output_path)
        os.makedirs(output_path, exist_ok=True)

        # Save report to txt file
        output_file = os.path.join(output_path, "clinical_term_analysis_report.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"Clinical term analysis report saved to {output_file}")

    # Export results to JSON
    save_analysis_results(corpus, errors, entities_frequencies, output_path)

    # Export parent hierarchy
    export_parent_hierarchy(entities_frequencies, output_path)


def save_analysis_results(
    corpus: Dict[str, Dict],
    errors: List[str],
    freqs: Dict,
    output_path: str,
    output_file: Optional[str] = "entities_analysis_results.json"
):
    """
    Saves the analysis results to a JSON file.

    Args:
        corpus: The document corpus.
        errors: List of validation errors.
        freqs: Frequency analysis results.
        output_path: Path to save the results.
        output_file: Filename for the saved JSON file.
    """
    # Prepare results dictionary
    results = {
        "corpus_info": {
            doc: {
                "num_entities": len(data["entities"]),
                "entity_types": list(set(e["type"] for e in data["entities"]))
            }
            for doc, data in corpus.items()
        },
        "validation_errors": errors,
        "frequencies": {
            "by_text": dict(freqs["by_text"]),
            "by_type": dict(freqs["by_type"])
        }
    }

    # Save to JSON file
    with open(os.path.join(output_path, output_file), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
