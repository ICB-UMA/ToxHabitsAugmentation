#!/usr/bin/env python
"""
ToxHabitsAugmentation project entry point

Author: Gallego Donoso, Fernando
        Moreno Barea, Francisco Javier
        Veredas Navarro, Francisco Javier
University: Universidad de Málaga
Group: Inteligencia Computacional en Biomedicina (ICB)
"""

# --- Standard Library Imports ---
import sys

# --- Local Imports ---
from .utils.logging import logger
from .utils.text_utils import load_complete_corpus
from .corpus_analysis.lexical_analysis import print_corpus_analysis_report
from .corpus_analysis.clinical_term_analysis import print_clinical_term_report
from .ingestion.dowload_zenodo import download_toxhabits_datasets
from .ingestion.document_segmentation import segmentate_toxhabits_documents
from .generation.document_generation import run_generation_workflow


def main(
    global_root: str = None,
    download_datasets: bool = False,
    segmentate_documents: bool = False,
    generate_documents: bool = False,
    analyze_corpus: bool = False,
    prompts: list[str] = ["method_1_var"],
    models: list[str] = ["llama3.2-3b"]
) -> None:
    """
    This function is the main entry point and should be called once to start the server.

    Args:
        global_root: Optional root path for resource files
        download_datasets: Flag to trigger dataset downloading
        segmentate_documents: Flag to trigger document segmentation
        generate_documents: Flag to trigger document generation
        analyze_corpus: Flag to trigger corpus analysis
        prompts: List of prompt types to use for generation
        models: List of models to use for generation
    """
    try:
        # Check for required packages
        required_packages = ['requests', 'tqdm', 'ollama']
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.error(f"Missing required packages: {', '.join(missing_packages)}")
            logger.error(f"Please install them using: pip install {' '.join(missing_packages)}")
            sys.exit(1)
        
        if download_datasets:
            # Start dataset download
            download_toxhabits_datasets(global_root)

        if segmentate_documents:
            # Start document segmentation
            segmentate_toxhabits_documents(global_root)
        
        if generate_documents:
            for prompt_type in prompts:
                for model in models:
                    # Start document generation
                    run_generation_workflow(prompt_type, model, global_root)

        if analyze_corpus:
            for corpus_type in ['train', 'test']:
                logger.info(f"\n\nStarting analysis for {corpus_type} corpus...\n")

                # Load complete corpus
                corpus = load_complete_corpus(data_path=f"{global_root}/data/raw/{corpus_type}")

                # Start corpus analysis
                print_corpus_analysis_report(
                    corpus,
                    top_n=15,
                    output_path=f"{global_root}/results/corpus_analysis/{corpus_type}"
                )

                # Start clinical term analysis
                print_clinical_term_report(
                    corpus,
                    top_n=25,
                    output_path=f"{global_root}/results/corpus_analysis/{corpus_type}"
                )

    except KeyboardInterrupt:
        logger.info("\n\nProcess interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nUnexpected error: {e}")
        sys.exit(1)
