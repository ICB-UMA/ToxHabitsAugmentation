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
from .corpus_analysis.lexical_analysis import print_corpus_analysis_report, print_corpus_word_comparison
from .corpus_analysis.clinical_term_analysis import print_clinical_term_report
from .ingestion.dowload_zenodo import download_toxhabits_datasets
from .ingestion.document_segmentation import segmentate_toxhabits_documents
from .generation.document_generation import run_generation_workflow
from .training.train import train_model


def main(
    global_root: str = None,
    download_datasets: bool = False,
    segmentate_documents: bool = False,
    generate_documents: bool = False,
    analyze_corpus: bool = False,
    comparise_word_frequencies: bool = False,
    prompts: list[str] = ["method_1_var"],
    models: list[str] = ["llama3.2-3b"],
    train: bool = False,
) -> None:
    """
    This function is the main entry point and should be called once to start the server.

    Args:
        global_root: Optional root path for resource files
        download_datasets: Flag to trigger dataset downloading
        segmentate_documents: Flag to trigger document segmentation
        generate_documents: Flag to trigger document generation
        analyze_corpus: Flag to trigger corpus analysis
        comparise_word_frequencies: Flag to trigger word frequency comparison across corpora
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
        
        if comparise_word_frequencies:
            for prompt_type in prompts:
                logger.info(f"\n\nComparing word frequencies for {prompt_type} prompt...\n")

                # Start word frequency comparison across corpora
                print_corpus_word_comparison(
                    data_path=f"{global_root}/data/",
                    prompt_type=prompt_type,
                    models=models,
                    top_n=25,
                    output_path=f"{global_root}/results/word_comparison/{prompt_type}"
                )

        if train:
            logger.info("Starting NER training pipeline...")

            train_model(
                data_path=global_root,
            )

            logger.info("NER training pipeline completed successfully.")

    except KeyboardInterrupt:
        logger.info("\n\nProcess interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"\nUnexpected error: {e}")
        sys.exit(1)
