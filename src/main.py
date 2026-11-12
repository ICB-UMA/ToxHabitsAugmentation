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
from .corpus_analysis.clinical_term_analysis import print_clinical_term_report
from .corpus_analysis.compare_reports import compare_reports
from .corpus_analysis.lexical_analysis import (
    print_corpus_analysis_report,
    print_corpus_word_comparison,
)
from .corpus_analysis.similarity_analysis import print_similarity_report
from .generation.document_generation import run_generation_workflow
from .imports import ABBREVIATIONS_MAPPING, CORRECTED_PATH
from .ingestion.document_segmentation import segmentate_toxhabits_documents
from .ingestion.dowload_zenodo import download_toxhabits_datasets
from .training.train import train_model
from .utils.logging import logger
from .utils.text_utils import load_complete_corpus
from .utils.transform_labels import transform_labels_function


def main(
    global_root: str = None,
    download_datasets: bool = False,
    segmentate_documents: bool = False,
    generate_documents: bool = False,
    transform_labels: bool = False,
    train_models: bool = False,
    analyze_corpus: bool = False,
    compare_word_frequencies: bool = False,
    similarity_analysis: bool = False,
    compare_reports_results: bool = False,
    prompts: list[str] = ["method_1_var"],
    models: list[str] = ["llama3.2-3b"],
) -> None:
    """
    This function is the main entry point and should be called once to start the server.

    Args:
        global_root: Optional root path for resource files
        download_datasets: Flag to trigger dataset downloading
        segmentate_documents: Flag to trigger document segmentation
        generate_documents: Flag to trigger document generation
        transform_labels: Flag to trigger label transformation
        train_models: Flag to trigger model training
        analyze_corpus: Flag to trigger corpus analysis
        compare_word_frequencies: Flag to trigger word frequency comparison across corpora
        similarity_analysis: Flag to trigger similarity analysis
        compare_reports_results: Flag to trigger report comparison
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

        if transform_labels:
            # Start label transformation
            transform_labels_function(
                methods=prompts,
                models=models
            )

        if analyze_corpus:
            # Define paths for original and augmented corpora
            corpus_paths = [f"{global_root}/data/raw/train", f"{global_root}/data/raw/test"]
            corpus_name = ["train", "test"]

            for prompt_type in prompts:
                # ["Baseline"]:
                for model in models:
                    # ["oversampling", "synonym_replacement", "synonym_replacement_dictionary"]:
                    prompt_abbr = ABBREVIATIONS_MAPPING.get(prompt_type, prompt_type)
                    model_abbr = model.replace(".", "")
                    corpus_paths.append(f"{global_root}/{CORRECTED_PATH}/{prompt_abbr}/{model_abbr}")
                    corpus_name.append(f"{model_abbr}_{prompt_abbr}")

            for data_path, corpus_type in zip(corpus_paths, corpus_name):
                logger.info(f"\n\nStarting analysis for {corpus_type} corpus...\n")

                # Load complete corpus
                corpus = load_complete_corpus(data_path=data_path)

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
        
        if compare_word_frequencies:
            for prompt_type in prompts:
                prompt_abbr = ABBREVIATIONS_MAPPING.get(prompt_type, prompt_type)
                logger.info(f"\n\nComparing word frequencies for {prompt_abbr} prompt...\n")

                # Start word frequency comparison across corpora
                print_corpus_word_comparison(
                    data_path=f"{global_root}/data/",
                    prompt_type=prompt_abbr,
                    models=models,
                    top_n=25,
                    output_path=f"{global_root}/results/word_comparison/{prompt_abbr}"
                )

        if similarity_analysis:
            logger.info("\n\nStarting similarity analysis...\n")

            # Start similarity analysis
            print_similarity_report(
                data_path=f"{global_root}/data/",
                prompts=prompts,
                models=models,
                output_path=f"{global_root}/results/similarity_analysis"
            )

        if train_models:
            logger.info("Starting NER training pipeline...")

            # Start NER model training
            train_model(
                data_path=global_root,
            )

            logger.info("NER training pipeline completed successfully.")

        if similarity_analysis:
            logger.info("\n\nStarting similarity analysis...\n")

            # Start similarity analysis
            print_similarity_report(
                data_path=f"{global_root}/data/",
                prompts=prompts,
                models=models,
                output_path=f"{global_root}/results/similarity_analysis"
            )

        if compare_reports_results:
            logger.info("\n\nStarting report comparison...\n")

            # Start report comparison
            compare_reports(results_dir=f"{global_root}/results/corpus_analysis")

    except KeyboardInterrupt:
        logger.info("\n\nProcess interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"\nUnexpected error: {e}")
        sys.exit(1)
