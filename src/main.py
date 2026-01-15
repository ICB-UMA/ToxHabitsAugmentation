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
from .ingestion.dowload_zenodo import download_toxhabits_datasets
from .ingestion.document_segmentation import segmentate_toxhabits_documents
from .generation.document_generation import run_generation_workflow


def main(
    global_root: str = None,
    download_datasets: bool = False,
    segmentate_documents: bool = False,
    generate_documents: bool = False,
    prompt_type: str = "method_1_var",
    model: str = "llama3.2-3b"
) -> None:
    """
    This function is the main entry point and should be called once to start the server.

    Args:
        global_root: Optional root path for resource files
        download_datasets: Flag to trigger dataset downloading
        segmentate_documents: Flag to trigger document segmentation
        generate_documents: Flag to trigger document generation
        prompt_type: Type of prompt to use for generation
        model: Model to use for document generation
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
            # Start document generation
            run_generation_workflow(prompt_type, model, global_root)

    except KeyboardInterrupt:
        logger.info("\n\nDownload interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nUnexpected error: {e}")
        sys.exit(1)
