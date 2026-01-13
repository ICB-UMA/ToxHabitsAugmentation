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



def main(
    global_root: str = None
) -> None:
    """
    This function is the main entry point and should be called once to start the server.

    Args:
        global_root: Optional root path for resource files
    """
    try:
        # Check for required packages
        required_packages = ['requests', 'tqdm']
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
        
        # Start dataset download
        download_toxhabits_datasets(global_root)
        
    except KeyboardInterrupt:
        logger.info("\n\nDownload interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nUnexpected error: {e}")
        sys.exit(1)
