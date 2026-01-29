#!/usr/bin/env python
"""
ToxHabitsAugmentation project entry point script.
"""

# --- Standard Library Imports ---
import os
import sys

# --- Local Imports ---
from src.main import main


def determine_root_path() -> str:
    """
    Determines the root path for resource files.

    This is crucial for finding assets (like icons) correctly,
    whether the application is running from source or as a
    frozen executable (e.g., created by PyInstaller).

    Returns:
        output (str): The determined root path.
    """
    if getattr(sys, 'frozen', False):
        # If the application is run as a bundle/frozen executable (e.g., PyInstaller)
        # sys._MEIPASS is a temporary folder created by PyInstaller
        root_path = sys._MEIPASS
    elif __file__:
        # If running from a standard Python script (__file__ is defined)
        # The root path is the directory containing this script
        root_path = os.path.dirname(__file__)
    else:
        # Fallback if __file__ is not defined (e.g., interactive session, though unlikely for entry point)
        root_path = os.path.abspath(".")
        
    return root_path


# --- Entry Point ---
if __name__ == '__main__':
    # Set the desired parameters
    models = ["gpt-5-nano"]  # List of models to use for generation ["llama3.2-3b", "gpt-5-nano", etc.]
    prompts = ["method_3_guides_inj"]  # List of prompts ["method_1_var", "method_2_context", "method_3_guides_inj"]
    
    main(
        global_root=determine_root_path(),
        download_datasets=False,
        segmentate_documents=False,
        generate_documents=False,
        analyze_corpus=True,
        prompts=prompts,
        models=models
    )
