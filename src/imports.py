#!/usr/bin/env python
"""
Import usually packages and define constants for the project.
"""

# ruff: noqa: F401

# --- Standard Library Imports ---
import os
import re
import sys
import json
import nltk
import time
import yaml
import torch
import spacy
import ollama
import random
import shutil
import logging
import zipfile
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from typing import Any, List, Dict, Tuple, Optional, Union


# --- NLTK Data Download ---
try:
    # Check if the required NLTK data packages are already downloaded
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("corpora/stopwords")
    nltk.data.find("wordnet")
except Exception:

    # If not found, download the necessary NLTK data packages
    print("NLTK data not found. Downloading required packages...")
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    print("NLTK data download complete.")


# --- Constants ---

# Set seed for randomizers
RANDOM_SEED = 26081993
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Device configuration for PyTorch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Default chunk size for downloads (64 MB)
CHUNK_SIZE = 64 * 1024 * 1024

# Maximum number of retries for generation operations
MAXIMUM_NUMBER_OF_RETRIES = 3

# Load spaCy model for sentence splitting
SENTENCE_SPLITTER = spacy.load("es_core_news_sm")

# API Keys
GOOGLE_API_KEY = ""
OPENAI_API_KEY = ""

# Supported Models
GEMINI_MODELS = ["gemini-2.5-flash"]
OLLAMA_MODELS = ["gemma3-4b", "llama3.2-3b", "ministral3-3b", "qwen3-4b"]
OPENAI_MODELS = ["gpt-4.1-nano", "gpt-5-nano"]
HUGGINGFACE_MODELS = ["biomistral-7b", "medical-mt5-large"]

# OpenAI API Cost
PRICE_PER_M_INPUT = 0.10   # $0.10 per million input tokens
PRICE_PER_M_OUTPUT = 0.40  # $0.40 per million output tokens

# Define augmentation method categories
SENTENCE_BY_SENTENCE_METHODS = ["method_1_var", "method_2_guides_inj"]
ALL_TOGETHER_METHODS = ["method_3_context", "method_4_full_gen"]
OVERSAMPLING_METHODS = ["oversampling"]
SYNONYM_REPLACEMENT_METHODS = ["synonym_replacement", "synonym_replacement_dictionary"]

# Annotation guide prompt
PROMPT_ANNOTATION_GUIDE = "The previous document contains the annotation guidelines for ToxHabits. Please use the annotation guide for the Subshared task 1 to assist in generating augmented text."

# Define paths for data storage
AUGMENTATION_PATH = "data/augmented"
CORRECTED_PATH = "data/corrected"

# Define path for experiment cost log CSV
CSV_FILE = "data/experiment_cost_log.csv"

# Define abbreviations mapping for models and methodss
ABBREVIATIONS_MAPPING = {
    "gpt-41-nano": "GPT-4.1",
    "gpt-41-nano-temp-0": "GPT-4.1-temp-0.0",
    "gpt-41-nano-temp-06": "GPT-4.1-temp-0.6",
    "gpt-41-nano-temp-1": "GPT-4.1-temp-1.0",
    "biomistral-7b": "BioMistral",
    "llama3.2-3b": "Llama-3.2",
    "method_1_var": "GSA#1",
    "method_2_guides_inj": "GSA#2",
    "method_3_context": "GSA#3",
    "method_4_full_gen": "GSA#4",
    "oversampling": "Oversampling",
    "synonym_replacement": "Synonym-MLM",
}
