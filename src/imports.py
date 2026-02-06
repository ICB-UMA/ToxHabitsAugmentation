#!/usr/bin/env python
"""
Import usually packages and define constants for the project.
"""

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
except:
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

# Define augmentation method categories
SENTENCE_BY_SENTENCE_METHODS = ["method_1_var", "method_3_guides_inj"]
ALL_TOGETHER_METHODS = ["method_2_context", "method_4_full_gen"]

# Annotation guide prompt
PROMPT_ANNOTATION_GUIDE = "The previous document contains the annotation guidelines for ToxHabits. Please use the annotation guide for the Subshared task 1 to assist in generating augmented text."
