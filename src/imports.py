#!/usr/bin/env python
"""
Import usually packages and define constants for the project.
"""

# --- Standard Library Imports ---
import os
import re
import sys
import json
import time
import yaml
import spacy
import ollama
import random
import shutil
import logging
import zipfile
import requests
import numpy as np

from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from typing import Any, List, Dict, Tuple, Optional, Union

# --- Constants ---

# Set seed for randomizers
RANDOM_SEED = 26081993
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Default chunk size for downloads (64 MB)
CHUNK_SIZE = 64 * 1024 * 1024

# Load spaCy model for sentence splitting
SENTENCE_SPLITTER = spacy.load("es_core_news_sm")

# API Keys
GOOGLE_API_KEY = ""
OPENAI_API_KEY = ""

# Supported Models
GEMINI_MODELS = ["gemini"]
OLLAMA_MODELS = ["gemma3-4b", "llama3.2-3b", "ministral3-3b", "qwen3-4b"]
OPENAI_MODELS = ["GPT-4.1_nano", "GPT-5_nano"]
HUGGINGFACE_MODELS = ["Llama-Doctor", "Specialized-Model"]

# Define augmentation method categories
SENTENCE_BY_SENTENCE_METHODS = ["method_1_var", "method_3_guides_inj"]
ALL_TOGETHER_METHODS = ["method_2_context", "method_4_full_gen"]

# Annotation guide prompt
PROMPT_ANNOTATION_GUIDE = "The previous document contains the annotation guidelines for ToxHabits. Please use the annotation guide for the Subshared task 1 to assist in generating augmented text."
