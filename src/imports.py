#!/usr/bin/env python
"""
Import usually packages and define constants for the project.
"""

# --- Standard Library Imports ---
import os
import re
import sys
import json
import spacy
import random
import shutil
import logging
import zipfile
import requests
import numpy as np

from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from typing import Any, List, Dict, Optional, Union

# --- Constants ---

# Set seed for randomizers
RANDOM_SEED = 26081993
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Default chunk size for downloads (64 MB)
CHUNK_SIZE = 64 * 1024 * 1024

# Load spaCy model for sentence splitting
SENTENCE_SPLITTER = spacy.load("es_core_news_sm")
