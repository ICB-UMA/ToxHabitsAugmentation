#!/usr/bin/env python3
"""
Utilities for file handling and downloads.

This module provides functions to:
- Create directory structures if they don't exist.
- Download files with progress bars and retry logic.
- Extract zip files to specified directories.
- Clean up extracted files by removing unnecessary folders.
- Load model parameters from YAML configuration files.
"""

# --- Third-Party Imports ---
from urllib3.util import Retry
from requests.adapters import HTTPAdapter

# --- Local Imports ---
from .logging import logger
from ..imports import (
    os,
    tqdm,
    yaml,
    shutil,
    zipfile,
    requests,
    Any,
    Dict,
    Path,
    Union,
    Optional,
    CHUNK_SIZE
)


def cleanup_extracted_files(
    parent_folder: Union[str, Path], 
    target_subfolder: str = None,
    task_2_folder: str = "annotations_task_2",
    macosx_folder: str = "__MACOSX"
) -> None:
    """
    Clean up extracted files by:
    1. Removing __MACOSX folder if it exists
    2. Removing annotations_task_2 folder if it exists
    3. Moving files from target subfolder to parent folder
    4. Removing the target subfolder
    
    Args:
        parent_folder: Path to the folder containing the extracted files
        target_subfolder: Name of the subfolder containing the actual data
                         (e.g., "test_good" or "train_annotations")
                         If None, will try to auto-detect common folder names
        task_2_folder: Name of the TASK-2 folder to remove
        macosx_folder: Name of the macOS metadata folder to remove
    """
    parent_folder = Path(parent_folder)
    
    # Auto-detect target subfolder if not specified
    if target_subfolder is None:
        # Look for common folder names
        possible_folders = ["test_annotations", "train_annotations", "test", "train"]
        for folder in possible_folders:
            candidate = parent_folder / folder
            if candidate.is_dir():
                target_subfolder = folder
                break
        
        if target_subfolder is None:
            # If no common folder found, use the first non-__MACOSX directory
            dirs = [d for d in parent_folder.iterdir() if d.is_dir() and d.name != macosx_folder]
            if dirs:
                target_subfolder = dirs[0].name
    
    if target_subfolder is None:
        logger.info(f"No target subfolder found in {parent_folder}")
        return
    
    # Define full path to target subfolder
    target_path = parent_folder / target_subfolder
    
    logger.info(f"Cleaning up {parent_folder}: moving files from '{target_subfolder}'")
    
    # Remove __MACOSX folder if it exists
    macosx_path = parent_folder / macosx_folder
    if macosx_path.exists() and macosx_path.is_dir():
        logger.info(f"Removing {macosx_folder} folder")
        try:
            shutil.rmtree(macosx_path)
            logger.info(f"Removed {macosx_folder} folder")
        except Exception as e:
            logger.warning(f"Could not remove {macosx_folder} folder: {e}")

    # Remove TASK-2 folder if it exists
    task_2_path = parent_folder / task_2_folder
    if task_2_path.exists() and task_2_path.is_dir():
        logger.info(f"Removing {task_2_path} folder")
        try:
            shutil.rmtree(task_2_path)
            logger.info(f"Removed {task_2_path} folder")
        except Exception as e:
            logger.warning(f"Could not remove {task_2_path} folder: {e}")
    
    # Move files from target subfolder to parent folder
    if target_path.exists() and target_path.is_dir():
        logger.info(f"Moving files from {target_path} to {parent_folder}")
        
        moved_count = 0
        for item in target_path.iterdir():
            new_path = parent_folder / item.name
            
            # Handle potential name conflicts
            if new_path.exists():
                if new_path.is_dir() and item.is_dir():
                    # Merge directories
                    logger.info(f"Merging directory {item.name}")
                    for subitem in item.iterdir():
                        shutil.move(str(subitem), str(parent_folder / subitem.name))
                    shutil.rmtree(item)
                else:
                    # Skip duplicated files
                    continue
            else:
                shutil.move(str(item), str(new_path))
                moved_count += 1
        
        logger.info(f"Moved {moved_count} items from {target_subfolder} to parent folder")
        
        # Remove the now-empty target subfolder
        try:
            target_path.rmdir()
            logger.info(f"Removed empty folder: {target_subfolder}")
        except OSError:
            # Folder might not be empty due to hidden files or permission issues
            try:
                shutil.rmtree(target_path)
                logger.info(f"Removed non-empty folder: {target_subfolder}")
            except Exception as e:
                logger.error(f"Could not remove {target_subfolder} folder: {e}")
    else:
        logger.info(f"Target subfolder '{target_subfolder}' not found in {parent_folder}")


def create_directories(
    base_path: str
) -> None:
    """
    Create directory structure if it doesn't exist and log the action.

    Args:
        base_path: Path-like string to create
    """
    Path(base_path).mkdir(parents=True, exist_ok=True)
    logger.info("Created/verified directory: %s", base_path)


def download_file(
    url: str,
    destination_path: str,
    filename: str,
    file_size: Optional[int] = None,
    session: Optional[requests.Session] = None,
    chunk_size: int = CHUNK_SIZE
) -> Path:
    """
    Download a file to ``destination_path`` with a progress bar.

    This function is resilient to transient network errors (uses a session with retries
    by default), validates an existing file by size, and cleans up partial files on error.

    Args:
        url: URL to download from
        destination_path: directory to save the file to
        filename: filename to use for saving
        file_size: expected file size in bytes (optional; helps skip re-downloads)
        session: optional :class:`requests.Session` to use
        chunk_size: bytes per chunk written to disk

    Returns:
        Path object pointing to the downloaded file

    Raises:
        requests.exceptions.RequestException: for network-related errors
    """
    session = session or get_requests_session()
    dest = Path(destination_path)
    dest.mkdir(parents=True, exist_ok=True)
    file_path = dest / filename

    # If file exists and matches expected size, skip download
    if file_path.exists():
        existing_size = file_path.stat().st_size
        if file_size and existing_size == file_size:
            logger.info("%s already exists with correct size (%d bytes); skipping.", filename, existing_size)
            return file_path
        else:
            logger.info(
                "%s exists but size mismatch (have=%d, expected=%s); re-downloading.",
                filename, existing_size, str(file_size)
            )
            file_path.unlink()

    try:
        # If file_size not provided, try a HEAD request to discover it
        if file_size is None:
            try:
                head = session.head(url, allow_redirects=True, timeout=10)
                head.raise_for_status()
                file_size = int(head.headers.get("content-length", 0)) or None
            except requests.exceptions.RequestException:
                # Not fatal — we'll rely on streaming response headers
                logger.debug("HEAD request failed for %s; will attempt GET to determine size.", url)

        # Stream the GET request and write to file in chunks
        with session.get(url, stream=True, timeout=30) as response:
            response.raise_for_status()
            total_size = file_size or int(response.headers.get("content-length", 0) or 0)

            with open(file_path, "wb") as fh:
                with tqdm(total=total_size, unit="B", unit_scale=True,
                               desc=f"Downloading {filename}") as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            fh.write(chunk)
                            pbar.update(len(chunk))

        logger.info("Downloaded: %s", filename)
        return file_path

    except requests.exceptions.RequestException as e:
        logger.exception("Error downloading %s: %s", filename, e)
        # Clean-up partial file
        if file_path.exists():
            try:
                file_path.unlink()
            except Exception:
                logger.warning("Failed to remove partial file %s", file_path)
        # Re-raise so callers can decide how to handle the error
        raise


def extract_filename_without_extension(
    file_path: str
) -> str:
    """
    Extract filename without extension from a path.

    Args:
        file_path: Path to the file to extract without its extension
    
    Returns:
        Extracted filename without extension
    """
    return os.path.splitext(os.path.basename(file_path))[0]


def extract_zip_file(
    zip_path: Union[str, Path],
    extract_to: Union[str, Path] = None
) -> Path:
    """
    Extract a zip file to the specified directory.
    
    Args:
        zip_path: Path to the zip file to extract
        extract_to: Directory to extract to (defaults to same directory as zip file)
    
    Returns:
        Path to the extraction directory
    """
    # Ensure zip_path is a Path object
    zip_path = Path(zip_path)
    
    if extract_to is None:
        extract_to = zip_path.parent
    else:
        extract_to = Path(extract_to)
    
    # Create extraction directory if it doesn't exist
    extract_to.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Extracting {zip_path.name} to {extract_to}")
    
    try:
        # Extract the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        logger.info(f"Successfully extracted {zip_path.name}")
        return extract_to
    except zipfile.BadZipFile:
        logger.error(f"Error: {zip_path} is not a valid zip file")
        raise
    except Exception as e:
        logger.error(f"Error extracting {zip_path}: {e}")
        raise


def get_requests_session(
    retries: int = 3,
    backoff_factor: float = 0.3,
    status_forcelist=(500, 502, 504)
) -> requests.Session:
    """
    Return a :class:`requests.Session` configured with retry/backoff.

    Args:
        retries: total number of retries for idempotent requests
        backoff_factor: backoff multiplier between retry attempts
        status_forcelist: HTTP status codes that should trigger a retry

    Returns:
        Configured ``requests.Session`` instance
    """
    session = requests.Session()

    # Configure retries with backoff
    retry = Retry(total=retries, backoff_factor=backoff_factor,
                  status_forcelist=status_forcelist, allowed_methods=frozenset(['GET', 'HEAD']))
    
    # Mount the HTTPAdapter with the retry configuration
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def load_params_from_yaml(
    yaml_file_path: str
) -> Dict[str, Any]:
    """
    Loads model configuration from a YAML file.

    Expected YAML structure:
    - model_name: Required name of the model
    - system_prompt: Optional system prompt for the model
    - generation_options: Optional dictionary of generation parameters

    Args:
        yaml_file_path: Full path to your YAML configuration file

    Returns:
        Dictionary with keys: model_name, system_prompt, generation_options
    """
    if not yaml_file_path.endswith((".yml", ".yaml")):
        logger.error("File must be a YAML file with .yml or .yaml extension")
        raise ValueError("File must be a YAML file with .yml or .yaml extension")

    try:
        with open(yaml_file_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"YAML config file not found at {yaml_file_path}")
        raise FileNotFoundError(f"YAML config file not found at {yaml_file_path}")

    # Validate required fields
    if "model_name" not in config:
        logger.error("YAML config is missing required field: `model_name`")
        raise KeyError("YAML config is missing required field: `model_name`")

    # Set defaults for optional fields
    return {
        "model_name": config["model_name"],
        "system_prompt": config.get("system_prompt"),
        "generation_options": config.get("generation_options", {})
    }
