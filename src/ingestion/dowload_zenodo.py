#!/usr/bin/env python3
"""
Download datasets from Zenodo repositories.

This module provides simple helpers to fetch metadata for a Zenodo record and
download all files associated with it. It uses a requests.Session with retries
and the project's `download_file` utility for robust downloads.
"""

# --- Local Imports ---
from ..utils.logging import logger
from ..utils.file_utils import (
    cleanup_extracted_files,
    create_directories,
    download_file,
    extract_zip_file,
    get_requests_session
)
from ..imports import (
    requests,
    Path,
    List,
    Dict,
    Union,
    Optional
)


def download_toxhabits_datasets(
    base_path: Optional[str] = None
) -> None:
    """
    Download the Training and Test datasets to ``base_path`` (defaults to home).

    Args:
        base_path: optional base directory to store datasets; defaults to the user's home directory
    """
    # Determine base path
    base = Path(base_path) if base_path else Path.home()

    # Define datasets to download
    datasets = [
        {
            "name": "Training",
            "record_id": "15425460",
            "path": str(base / "data" / "raw" / "train"),
            "subfolder": "train_annotations",
        },
        {
            "name": "Test",
            "record_id": "17566029",
            "path": str(base / "data" / "raw" / "test"),
            "subfolder": "test_good",
        },
    ]

    # Create a requests session with retries
    session = get_requests_session()

    for dataset in datasets:
        try:
            download_zenodo_dataset(dataset["record_id"], dataset["path"], dataset["name"], session=session)
        except Exception as exc:
            logger.error("Error downloading %s dataset: %s", dataset["name"], exc, exc_info=True)
            continue

    # Summary
    for dataset in datasets:
        path = Path(dataset["path"])
        if path.exists():
            # List downloaded files
            files = list(path.iterdir())
            total_size = sum(f.stat().st_size for f in files) / (1024 ** 2)
            logger.info("%s: %d files, %.2f MB — Location: %s", dataset["name"], len(files), total_size, path)

            # Process downloaded zip files
            process_dataset_folder(dataset["path"], target_subfolder=dataset["subfolder"])
        else:
            logger.warning("%s: Not downloaded", dataset["name"])


def download_zenodo_dataset(
    record_id: str,
    destination_path: str,
    dataset_name: str,
    session: Optional[requests.Session] = None
) -> None:
    """
    Download all files for a Zenodo record into ``destination_path``.

    This function logs progress and continues to the next file if a single file
    fails to download (so partial progress can still be available).
    
    Args:
        record_id: Zenodo record identifier (as string)
        destination_path: Directory to save downloaded files
        session: optional requests.Session (uses retry-enabled session by default)
    """
    session = session or get_requests_session()

    logger.info("Downloading %s dataset (Record ID: %s)", dataset_name, record_id)

    # Ensure destination directory exists
    create_directories(destination_path)

    # Retrieve file metadata
    files = get_zenodo_files(record_id, session=session)

    if not files:
        # No files to download
        logger.warning("No files found for record %s", record_id)
        return

    logger.info("Starting download of %d file(s)", len(files))

    for i, file_info in enumerate(files, start=1):
        # Extract file details
        filename = file_info.get("filename")
        size = file_info.get("size")
        url = file_info.get("download_url")

        logger.info("[%d/%d] Processing: %s (%.2f MB)", i, len(files), filename, (size or 0) / (1024 ** 2))

        if (dataset_name.lower() in ["test"] and "train" in filename.lower() or 
            dataset_name.lower() in ["train"] and "test" in filename.lower()):
            # Skip files not matching the dataset type
            logger.info("Skipping file %s as it does not match dataset type %s", filename, dataset_name)
            continue
        else:
            try:
                download_file(url, destination_path, filename, file_size=size, session=session)
            except Exception as exc:
                logger.error("Failed to download %s: %s", filename, exc, exc_info=True)
                # Continue downloading other files
                continue

    logger.info("Completed downloading %s dataset to %s", dataset_name, destination_path)


def get_zenodo_files(
    record_id: str,
    session: Optional[requests.Session] = None
) -> List[Dict]:
    """
    Fetch file metadata for a Zenodo record.

    Args:
        record_id: Zenodo record identifier (as string)
        session: optional requests.Session (uses retry-enabled session by default)

    Returns:
        List of dictionaries with keys: 'filename', 'download_url', 'size'

    Raises:
        requests.exceptions.RequestException on network errors
    """
    session = session or get_requests_session()
    api_url = f"https://zenodo.org/api/records/{record_id}"

    logger.debug("Fetching Zenodo metadata for record %s", record_id)
    response = session.get(api_url, timeout=20)
    response.raise_for_status()
    data = response.json()

    files = []
    for file_info in data.get("files", []):
        files.append({
            "filename": file_info.get("key"),
            "download_url": file_info.get("links", {}).get("self"),
            "size": file_info.get("size")
        })

    logger.info("Found %d file(s) for record %s", len(files), record_id)
    return files


def process_dataset_folder(
    folder_path: Union[str, Path], 
    target_subfolder: str = None
) -> None:
    """
    Process all zip files in a folder: extract them and clean up.
    
    Args:
        folder_path: Path to folder containing zip files
        target_subfolder: Specific subfolder name to look for (optional)
    """
    folder_path = Path(folder_path)
    
    # Find all zip files
    zip_files = list(folder_path.glob("*.zip"))
    
    if not zip_files:
        logger.warning(f"No zip files found in {folder_path}")
        return
    
    logger.info(f"Found {len(zip_files)} zip files in {folder_path}")
    
    for zip_file in zip_files:
        logger.info(f"\nProcessing: {zip_file.name}")
        
        try:
            # Extract the zip file
            extract_dir = extract_zip_file(zip_file)
            
            # Clean up the extracted files
            cleanup_extracted_files(extract_dir, target_subfolder)
            
            # Remove the original zip file after successful extraction
            zip_file.unlink()
            logger.info(f"Removed original zip file: {zip_file.name}")
            
        except Exception as e:
            logger.error(f"Failed to process {zip_file.name}: {e}")
