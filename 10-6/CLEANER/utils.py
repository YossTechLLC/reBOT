"""
Utility functions for Excel Cleaner.
Contains helper functions for file discovery, column detection, and data filtering.
"""

import re
import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def discover_excel_files(directory: Path, extension: str = '.xlsx', skip_temp: bool = True) -> List[Path]:
    """
    Discover all Excel files in a directory.

    Args:
        directory: Directory to search for Excel files
        extension: File extension to search for (default: .xlsx)
        skip_temp: Skip temporary/hidden Excel files (starting with ~$)

    Returns:
        List of Path objects for Excel files
    """
    if not directory.exists():
        logger.error(f"Directory does not exist: {directory}")
        return []

    excel_files = []

    for file_path in directory.glob(f'*{extension}'):
        # Skip temporary files created by Excel
        if skip_temp and file_path.name.startswith('~$'):
            logger.debug(f"Skipping temporary file: {file_path.name}")
            continue

        excel_files.append(file_path)

    logger.info(f"Found {len(excel_files)} Excel files in {directory}")
    return sorted(excel_files)


def find_address_column(df: pd.DataFrame, possible_names: List[str]) -> Optional[str]:
    """
    Find the address column in a DataFrame by trying multiple possible names.

    Args:
        df: DataFrame to search
        possible_names: List of possible column names to try

    Returns:
        Name of the address column if found, None otherwise
    """
    # Get all column names from the DataFrame
    columns = df.columns.tolist()

    # Try exact case-insensitive matching
    for possible_name in possible_names:
        for col in columns:
            if col.lower() == possible_name.lower():
                logger.debug(f"Found address column: '{col}' (matched '{possible_name}')")
                return col

    # Try partial matching if exact match fails
    for possible_name in possible_names:
        for col in columns:
            if possible_name.lower() in col.lower() or col.lower() in possible_name.lower():
                logger.debug(f"Found address column via partial match: '{col}' (matched '{possible_name}')")
                return col

    logger.warning(f"Address column not found. Available columns: {columns}")
    return None


def contains_test_keyword(text: str, patterns: List[str]) -> bool:
    """
    Check if text contains any test-related keywords using regex patterns.

    Args:
        text: Text to check
        patterns: List of regex patterns to match

    Returns:
        True if any pattern matches, False otherwise
    """
    # Handle NaN, None, or empty values
    if pd.isna(text) or text is None:
        return False

    # Convert to string and strip whitespace
    text_str = str(text).strip()

    # If empty string after stripping, return False
    if not text_str:
        return False

    # Check each pattern (case-insensitive)
    for pattern in patterns:
        try:
            if re.search(pattern, text_str, re.IGNORECASE):
                logger.debug(f"Test keyword found: '{text_str}' matches pattern '{pattern}'")
                return True
        except re.error as e:
            logger.error(f"Invalid regex pattern '{pattern}': {e}")
            continue

    return False


def clean_dataframe(
    df: pd.DataFrame,
    address_column: str,
    test_patterns: List[str],
    output_column_name: str = 'address'
) -> Tuple[pd.DataFrame, dict]:
    """
    Clean a DataFrame by keeping only the address column and filtering test entries.

    Args:
        df: Input DataFrame
        address_column: Name of the address column to keep
        test_patterns: List of regex patterns for test keyword filtering
        output_column_name: Name for the output address column

    Returns:
        Tuple of (cleaned DataFrame, statistics dict)
    """
    stats = {
        'original_rows': len(df),
        'original_columns': len(df.columns),
        'test_entries_removed': 0,
        'empty_entries_removed': 0,
        'final_rows': 0,
        'final_columns': 1
    }

    # Step 1: Keep only the address column
    df_cleaned = df[[address_column]].copy()
    logger.debug(f"Kept only address column: '{address_column}'")

    # Step 2: Remove rows with NaN/empty addresses
    initial_count = len(df_cleaned)
    df_cleaned = df_cleaned.dropna(subset=[address_column])
    stats['empty_entries_removed'] = initial_count - len(df_cleaned)
    logger.debug(f"Removed {stats['empty_entries_removed']} empty address entries")

    # Step 3: Remove rows with empty strings
    df_cleaned = df_cleaned[df_cleaned[address_column].str.strip() != '']

    # Step 4: Filter out test entries
    initial_count = len(df_cleaned)
    mask = df_cleaned[address_column].apply(
        lambda x: not contains_test_keyword(str(x), test_patterns)
    )
    df_cleaned = df_cleaned[mask]
    stats['test_entries_removed'] = initial_count - len(df_cleaned)
    logger.debug(f"Removed {stats['test_entries_removed']} test entries")

    # Step 5: Rename column to standard output name
    df_cleaned = df_cleaned.rename(columns={address_column: output_column_name})

    # Step 6: Reset index
    df_cleaned = df_cleaned.reset_index(drop=True)

    stats['final_rows'] = len(df_cleaned)

    logger.info(
        f"Cleaning complete: {stats['original_rows']} → {stats['final_rows']} rows "
        f"({stats['test_entries_removed']} test entries removed, "
        f"{stats['empty_entries_removed']} empty entries removed)"
    )

    return df_cleaned, stats


def clean_excel_file(
    input_path: Path,
    output_path: Path,
    address_column_names: List[str],
    test_patterns: List[str],
    sheet_name: int = 0,
    output_column_name: str = 'address'
) -> dict:
    """
    Clean a single Excel file.

    Args:
        input_path: Path to input Excel file
        output_path: Path to output Excel file
        address_column_names: List of possible address column names
        test_patterns: List of regex patterns for filtering
        sheet_name: Sheet to process (0 = first sheet)
        output_column_name: Name for output address column

    Returns:
        Dictionary with processing statistics
    """
    result = {
        'success': False,
        'input_file': input_path.name,
        'output_file': output_path.name,
        'error': None,
        'stats': {}
    }

    try:
        # Read Excel file
        logger.info(f"Reading Excel file: {input_path.name}")
        df = pd.read_excel(input_path, sheet_name=sheet_name)

        # Find address column
        address_column = find_address_column(df, address_column_names)

        if address_column is None:
            raise ValueError(
                f"Address column not found in file. Available columns: {df.columns.tolist()}"
            )

        # Clean the DataFrame
        df_cleaned, stats = clean_dataframe(
            df=df,
            address_column=address_column,
            test_patterns=test_patterns,
            output_column_name=output_column_name
        )

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save cleaned Excel file
        logger.info(f"Saving cleaned file to: {output_path.name}")
        df_cleaned.to_excel(output_path, index=False, sheet_name='Cleaned')

        result['success'] = True
        result['stats'] = stats

    except FileNotFoundError as e:
        result['error'] = f"File not found: {e}"
        logger.error(result['error'])

    except ValueError as e:
        result['error'] = str(e)
        logger.error(result['error'])

    except Exception as e:
        result['error'] = f"Unexpected error: {e}"
        logger.error(result['error'], exc_info=True)

    return result


def format_statistics(stats: dict) -> str:
    """
    Format statistics dictionary as a readable string.

    Args:
        stats: Statistics dictionary

    Returns:
        Formatted string
    """
    lines = [
        f"  Original rows: {stats.get('original_rows', 0)}",
        f"  Test entries removed: {stats.get('test_entries_removed', 0)}",
        f"  Empty entries removed: {stats.get('empty_entries_removed', 0)}",
        f"  Final rows: {stats.get('final_rows', 0)}"
    ]
    return '\n'.join(lines)


def setup_logging(verbose: bool = True):
    """
    Set up logging configuration.

    Args:
        verbose: Enable verbose logging (DEBUG level)
    """
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
