#!/usr/bin/env python3
"""
Excel Cleaner - Batch processing utility for cleaning Excel files.

This script:
1. Scans the ToBeCleaned directory for .xlsx files
2. Keeps only the address column from each file
3. Filters out rows containing test-related keywords
4. Saves cleaned files to the Cleaned directory

Usage:
    python cleaner.py
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

# Import configuration and utilities
import config
import utils

# Set up logging
logger = logging.getLogger(__name__)


def print_header():
    """Print application header."""
    print("=" * 80)
    print("Excel Cleaner - Address Data Cleaning Utility")
    print("=" * 80)
    print()


def print_summary(results: list):
    """
    Print summary of processing results.

    Args:
        results: List of result dictionaries from processing
    """
    print("\n" + "=" * 80)
    print("PROCESSING SUMMARY")
    print("=" * 80)

    total_files = len(results)
    successful = sum(1 for r in results if r['success'])
    failed = total_files - successful

    total_original_rows = sum(r['stats'].get('original_rows', 0) for r in results if r['success'])
    total_final_rows = sum(r['stats'].get('final_rows', 0) for r in results if r['success'])
    total_test_removed = sum(r['stats'].get('test_entries_removed', 0) for r in results if r['success'])
    total_empty_removed = sum(r['stats'].get('empty_entries_removed', 0) for r in results if r['success'])

    print(f"\nFiles Processed: {total_files}")
    print(f"  ✓ Successful: {successful}")
    if failed > 0:
        print(f"  ✗ Failed: {failed}")

    print(f"\nData Statistics:")
    print(f"  Total original rows: {total_original_rows}")
    print(f"  Test entries removed: {total_test_removed}")
    print(f"  Empty entries removed: {total_empty_removed}")
    print(f"  Total final rows: {total_final_rows}")

    if total_original_rows > 0:
        retention_rate = (total_final_rows / total_original_rows) * 100
        print(f"  Retention rate: {retention_rate:.1f}%")

    # Show failed files
    if failed > 0:
        print(f"\nFailed Files:")
        for result in results:
            if not result['success']:
                print(f"  ✗ {result['input_file']}: {result['error']}")

    print("\n" + "=" * 80)


def process_file(input_path: Path, output_dir: Path, index: int, total: int) -> dict:
    """
    Process a single Excel file.

    Args:
        input_path: Path to input Excel file
        output_dir: Directory for output files
        index: Current file index (1-based)
        total: Total number of files

    Returns:
        Result dictionary with processing information
    """
    print(f"\n[{index}/{total}] Processing: {input_path.name}")

    # Determine output path
    if config.ADD_TIMESTAMP:
        timestamp = datetime.now().strftime(config.TIMESTAMP_FORMAT)
        output_name = f"{input_path.stem}_{timestamp}{input_path.suffix}"
    else:
        output_name = input_path.name

    output_path = output_dir / output_name

    # Check if output file exists and overwrite is disabled
    if output_path.exists() and not config.OVERWRITE_EXISTING:
        logger.warning(f"Output file exists and overwrite is disabled: {output_path.name}")
        return {
            'success': False,
            'input_file': input_path.name,
            'output_file': output_name,
            'error': 'Output file exists (overwrite disabled)',
            'stats': {}
        }

    # Process the file
    result = utils.clean_excel_file(
        input_path=input_path,
        output_path=output_path,
        address_column_names=config.ADDRESS_COLUMN_NAMES,
        test_patterns=config.TEST_PATTERNS,
        sheet_name=config.SHEET_NAME,
        output_column_name=config.OUTPUT_COLUMN_NAME
    )

    # Print file-specific results
    if result['success']:
        print(f"  ✓ Address column found and processed")
        if config.SHOW_STATISTICS:
            print(utils.format_statistics(result['stats']))
        print(f"  ✓ Saved to: {result['output_file']}")
    else:
        print(f"  ✗ Error: {result['error']}")

    return result


def main():
    """Main application entry point."""
    try:
        # Set up logging
        utils.setup_logging(verbose=config.VERBOSE)

        # Print header
        print_header()

        # Discover Excel files
        print(f"Scanning directory: {config.INPUT_DIR}")
        excel_files = utils.discover_excel_files(
            directory=config.INPUT_DIR,
            extension=config.FILE_EXTENSION,
            skip_temp=config.SKIP_TEMP_FILES
        )

        if not excel_files:
            print("\n⚠️  No Excel files found in ToBeCleaned directory.")
            print(f"   Please place .xlsx files in: {config.INPUT_DIR}")
            return 0

        print(f"Found {len(excel_files)} Excel file(s) to process\n")

        # Process each file
        results = []

        for index, excel_file in enumerate(excel_files, start=1):
            result = process_file(
                input_path=excel_file,
                output_dir=config.OUTPUT_DIR,
                index=index,
                total=len(excel_files)
            )
            results.append(result)

        # Print summary
        if config.SHOW_STATISTICS:
            print_summary(results)

        # Determine exit code
        failed_count = sum(1 for r in results if not r['success'])
        if failed_count > 0:
            print(f"\n⚠️  Completed with {failed_count} error(s)")
            return 1
        else:
            print("\n✓ All files processed successfully!")
            return 0

    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user (Ctrl+C)")
        return 130

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"\n✗ Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
