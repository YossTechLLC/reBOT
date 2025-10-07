"""
Excel handler module for reading input addresses and writing output results.
Supports both pandas and openpyxl for flexibility.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)


class ExcelHandler:
    """Handles all Excel read/write operations for the scraper application."""

    @staticmethod
    def read_addresses(
        file_path: Path,
        address_column: str = "address",
        sheet_name: str = "Sheet1"
    ) -> List[str]:
        """
        Read addresses from an Excel file.

        Args:
            file_path: Path to input Excel file
            address_column: Name of the column containing addresses
            sheet_name: Name of the sheet to read from

        Returns:
            List of address strings

        Raises:
            FileNotFoundError: If Excel file doesn't exist
            ValueError: If address column is not found
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Input Excel file not found: {file_path}")

        try:
            logger.info(f"Reading addresses from {file_path}")

            # Read Excel file
            df = pd.read_excel(file_path, sheet_name=sheet_name)

            # Check if address column exists
            if address_column not in df.columns:
                available_columns = ", ".join(df.columns)
                raise ValueError(
                    f"Column '{address_column}' not found in Excel file. "
                    f"Available columns: {available_columns}"
                )

            # Extract addresses and clean data
            addresses = df[address_column].dropna().astype(str).str.strip().tolist()

            # Remove empty strings
            addresses = [addr for addr in addresses if addr]

            logger.info(f"Successfully read {len(addresses)} addresses from Excel")
            return addresses

        except Exception as e:
            logger.error(f"Error reading Excel file: {e}")
            raise

    @staticmethod
    def validate_input_file(
        file_path: Path,
        address_column: str = "address",
        sheet_name: str = "Sheet1"
    ) -> Dict[str, Any]:
        """
        Validate the input Excel file structure.

        Args:
            file_path: Path to input Excel file
            address_column: Expected address column name
            sheet_name: Sheet name to validate

        Returns:
            Dictionary with validation results
        """
        validation_result = {
            "valid": False,
            "file_exists": False,
            "sheet_exists": False,
            "column_exists": False,
            "address_count": 0,
            "errors": []
        }

        try:
            # Check file existence
            if not file_path.exists():
                validation_result["errors"].append(f"File not found: {file_path}")
                return validation_result

            validation_result["file_exists"] = True

            # Read Excel file
            excel_file = pd.ExcelFile(file_path)

            # Check sheet existence
            if sheet_name not in excel_file.sheet_names:
                validation_result["errors"].append(
                    f"Sheet '{sheet_name}' not found. Available sheets: {', '.join(excel_file.sheet_names)}"
                )
                return validation_result

            validation_result["sheet_exists"] = True

            # Read the sheet
            df = pd.read_excel(file_path, sheet_name=sheet_name)

            # Check column existence
            if address_column not in df.columns:
                validation_result["errors"].append(
                    f"Column '{address_column}' not found. Available columns: {', '.join(df.columns)}"
                )
                return validation_result

            validation_result["column_exists"] = True

            # Count valid addresses
            addresses = df[address_column].dropna().astype(str).str.strip()
            addresses = [addr for addr in addresses if addr]
            validation_result["address_count"] = len(addresses)

            if validation_result["address_count"] == 0:
                validation_result["errors"].append("No valid addresses found in the file")
                return validation_result

            validation_result["valid"] = True
            logger.info(f"Input file validation successful: {validation_result['address_count']} addresses found")

        except Exception as e:
            validation_result["errors"].append(f"Validation error: {str(e)}")
            logger.error(f"Input file validation failed: {e}")

        return validation_result

    @staticmethod
    def write_results(
        results: List[Dict[str, Any]],
        output_path: Path,
        include_timestamp: bool = True
    ) -> None:
        """
        Write scrape results to an Excel file.

        Args:
            results: List of dictionaries containing scrape results
            output_path: Path to output Excel file
            include_timestamp: Whether to include timestamp column

        Raises:
            ValueError: If results list is empty
        """
        if not results:
            raise ValueError("No results to write to Excel")

        try:
            logger.info(f"Writing {len(results)} results to {output_path}")

            # Create DataFrame from results
            df = pd.DataFrame(results)

            # Ensure required columns exist
            required_columns = ["address", "owner", "buyer"]
            for col in required_columns:
                if col not in df.columns:
                    df[col] = None

            # Reorder columns for better readability
            column_order = ["address", "owner", "buyer"]

            if include_timestamp and "timestamp" in df.columns:
                column_order.append("timestamp")

            if "status" in df.columns:
                column_order.append("status")

            if "error_message" in df.columns:
                column_order.append("error_message")

            # Add any remaining columns
            for col in df.columns:
                if col not in column_order:
                    column_order.append(col)

            df = df[column_order]

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Write to Excel
            df.to_excel(output_path, index=False, sheet_name="Results")

            logger.info(f"Successfully wrote results to {output_path}")

        except Exception as e:
            logger.error(f"Error writing results to Excel: {e}")
            raise

    @staticmethod
    def create_sample_input(
        output_path: Path,
        sample_addresses: Optional[List[str]] = None,
        address_column: str = "address"
    ) -> None:
        """
        Create a sample input Excel file with example addresses.

        Args:
            output_path: Path where sample file should be created
            sample_addresses: List of sample addresses (optional)
            address_column: Name of the address column
        """
        if sample_addresses is None:
            sample_addresses = [
                "123 Main Street, New York, NY 10001",
                "456 Oak Avenue, Los Angeles, CA 90001",
                "789 Pine Road, Chicago, IL 60601",
                "321 Elm Street, Houston, TX 77001",
                "654 Maple Drive, Phoenix, AZ 85001"
            ]

        try:
            df = pd.DataFrame({address_column: sample_addresses})

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            df.to_excel(output_path, index=False, sheet_name="Sheet1")
            logger.info(f"Sample input file created at {output_path}")

        except Exception as e:
            logger.error(f"Error creating sample input file: {e}")
            raise

    @staticmethod
    def append_to_output(
        result: Dict[str, Any],
        output_path: Path
    ) -> None:
        """
        Append a single result to an existing Excel file (or create new one).
        Useful for incremental updates.

        Args:
            result: Dictionary containing single scrape result
            output_path: Path to output Excel file
        """
        try:
            # Check if file exists
            if output_path.exists():
                # Read existing data
                existing_df = pd.read_excel(output_path)
                # Append new result
                new_df = pd.concat([existing_df, pd.DataFrame([result])], ignore_index=True)
            else:
                # Create new DataFrame
                new_df = pd.DataFrame([result])

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Write to Excel
            new_df.to_excel(output_path, index=False, sheet_name="Results")

            logger.debug(f"Appended result to {output_path}")

        except Exception as e:
            logger.error(f"Error appending to Excel file: {e}")
            raise

    @staticmethod
    def get_file_info(file_path: Path) -> Dict[str, Any]:
        """
        Get information about an Excel file.

        Args:
            file_path: Path to Excel file

        Returns:
            Dictionary with file information
        """
        if not file_path.exists():
            return {"exists": False}

        try:
            excel_file = pd.ExcelFile(file_path)
            info = {
                "exists": True,
                "sheets": excel_file.sheet_names,
                "sheet_count": len(excel_file.sheet_names)
            }

            # Get info for each sheet
            sheet_info = {}
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                sheet_info[sheet_name] = {
                    "rows": len(df),
                    "columns": list(df.columns)
                }

            info["sheet_details"] = sheet_info
            return info

        except Exception as e:
            logger.error(f"Error getting file info: {e}")
            return {"exists": True, "error": str(e)}
