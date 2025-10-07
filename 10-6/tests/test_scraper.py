"""
Basic tests for the property scraper application.
Run with: pytest tests/
"""

import unittest
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Settings
from src import DatabaseManager, ExcelHandler
from src.utils import extract_with_regex, clean_text, validate_url


class TestUtils(unittest.TestCase):
    """Test utility functions."""

    def test_extract_with_regex(self):
        """Test regex extraction."""
        text = "Owner: John Doe, Buyer: Jane Smith"
        patterns = [r"Owner[:\s]+([A-Za-z\s]+),"]

        result = extract_with_regex(text, patterns)
        self.assertEqual(result, "John Doe")

    def test_clean_text(self):
        """Test text cleaning."""
        text = "  Multiple   spaces   here  "
        result = clean_text(text)
        self.assertEqual(result, "Multiple spaces here")

    def test_validate_url(self):
        """Test URL validation."""
        self.assertTrue(validate_url("https://example.com"))
        self.assertTrue(validate_url("http://example.com/search"))
        self.assertFalse(validate_url("not a url"))


class TestSettings(unittest.TestCase):
    """Test configuration settings."""

    def test_paths_exist(self):
        """Test that configured paths are valid."""
        self.assertIsInstance(Settings.INPUT_DIR, Path)
        self.assertIsInstance(Settings.OUTPUT_DIR, Path)
        self.assertIsInstance(Settings.DATABASE_DIR, Path)


class TestDatabaseManager(unittest.TestCase):
    """Test database operations."""

    def setUp(self):
        """Set up test database."""
        self.test_db_path = Path(__file__).parent / "test.db"
        self.db = DatabaseManager(self.test_db_path)

    def tearDown(self):
        """Clean up test database."""
        if self.test_db_path.exists():
            self.test_db_path.unlink()

    def test_insert_result(self):
        """Test inserting a result."""
        row_id = self.db.insert_result(
            address="123 Test St",
            owner="Test Owner",
            buyer="Test Buyer",
            status="success"
        )
        self.assertIsInstance(row_id, int)
        self.assertGreater(row_id, 0)

    def test_get_result_by_address(self):
        """Test retrieving result by address."""
        self.db.insert_result(
            address="456 Test Ave",
            owner="Owner Name",
            status="success"
        )

        result = self.db.get_result_by_address("456 Test Ave")
        self.assertIsNotNone(result)
        self.assertEqual(result['address'], "456 Test Ave")
        self.assertEqual(result['owner'], "Owner Name")

    def test_statistics(self):
        """Test statistics generation."""
        self.db.insert_result(address="1 St", status="success")
        self.db.insert_result(address="2 St", status="failed")
        self.db.insert_result(address="3 St", status="no_results")

        stats = self.db.get_statistics()
        self.assertEqual(stats['total'], 3)
        self.assertEqual(stats['success'], 1)
        self.assertEqual(stats['failed'], 1)
        self.assertEqual(stats['no_results'], 1)


if __name__ == '__main__':
    unittest.main()
