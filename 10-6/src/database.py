"""
Database module for managing SQLite operations.
Handles creation, insertion, and querying of scrape results.
"""

import sqlite3
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages all database operations for the scraper application."""

    def __init__(self, db_path: Path, timeout: int = 30):
        """
        Initialize database manager.

        Args:
            db_path: Path to SQLite database file
            timeout: Database connection timeout in seconds
        """
        self.db_path = db_path
        self.timeout = timeout
        self.connection = None
        self._initialize_database()

    def _initialize_database(self):
        """Create database and tables if they don't exist."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Create scrape_results table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS scrape_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        address TEXT NOT NULL,
                        owner TEXT,
                        buyer TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        raw_html TEXT,
                        status TEXT NOT NULL,
                        error_message TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Create index on address for faster lookups
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_address
                    ON scrape_results(address)
                """)

                # Create index on status for filtering
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_status
                    ON scrape_results(status)
                """)

                conn.commit()
                logger.info(f"Database initialized successfully at {self.db_path}")

        except sqlite3.Error as e:
            logger.error(f"Error initializing database: {e}")
            raise

    def _get_connection(self) -> sqlite3.Connection:
        """
        Get a database connection.

        Returns:
            SQLite connection object
        """
        conn = sqlite3.connect(str(self.db_path), timeout=self.timeout)
        conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        return conn

    def insert_result(
        self,
        address: str,
        owner: Optional[str] = None,
        buyer: Optional[str] = None,
        raw_html: Optional[str] = None,
        status: str = "success",
        error_message: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ) -> int:
        """
        Insert a scrape result into the database.

        Args:
            address: Property address that was searched
            owner: Extracted owner name(s)
            buyer: Extracted buyer name(s)
            raw_html: Full HTML snapshot of the page
            status: Status of the scrape ('success', 'failed', 'no_results')
            error_message: Error message if scrape failed
            timestamp: Custom timestamp (defaults to current time)

        Returns:
            Row ID of inserted record
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                if timestamp is None:
                    timestamp = datetime.now()

                cursor.execute("""
                    INSERT INTO scrape_results
                    (address, owner, buyer, raw_html, status, error_message, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (address, owner, buyer, raw_html, status, error_message, timestamp))

                conn.commit()
                row_id = cursor.lastrowid

                logger.debug(f"Inserted result for address '{address}' with ID {row_id}")
                return row_id

        except sqlite3.Error as e:
            logger.error(f"Error inserting result for address '{address}': {e}")
            raise

    def get_result_by_address(self, address: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve the most recent result for a given address.

        Args:
            address: Property address to search for

        Returns:
            Dictionary containing result data, or None if not found
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT id, address, owner, buyer, timestamp, status, error_message
                    FROM scrape_results
                    WHERE address = ?
                    ORDER BY created_at DESC
                    LIMIT 1
                """, (address,))

                row = cursor.fetchone()

                if row:
                    return dict(row)
                return None

        except sqlite3.Error as e:
            logger.error(f"Error retrieving result for address '{address}': {e}")
            raise

    def is_address_processed(self, address: str, status: str = "success") -> bool:
        """
        Check if an address has already been successfully processed.

        Args:
            address: Property address to check
            status: Status to check for (default: 'success')

        Returns:
            True if address exists with the specified status, False otherwise
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT COUNT(*) as count
                    FROM scrape_results
                    WHERE address = ? AND status = ?
                """, (address, status))

                result = cursor.fetchone()
                return result['count'] > 0

        except sqlite3.Error as e:
            logger.error(f"Error checking if address '{address}' is processed: {e}")
            return False

    def get_all_results(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve all results, optionally filtered by status.

        Args:
            status: Optional status filter ('success', 'failed', 'no_results')

        Returns:
            List of dictionaries containing result data
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                if status:
                    cursor.execute("""
                        SELECT id, address, owner, buyer, timestamp, status, error_message
                        FROM scrape_results
                        WHERE status = ?
                        ORDER BY created_at ASC
                    """, (status,))
                else:
                    cursor.execute("""
                        SELECT id, address, owner, buyer, timestamp, status, error_message
                        FROM scrape_results
                        ORDER BY created_at ASC
                    """)

                rows = cursor.fetchall()
                return [dict(row) for row in rows]

        except sqlite3.Error as e:
            logger.error(f"Error retrieving all results: {e}")
            raise

    def get_statistics(self) -> Dict[str, int]:
        """
        Get statistics about scrape results.

        Returns:
            Dictionary with counts for different statuses
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT
                        COUNT(*) as total,
                        SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as success,
                        SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                        SUM(CASE WHEN status = 'no_results' THEN 1 ELSE 0 END) as no_results
                    FROM scrape_results
                """)

                row = cursor.fetchone()
                return dict(row)

        except sqlite3.Error as e:
            logger.error(f"Error retrieving statistics: {e}")
            raise

    def clear_all_results(self):
        """Delete all results from the database. Use with caution!"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM scrape_results")
                conn.commit()
                logger.warning("All results have been cleared from the database")

        except sqlite3.Error as e:
            logger.error(f"Error clearing results: {e}")
            raise

    def close(self):
        """Close database connection if open."""
        if self.connection:
            self.connection.close()
            logger.debug("Database connection closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
