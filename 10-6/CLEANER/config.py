"""
Configuration file for Excel Cleaner utility.
Define test patterns, column names, and settings here.
"""

from pathlib import Path

# ============================================================================
# DIRECTORY CONFIGURATION
# ============================================================================

# Base directory (CLEANER folder)
BASE_DIR = Path(__file__).resolve().parent

# Input directory containing .xlsx files to be cleaned
INPUT_DIR = BASE_DIR / "ToBeCleaned"

# Output directory for cleaned .xlsx files
OUTPUT_DIR = BASE_DIR / "Cleaned"

# Ensure directories exist
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# TEST KEYWORD PATTERNS
# ============================================================================

# Regex patterns to identify test entries (case-insensitive)
# These patterns will match various forms of "test" in addresses
TEST_PATTERNS = [
    r'\btest\b',           # Matches: "test"
    r'\btests\b',          # Matches: "tests"
    r'\btesty\b',          # Matches: "testy"
    r'\btesting\b',        # Matches: "testing"
    r'\btested\b',         # Matches: "tested"
    r'\btester\b',         # Matches: "tester"
    r'\btest\w*\b',        # Matches: test + any word characters (testable, testing, etc.)
    r'\bte+s+t+\b',        # Matches: teeeessstt (multiple letters)
]

# ============================================================================
# ADDRESS COLUMN CONFIGURATION
# ============================================================================

# Possible column names for the address field (case-insensitive matching)
# The script will search for these column names in order
ADDRESS_COLUMN_NAMES = [
    'address',
    'Address',
    'ADDRESS',
    'street_address',
    'Street Address',
    'property_address',
    'Property Address',
    'location',
    'Location',
    'addr',
    'property',
    'full_address',
]

# ============================================================================
# EXCEL PROCESSING SETTINGS
# ============================================================================

# Sheet name to process (None = first sheet)
SHEET_NAME = 0  # 0 means first sheet

# Skip hidden/temporary Excel files (starting with ~$)
SKIP_TEMP_FILES = True

# File extension to process
FILE_EXTENSION = '.xlsx'

# ============================================================================
# OUTPUT SETTINGS
# ============================================================================

# Output column name for the address
OUTPUT_COLUMN_NAME = 'address'

# Overwrite existing files in Cleaned directory
OVERWRITE_EXISTING = True

# Add timestamp to output filenames (if False, uses original name)
ADD_TIMESTAMP = False

# Timestamp format (if ADD_TIMESTAMP is True)
TIMESTAMP_FORMAT = '%Y%m%d_%H%M%S'

# ============================================================================
# LOGGING SETTINGS
# ============================================================================

# Enable verbose logging
VERBOSE = True

# Show progress for each file
SHOW_PROGRESS = True

# Display statistics after processing
SHOW_STATISTICS = True
