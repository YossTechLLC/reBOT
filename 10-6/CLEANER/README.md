# Excel Cleaner

A Python utility for batch cleaning Excel files containing address data. Removes test entries and keeps only the address column.

## 📋 Features

- **Batch Processing**: Automatically processes all .xlsx files in the input directory
- **Smart Column Detection**: Finds address columns using flexible name matching
- **Test Entry Filtering**: Removes rows containing "test", "testy", "testing", and variations
- **Case-Insensitive Matching**: Catches test entries regardless of capitalization
- **Progress Reporting**: Shows detailed statistics for each file processed
- **Configurable**: Easy-to-modify settings in `config.py`

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd CLEANER
pip install -r requirements.txt
```

### 2. Prepare Your Files

Place your Excel files (.xlsx) in the `ToBeCleaned/` directory:

```bash
cp /path/to/your/file.xlsx ToBeCleaned/
```

### 3. Run the Cleaner

```bash
python cleaner.py
```

### 4. Get Your Results

Cleaned files will be saved in the `Cleaned/` directory with only the address column retained.

## 📊 What It Does

### Input File Example:
| address | name | phone | notes |
|---------|------|-------|-------|
| 123 Main St, NY | John | 555-1234 | Customer |
| Test Address | Test | 555-0000 | Testing |
| 456 Oak Ave, CA | Jane | 555-5678 | Customer |
| testing property | Bob | 555-9999 | Test data |

### Output File:
| address |
|---------|
| 123 Main St, NY |
| 456 Oak Ave, CA |

**Changes:**
- ✅ Only address column retained
- ✅ Test entries removed (case-insensitive)
- ✅ Empty rows removed
- ✅ Clean, ready-to-use data

## ⚙️ Configuration

Edit `config.py` to customize behavior:

### Test Patterns

Add or modify patterns to catch different test variations:

```python
TEST_PATTERNS = [
    r'\btest\b',           # Matches: "test"
    r'\btesty\b',          # Matches: "testy"
    r'\btesting\b',        # Matches: "testing"
    r'\btest\w*\b',        # Matches: test, tests, testing, testable, etc.
]
```

### Address Column Names

Add column names your Excel files might use:

```python
ADDRESS_COLUMN_NAMES = [
    'address',
    'street_address',
    'property_address',
    'location',
    # Add your custom names here
]
```

### Output Settings

```python
OVERWRITE_EXISTING = True   # Overwrite files in Cleaned directory
ADD_TIMESTAMP = False       # Add timestamp to output filenames
SHOW_STATISTICS = True      # Display detailed statistics
```

## 📁 Directory Structure

```
CLEANER/
├── ToBeCleaned/           # Put your .xlsx files here
├── Cleaned/               # Cleaned files appear here
├── cleaner.py            # Main script
├── config.py             # Configuration
├── utils.py              # Helper functions
├── requirements.txt      # Dependencies
└── README.md            # This file
```

## 🔍 How It Works

1. **Discovery**: Scans `ToBeCleaned/` for .xlsx files
2. **Column Detection**: Finds the address column (flexible naming)
3. **Filtering**: Removes rows containing test keywords
4. **Cleaning**: Keeps only the address column
5. **Saving**: Writes cleaned file to `Cleaned/` directory

## 📈 Example Output

```
================================================================================
Excel Cleaner - Address Data Cleaning Utility
================================================================================

Scanning directory: /path/to/CLEANER/ToBeCleaned
Found 2 Excel file(s) to process

[1/2] Processing: properties.xlsx
  ✓ Address column found and processed
  Original rows: 150
  Test entries removed: 12
  Empty entries removed: 3
  Final rows: 135
  ✓ Saved to: properties.xlsx

[2/2] Processing: listings.xlsx
  ✓ Address column found and processed
  Original rows: 200
  Test entries removed: 8
  Empty entries removed: 2
  Final rows: 190
  ✓ Saved to: listings.xlsx

================================================================================
PROCESSING SUMMARY
================================================================================

Files Processed: 2
  ✓ Successful: 2

Data Statistics:
  Total original rows: 350
  Test entries removed: 20
  Empty entries removed: 5
  Total final rows: 325
  Retention rate: 92.9%

================================================================================

✓ All files processed successfully!
```

## 🛠️ Advanced Usage

### Custom Test Patterns

To catch specific test variations, add regex patterns to `config.py`:

```python
TEST_PATTERNS = [
    r'\btest\b',           # Basic "test"
    r'\bdemo\b',           # Demo entries
    r'\bsample\b',         # Sample entries
    r'\bexample\b',        # Example entries
    r'\btest\d+\b',        # test1, test2, etc.
]
```

### Processing Specific Sheets

By default, the first sheet is processed. To change:

```python
# In config.py
SHEET_NAME = 'Sheet1'  # Process by name
# or
SHEET_NAME = 1         # Process second sheet (0-indexed)
```

### Adding Timestamps to Outputs

To avoid overwriting existing files:

```python
# In config.py
ADD_TIMESTAMP = True
TIMESTAMP_FORMAT = '%Y%m%d_%H%M%S'
```

Output: `properties_20240101_143022.xlsx`

## 🐛 Troubleshooting

### "Address column not found"

**Cause**: The script couldn't identify the address column.

**Solution**:
1. Check your Excel file column names
2. Add the exact column name to `ADDRESS_COLUMN_NAMES` in `config.py`

### "No Excel files found"

**Cause**: No .xlsx files in `ToBeCleaned/` directory.

**Solution**:
- Ensure files have `.xlsx` extension (not `.xls`)
- Check files aren't hidden or temporary (`~$` prefix)

### "Permission denied" errors

**Cause**: Output file is open in Excel.

**Solution**: Close the file in Excel and run again.

### Test entries not being filtered

**Cause**: Pattern doesn't match your test entries.

**Solution**:
1. Check the exact text in your test entries
2. Add a custom pattern to `TEST_PATTERNS` in `config.py`

## 🧪 Testing

To test the cleaner with sample data:

1. Create a test Excel file with mixed data
2. Include some entries with "test", "Testing", "TESTY"
3. Place in `ToBeCleaned/`
4. Run `python cleaner.py`
5. Verify output in `Cleaned/`

## 📝 Notes

- **Case-Insensitive**: All filtering is case-insensitive (Test = test = TEST)
- **Word Boundaries**: Patterns use word boundaries to avoid false matches
  - Matches: "test address", "Testing 123"
  - Doesn't match: "attest", "contest", "latest"
- **Preserves Data**: Original files in `ToBeCleaned/` are never modified
- **Excel Format**: Only .xlsx files are supported (not .xls or .csv)

## 🔧 Requirements

- Python 3.7+
- pandas >= 2.1.0
- openpyxl >= 3.1.0

## 📄 License

Part of the reBOT/10-6 property scraper project.

---

**Need help?** Check the configuration in `config.py` or review the console output for detailed error messages.
