# Quick Start Guide

Get up and running with the Property Scraper in 5 minutes!

## 🚀 Quick Setup (Local)

### 1. Install Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install packages
pip install -r requirements.txt
```

### 2. Configure

```bash
# Copy environment template
cp .env.example .env

# Edit .env and set your target URL
nano .env  # or use any text editor
```

**Minimum required configuration:**
```bash
TARGET_URL=https://your-website.com/search
SEARCH_INPUT_SELECTOR=input[name='search']
SEARCH_BUTTON_SELECTOR=button[type='submit']
```

### 3. Prepare Input Data

Create `data/input/input_addresses.xlsx` with:

| address |
|---------|
| Your address 1 |
| Your address 2 |

**OR** run the app once to generate a sample template:

```bash
python main.py
# This will create a sample file if none exists
```

### 4. Run

```bash
python main.py
```

### 5. Check Results

- 📊 Excel output: `data/output/output_results.xlsx`
- 💾 Database: `data/database/scrape.db`
- 📝 Logs: `logs/scraper.log`

---

## 🐳 Quick Start (Docker)

### 1. Build Image

```bash
docker build -t property-scraper .
```

### 2. Run Container

```bash
docker run \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  -e TARGET_URL=https://your-website.com \
  property-scraper
```

---

## 🔍 Finding CSS Selectors

1. Open target website in Chrome
2. Right-click search box → **Inspect**
3. In DevTools, right-click element → **Copy** → **Copy selector**
4. Paste into `.env` file

Example:
```bash
SEARCH_INPUT_SELECTOR=#search-box
SEARCH_BUTTON_SELECTOR=button.search-btn
```

---

## ⚙️ Common Configurations

### Debug Mode (See Browser)
```bash
HEADLESS_MODE=false
LOG_LEVEL=DEBUG
```

### Faster Scraping
```bash
MIN_DELAY_BETWEEN_REQUESTS=1.0
MAX_DELAY_BETWEEN_REQUESTS=2.0
```

### Slower/Safer Scraping
```bash
MIN_DELAY_BETWEEN_REQUESTS=5.0
MAX_DELAY_BETWEEN_REQUESTS=10.0
MAX_RETRIES=5
```

### Follow First Result Link
```bash
FOLLOW_FIRST_RESULT=true
FIRST_RESULT_SELECTOR=a.property-link:first-child
```

---

## 🔧 Troubleshooting

### ChromeDriver not found
```bash
# Linux/Mac
which chromedriver

# Install if missing:
# See README.md Installation section
```

### Selectors not working
1. Set `HEADLESS_MODE=false` to debug
2. Watch browser automation
3. Verify selectors in DevTools

### No results extracted
1. Check regex patterns in `config/settings.py`
2. View raw HTML in database
3. Adjust `OWNER_PATTERNS` and `BUYER_PATTERNS`

---

## 📊 Output Format

**Excel Output (`output_results.xlsx`):**

| address | owner | buyer | timestamp | status |
|---------|-------|-------|-----------|--------|
| 123 Main St | John Doe | Jane Smith | 2024-01-01 10:00:00 | success |

**Database Schema:**
- Full HTML snapshots stored for re-parsing
- Error messages for failed scrapes
- Timestamps for all operations

---

## 🎯 Next Steps

1. ✅ Test with sample data
2. ✅ Configure for your target website
3. ✅ Run on real data
4. ✅ Deploy to cloud (see README.md)

For detailed documentation, see **README.md**
