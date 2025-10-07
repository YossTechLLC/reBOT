# Property Scraper Application

A robust Python web automation and scraping application designed to extract property owner and buyer information from websites. Built for deployment on Google Cloud Platform (VM or Cloud Run).

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Docker Deployment](#docker-deployment)
- [Google Cloud Deployment](#google-cloud-deployment)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Development](#development)

## ✨ Features

- **Automated Web Scraping**: Selenium-based browser automation for JavaScript-heavy websites
- **Excel Integration**: Read addresses from Excel, export results to Excel
- **Database Storage**: SQLite database with full HTML snapshots for re-parsing
- **Configurable**: Environment-based configuration with sensible defaults
- **Rate Limiting**: Built-in delays and retry logic to avoid detection
- **Error Handling**: Comprehensive error handling with detailed logging
- **Resumable**: Skip already-processed addresses automatically
- **Cloud-Ready**: Docker containerization for Cloud Run deployment
- **Headless Mode**: Run without GUI on virtual machines

## 🏗️ Architecture

```
┌─────────────────┐
│  Excel Input    │
│ (addresses.xlsx)│
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│   Main Processor    │
│  - Read addresses   │
│  - Validate input   │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Property Scraper   │
│  - Search website   │
│  - Extract data     │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐      ┌──────────────────┐
│  Database Manager   │◄────►│  SQLite Database │
│  - Store results    │      │  (scrape.db)     │
└────────┬────────────┘      └──────────────────┘
         │
         ▼
┌─────────────────────┐
│   Excel Handler     │
│ - Export results    │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Excel Output      │
│  (results.xlsx)     │
└─────────────────────┘
```

## 📦 Prerequisites

### Local Development
- Python 3.11+
- Google Chrome (latest stable)
- ChromeDriver (matching Chrome version)
- pip (Python package manager)

### Cloud Deployment
- Docker
- Google Cloud SDK (for Cloud Run/VM deployment)
- Google Cloud account with appropriate permissions

## 🚀 Installation

### 1. Clone and Navigate

```bash
cd reBOT/10-6
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Chrome and ChromeDriver

#### On Debian/Ubuntu (Bookworm):
```bash
# Install Chrome
wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | sudo apt-key add -
echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" | sudo tee /etc/apt/sources.list.d/google-chrome.list
sudo apt-get update
sudo apt-get install -y google-chrome-stable

# Install ChromeDriver
CHROMEDRIVER_VERSION=$(curl -sS chromedriver.storage.googleapis.com/LATEST_RELEASE)
wget "https://chromedriver.storage.googleapis.com/${CHROMEDRIVER_VERSION}/chromedriver_linux64.zip"
unzip chromedriver_linux64.zip
sudo mv chromedriver /usr/local/bin/
sudo chmod +x /usr/local/bin/chromedriver
```

#### On macOS:
```bash
brew install --cask google-chrome
brew install chromedriver
```

## ⚙️ Configuration

### 1. Create Environment File

```bash
cp .env.example .env
```

### 2. Configure Settings

Edit `.env` with your specific values:

```bash
# REQUIRED: Update target website URL
TARGET_URL=https://your-target-website.com/search

# CSS Selectors (inspect target website to find these)
SEARCH_INPUT_SELECTOR=input[name='search']
SEARCH_BUTTON_SELECTOR=button[type='submit']

# Optional: Adjust rate limiting
MIN_DELAY_BETWEEN_REQUESTS=2.0
MAX_DELAY_BETWEEN_REQUESTS=5.0

# Headless mode (set to false for debugging)
HEADLESS_MODE=true
```

### 3. Finding CSS Selectors

1. Open target website in Chrome
2. Right-click on search input → Inspect
3. Copy the CSS selector from DevTools
4. Update `.env` with correct selectors

### 4. Customize Extraction Patterns

For advanced extraction, edit `config/settings.py`:

```python
OWNER_PATTERNS = [
    r"owner[:\s]+([A-Za-z\s\.,]+)",
    r"property owner[:\s]+([A-Za-z\s\.,]+)",
    # Add your custom patterns
]
```

## 📊 Usage

### 1. Prepare Input File

Place your Excel file with addresses at `data/input/input_addresses.xlsx`:

| address |
|---------|
| 123 Main Street, New York, NY 10001 |
| 456 Oak Avenue, Los Angeles, CA 90001 |

**Or** run the application once to generate a sample template.

### 2. Run the Application

```bash
python main.py
```

### 3. Monitor Progress

The application will:
- ✅ Validate input file
- 🔍 Process each address sequentially
- 💾 Save results to database
- 📤 Export to Excel
- 📊 Display summary statistics

### 4. Check Output

- **Excel Results**: `data/output/output_results.xlsx`
- **Database**: `data/database/scrape.db`
- **Logs**: `logs/scraper.log`

## 🐳 Docker Deployment

### Build Docker Image

```bash
docker build -t property-scraper .
```

### Run Locally with Docker

```bash
docker run -v $(pwd)/data:/app/data -v $(pwd)/logs:/app/logs property-scraper
```

### Push to Container Registry

```bash
# Tag for Google Container Registry
docker tag property-scraper gcr.io/YOUR_PROJECT_ID/property-scraper

# Push to GCR
docker push gcr.io/YOUR_PROJECT_ID/property-scraper
```

## ☁️ Google Cloud Deployment

### Option 1: Cloud Run

```bash
# Deploy to Cloud Run
gcloud run deploy property-scraper \
  --image gcr.io/YOUR_PROJECT_ID/property-scraper \
  --platform managed \
  --region us-central1 \
  --timeout 3600 \
  --memory 2Gi \
  --set-env-vars TARGET_URL=https://example.com

# Trigger via HTTP
curl https://property-scraper-xxx.run.app
```

### Option 2: Compute Engine VM (Debian Bookworm)

```bash
# Create VM instance
gcloud compute instances create scraper-vm \
  --image-family debian-12 \
  --image-project debian-cloud \
  --machine-type e2-medium \
  --zone us-central1-a

# SSH into VM
gcloud compute ssh scraper-vm

# Install dependencies and run (follow Installation steps above)
```

### Option 3: Cloud Functions (for scheduled runs)

```bash
# Deploy as Cloud Function
gcloud functions deploy scrape-properties \
  --runtime python311 \
  --trigger-http \
  --entry-point main \
  --timeout 540s
```

## 📁 Project Structure

```
reBOT/10-6/
├── config/                 # Configuration module
│   ├── __init__.py
│   └── settings.py        # Centralized settings
├── data/                  # Data directory
│   ├── input/            # Input Excel files
│   ├── output/           # Output Excel files
│   └── database/         # SQLite database
├── src/                   # Source code
│   ├── __init__.py
│   ├── database.py       # Database operations
│   ├── excel_handler.py  # Excel read/write
│   ├── scraper.py        # Web scraping logic
│   └── utils.py          # Utility functions
├── logs/                  # Application logs
├── tests/                 # Unit tests
│   ├── __init__.py
│   └── test_scraper.py
├── main.py               # Entry point
├── requirements.txt      # Python dependencies
├── Dockerfile           # Container definition
├── .env.example         # Environment template
└── README.md           # This file
```

## 🔧 Troubleshooting

### Issue: ChromeDriver version mismatch

```bash
# Check Chrome version
google-chrome --version

# Install matching ChromeDriver
CHROME_VERSION=$(google-chrome --version | awk '{print $3}' | cut -d. -f1)
wget "https://chromedriver.storage.googleapis.com/LATEST_RELEASE_${CHROME_VERSION}"
```

### Issue: Timeout errors

Increase timeouts in `.env`:
```bash
PAGE_LOAD_TIMEOUT=60
ELEMENT_WAIT_TIMEOUT=20
```

### Issue: Selectors not found

1. Verify selectors in browser DevTools
2. Set `HEADLESS_MODE=false` to debug visually
3. Check if website requires authentication

### Issue: Rate limiting / IP blocking

Increase delays:
```bash
MIN_DELAY_BETWEEN_REQUESTS=5.0
MAX_DELAY_BETWEEN_REQUESTS=10.0
```

### Issue: Memory errors in Cloud Run

Increase memory allocation:
```bash
gcloud run deploy property-scraper --memory 4Gi
```

## 🧪 Development

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

### Code Style

```bash
# Install formatters
pip install black flake8

# Format code
black .

# Lint
flake8 src/ tests/
```

### Adding New Features

1. Create feature branch
2. Implement in appropriate module (`src/`)
3. Add tests to `tests/`
4. Update configuration if needed
5. Document in README

## 📝 Database Schema

```sql
CREATE TABLE scrape_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    address TEXT NOT NULL,
    owner TEXT,
    buyer TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    raw_html TEXT,
    status TEXT NOT NULL,  -- 'success', 'failed', 'no_results'
    error_message TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

## 🔒 Security Considerations

- **Credentials**: Never commit `.env` files with credentials
- **Rate Limiting**: Respect website terms of service
- **User Agents**: Use legitimate user agent strings
- **Data Privacy**: Handle extracted data according to privacy laws
- **Container Security**: Run containers as non-root user (implemented)

## 📄 License

This project is proprietary. All rights reserved.

## 🤝 Support

For issues or questions:
1. Check logs at `logs/scraper.log`
2. Review error messages in console output
3. Verify configuration in `.env`
4. Test selectors manually in browser

## 🔄 Workflow Summary

1. **Prepare** → Create/update `input_addresses.xlsx`
2. **Configure** → Update `.env` with target website settings
3. **Run** → Execute `python main.py`
4. **Verify** → Check `output_results.xlsx` and logs
5. **Re-parse** → Query `scrape.db` for raw HTML if needed

---

**Built with ❤️ for automated property data extraction**
