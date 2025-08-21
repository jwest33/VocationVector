"""
Simplified configuration for the job crawler
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
JOBS_DIR = DATA_DIR / "jobs"

# Ensure directories exist
JOBS_DIR.mkdir(parents=True, exist_ok=True)

# Crawler settings
CRAWLER_SETTINGS = {
    "headless": os.getenv("PLAYWRIGHT_HEADLESS", "true").lower() == "true",
    "default_limit": int(os.getenv("DEFAULT_JOB_LIMIT", "10")),
    "timeout": 60000,  # 60 seconds
    "wait_between_jobs": 1000,  # 1 second between clicking jobs
}

# Search defaults
SEARCH_DEFAULTS = {
    "location": os.getenv("DEFAULT_LOCATION", ""),
    "limit": int(os.getenv("DEFAULT_JOB_LIMIT", "10"))
}
