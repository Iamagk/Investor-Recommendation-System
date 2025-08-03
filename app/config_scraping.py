"""
Configuration settings for the Investment Recommender application
"""

# Scraping Configuration
ENABLE_STARTUP_SCRAPING = False  # Set to False to disable scraping on server startup
ENABLE_PERIODIC_SCRAPING = False  # Set to False to disable periodic background scraping

# Scraping Schedule (only used if periodic scraping is enabled)
SCRAPING_INTERVAL_HOURS = 24  # How often to run scraping (in hours)

# Database Configuration
POSTGRES_URL = "postgresql://username:password@localhost/investment_db"
MONGODB_URL = "mongodb://localhost:27017/investment_db"

# API Configuration
API_TITLE = "Smart Sector Investment Recommender"
API_DESCRIPTION = "ML-powered investment recommendation system with technical analysis"
API_VERSION = "2.0.0"

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_TO_FILE = True
LOG_DIRECTORY = "logs"
