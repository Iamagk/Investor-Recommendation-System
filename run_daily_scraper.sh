#!/bin/bash

# Daily Investment Data Scraper Script
# This script sets up the environment and runs the daily scraper

# Set the project directory
PROJECT_DIR="/Users/allenngeorge/Projects/investment_recommender"

# Change to project directory
cd "$PROJECT_DIR" || exit 1

# Activate virtual environment
source venv/bin/activate || exit 1

# Run the daily scraper
python daily_scraper.py

# Log the completion
echo "Daily scraper completed at $(date)" >> logs/cron.log
