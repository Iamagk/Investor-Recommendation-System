# Daily Scraping Setup Instructions

## Overview
You now have multiple options to run daily scraping independently of your server startup:

## Option 1: Disable Server-Side Scraping (Recommended)

### Step 1: Update Configuration
The file `app/config_scraping.py` has been created with these settings:

```python
ENABLE_STARTUP_SCRAPING = False  # Disabled - no scraping on server start
ENABLE_PERIODIC_SCRAPING = False  # Disabled - no background scraping
```

### Step 2: Restart Your Server
Your server will now start without any scraping:

```bash
cd /Users/allenngeorge/Projects/investment_recommender
python start_server.py
```

## Option 2: Standalone Daily Scraper

### What's Created:
- `daily_scraper.py` - Standalone scraping script
- `run_daily_scraper.sh` - Shell script wrapper
- `logs/` directory for logging

### Manual Run:
```bash
cd /Users/allenngeorge/Projects/investment_recommender
python daily_scraper.py
```

### Features:
- ✅ Runs independently of server
- ✅ Comprehensive logging
- ✅ Error handling with exit codes
- ✅ Concurrent scraping of all asset types
- ✅ Progress tracking

## Option 3: Automated Daily Scheduling

### A. Using cron (Unix/Linux style):

1. Open terminal and edit crontab:
```bash
crontab -e
```

2. Add this line to run daily at 9 AM:
```bash
0 9 * * * /Users/allenngeorge/Projects/investment_recommender/run_daily_scraper.sh
```

### B. Using macOS LaunchAgent (Recommended for Mac):

1. Copy the launch agent file:
```bash
cp /Users/allenngeorge/Projects/investment_recommender/com.investment.dailyscraper.plist ~/Library/LaunchAgents/
```

2. Load the launch agent:
```bash
launchctl load ~/Library/LaunchAgents/com.investment.dailyscraper.plist
```

3. Start it immediately (optional):
```bash
launchctl start com.investment.dailyscraper
```

4. Check if it's loaded:
```bash
launchctl list | grep investment
```

### C. To unload/disable the automated scraping:
```bash
launchctl unload ~/Library/LaunchAgents/com.investment.dailyscraper.plist
```

## Option 4: Keep Existing Setup but Configure It

If you want to keep scraping in your server but control it:

### Configure Scraping Intervals:
Edit `app/config_scraping.py`:

```python
ENABLE_STARTUP_SCRAPING = False  # No scraping on startup
ENABLE_PERIODIC_SCRAPING = True   # Keep background scraping
SCRAPING_INTERVAL_HOURS = 24      # Run every 24 hours
```

## Monitoring and Logs

### Check Scraping Status:
Visit: `http://localhost:8000/scrape/status`

### View Logs:
```bash
# Daily scraper logs
tail -f logs/daily_scraper.log

# LaunchAgent logs (if using macOS scheduling)
tail -f logs/launchagent.log
tail -f logs/launchagent.error.log
```

### Manual Triggers:
You can still manually trigger scraping via API:
- `POST http://localhost:8000/scrape/all` - All assets
- `POST http://localhost:8000/scrape/stocks` - Stocks only
- `POST http://localhost:8000/scrape/mutual-funds` - Mutual funds only
- `POST http://localhost:8000/scrape/gold` - Gold only

## Recommended Setup

For your use case, I recommend:

1. **Set configuration** in `app/config_scraping.py`:
   ```python
   ENABLE_STARTUP_SCRAPING = False
   ENABLE_PERIODIC_SCRAPING = False
   ```

2. **Use macOS LaunchAgent** for daily automation at 9 AM:
   ```bash
   cp com.investment.dailyscraper.plist ~/Library/LaunchAgents/
   launchctl load ~/Library/LaunchAgents/com.investment.dailyscraper.plist
   ```

3. **Start your server** - it will start instantly without scraping:
   ```bash
   python start_server.py
   ```

This way:
- ✅ Server starts instantly
- ✅ Data is scraped daily at 9 AM automatically  
- ✅ Scraping runs independently of server
- ✅ You have full control and monitoring
- ✅ Logs are maintained for debugging

## Testing

Test the standalone scraper first:
```bash
python daily_scraper.py
```

Then check the logs in `logs/daily_scraper.log` to ensure everything works before setting up automation.
