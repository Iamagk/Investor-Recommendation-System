#!/bin/bash

# Setup script for daily investment data scraper
# This script configures the cron job to run at 9 AM daily

echo "Setting up Daily Investment Data Scraper..."

PROJECT_DIR="/Users/allenngeorge/Projects/investment_recommender"
PYTHON_ENV="$PROJECT_DIR/venv/bin/python"
SCRAPER_SCRIPT="$PROJECT_DIR/daily_scraper.py"

# Check if virtual environment exists
if [ ! -f "$PYTHON_ENV" ]; then
    echo "❌ Virtual environment not found at $PYTHON_ENV"
    exit 1
fi

# Check if scraper script exists
if [ ! -f "$SCRAPER_SCRIPT" ]; then
    echo "❌ Scraper script not found at $SCRAPER_SCRIPT"
    exit 1
fi

# Create cron job entry
CRON_ENTRY="0 9 * * * cd $PROJECT_DIR && $PYTHON_ENV $SCRAPER_SCRIPT >> logs/cron.log 2>&1"

# Add to crontab
echo "Adding cron job to run daily at 9:00 AM..."
(crontab -l 2>/dev/null; echo "$CRON_ENTRY") | crontab -

# Create logs directory if it doesn't exist
mkdir -p "$PROJECT_DIR/logs"

# Create a launchd plist for macOS (alternative to cron)
PLIST_PATH="$HOME/Library/LaunchAgents/com.investment.dailyscraper.plist"

cat > "$PLIST_PATH" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.investment.dailyscraper</string>
    <key>ProgramArguments</key>
    <array>
        <string>$PYTHON_ENV</string>
        <string>$SCRAPER_SCRIPT</string>
    </array>
    <key>WorkingDirectory</key>
    <string>$PROJECT_DIR</string>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>9</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>$PROJECT_DIR/logs/launchagent.log</string>
    <key>StandardErrorPath</key>
    <string>$PROJECT_DIR/logs/launchagent.error.log</string>
    <key>RunAtLoad</key>
    <false/>
</dict>
</plist>
EOF

# Load the launch agent
launchctl load "$PLIST_PATH"

echo "✅ Daily scraper setup complete!"
echo ""
echo "📅 Configured to run daily at 9:00 AM"
echo "📊 Will update:"
echo "   - Stock data (prices, sectors, performance)"
echo "   - Mutual fund NAVs and returns"
echo "   - Gold prices from multiple sources"
echo "   - Sector analysis and recommendations"
echo ""
echo "📝 Logs will be written to:"
echo "   - Cron: $PROJECT_DIR/logs/cron.log"
echo "   - LaunchAgent: $PROJECT_DIR/logs/launchagent.log"
echo ""
echo "🔧 To check cron job status:"
echo "   crontab -l | grep investment"
echo ""
echo "🔧 To check LaunchAgent status:"
echo "   launchctl list | grep com.investment"
echo ""
echo "🚀 To test manually:"
echo "   cd $PROJECT_DIR && python daily_scraper.py"
