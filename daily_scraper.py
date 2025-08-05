#!/usr/bin/env python3
"""
Standalone Daily Data Scraper for Investment Recommender
This script runs independently of the main server and can be scheduled via cron
"""

import sys
import os
import asyncio
import datetime
import pandas as pd
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import scraping functions
from app.scrapers.stock_scraper import scrape_all_stocks
from app.scrapers.mutual_funds_scraper import get_all_mutual_funds
from app.scrapers.gold_scraper_db import insert_gold_data
from data.fetch_gold import get_gold_rates
from app.services.gold_service import store_gold_rates

# Import database setup
from app.db import get_postgres_connection, Base, engine
from app.models import analysis_models

# Create database tables if they don't exist
Base.metadata.create_all(bind=engine)

class DailyScraper:
    def __init__(self):
        self.timestamp = datetime.datetime.now()
        self.results = {
            "stocks": {"status": "pending", "result": None},
            "mutual_funds": {"status": "pending", "result": None},
            "gold": {"status": "pending", "result": None}
        }
    
    def log(self, message):
        """Log messages with timestamp"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")
        
        # Also write to log file
        log_file = project_root / "logs" / "daily_scraper.log"
        log_file.parent.mkdir(exist_ok=True)
        
        with open(log_file, "a") as f:
            f.write(f"[{timestamp}] {message}\n")
    
    async def scrape_stocks(self):
        """Scrape stock data"""
        try:
            self.log("Starting stock data scraping...")
            self.results["stocks"]["status"] = "running"
            
            result = await scrape_all_stocks()
            
            self.results["stocks"]["result"] = result
            self.results["stocks"]["status"] = "completed"
            self.log(f"Stock scraping completed: {result}")
            
        except Exception as e:
            self.results["stocks"]["status"] = "error"
            self.results["stocks"]["result"] = {"error": str(e)}
            self.log(f"Stock scraping failed: {e}")
    
    async def scrape_mutual_funds(self):
        """Scrape mutual fund data"""
        try:
            self.log("Starting mutual fund data scraping...")
            self.results["mutual_funds"]["status"] = "running"
            
            result = await get_all_mutual_funds()
            
            self.results["mutual_funds"]["result"] = result
            self.results["mutual_funds"]["status"] = "completed"
            self.log(f"Mutual fund scraping completed: {result}")
            
        except Exception as e:
            self.results["mutual_funds"]["status"] = "error"
            self.results["mutual_funds"]["result"] = {"error": str(e)}
            self.log(f"Mutual fund scraping failed: {e}")
    
    async def scrape_gold(self):
        """Scrape gold data with enhanced error handling"""
        try:
            self.log("Starting gold data scraping...")
            self.results["gold"]["status"] = "running"
            
            # Get gold rates using enhanced multi-source approach
            gold_rates = get_gold_rates()
            
            if gold_rates is not None and not gold_rates.empty:
                self.log(f"Successfully fetched {len(gold_rates)} gold records")
                
                # Store in PostgreSQL
                store_result = store_gold_rates(gold_rates)
                
                # Store in MongoDB  
                mongo_result = insert_gold_data(gold_rates)
                
                result = {
                    "postgres_result": store_result,
                    "mongo_result": mongo_result,
                    "rates_fetched": len(gold_rates)
                }
            else:
                result = {"error": "No gold rates fetched"}
            
            self.results["gold"]["result"] = result
            self.results["gold"]["status"] = "completed"
            self.log(f"Gold scraping completed: {result}")
            
        except Exception as e:
            self.results["gold"]["status"] = "error"
            self.results["gold"]["result"] = {"error": str(e)}
            self.log(f"Gold scraping failed: {e}")

    async def regenerate_analysis_data(self):
        """Regenerate analysis data and store directly in database tables"""
        try:
            self.log("Starting database-based analysis data regeneration...")
            
            # Import the new database analysis function
            from app.analysis.db_sector_analysis import DatabaseSectorAnalyzer
            
            # Run the database-based analysis
            analyzer = DatabaseSectorAnalyzer()
            analyzer.run_full_analysis()
            
            self.log("✅ Database analysis regeneration completed successfully!")
            
        except Exception as e:
            self.log(f"❌ Database analysis regeneration failed: {e}")

    async def regenerate_analysis_files(self):
        """DEPRECATED: Use regenerate_analysis_data() instead. 
        Kept for backwards compatibility - generates CSV files"""
        try:
            self.log("Starting analysis CSV file regeneration...")
            
            # Import analysis functions
            from app.analysis.sector_analysis import (
                fetch_data, calculate_sector_performance,
                fetch_all_investment_data, calculate_comprehensive_sector_performance,
                calculate_sector_scores
            )
            
            # Generate current timestamp for file naming
            current_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 1. Generate Stock Sector Analysis CSV
            self.log("Generating stock sector analysis...")
            stock_df = fetch_data()
            if not stock_df.empty:
                stock_summary = calculate_sector_performance(stock_df)
                if not stock_summary.empty:
                    stock_file = f"data/stock_sector_analysis_{current_timestamp}.csv"
                    stock_summary.to_csv(stock_file, index=False)
                    self.log(f"✅ Stock analysis saved to: {stock_file}")
            
            # 2. Generate Comprehensive Sector Analysis CSV
            self.log("Generating comprehensive sector analysis...")
            combined_df = fetch_all_investment_data()
            if not combined_df.empty:
                comprehensive_summary = calculate_comprehensive_sector_performance(combined_df)
                if not comprehensive_summary.empty:
                    comprehensive_file = f"data/comprehensive_sector_analysis_{current_timestamp}.csv"
                    comprehensive_summary.to_csv(comprehensive_file, index=False)
                    self.log(f"✅ Comprehensive analysis saved to: {comprehensive_file}")
            
            # 3. Generate Enhanced Sector Scores CSV
            self.log("Generating enhanced sector scores...")
            try:
                all_scores = calculate_sector_scores()
                if all_scores:
                    scores_df = pd.DataFrame(all_scores)
                    enhanced_file = f"data/enhanced_sector_scores_{current_timestamp}.csv"
                    scores_df.to_csv(enhanced_file, index=False)
                    self.log(f"✅ Enhanced scores saved to: {enhanced_file}")
            except Exception as e:
                self.log(f"⚠️  Enhanced scoring failed: {e}")
            
            # 4. Clean up old CSV files (keep only latest)
            self.cleanup_old_analysis_files()
            
            self.log("✅ Analysis CSV regeneration completed successfully!")
            
        except Exception as e:
            self.log(f"❌ Analysis regeneration failed: {e}")

    def cleanup_old_analysis_files(self):
        """Remove old analysis CSV files, keeping only the latest ones"""
        try:
            import glob
            import os
            
            data_dir = project_root / "data"
            
            # File patterns to clean up
            patterns = [
                'stock_sector_analysis_*.csv',
                'comprehensive_sector_analysis_*.csv', 
                'enhanced_sector_scores_*.csv'
            ]
            
            for pattern in patterns:
                files = list(data_dir.glob(pattern))
                if len(files) > 1:  # Keep only the latest file
                    files.sort(key=os.path.getmtime)
                    old_files = files[:-1]  # All except the newest
                    
                    for old_file in old_files:
                        old_file.unlink()
                        self.log(f"🗑️  Removed old file: {old_file.name}")
            
        except Exception as e:
            self.log(f"⚠️  Cleanup failed: {e}")
    
    async def run_all_scrapers(self):
        """Run all scrapers concurrently"""
        self.log("=" * 60)
        self.log("Starting Daily Data Scraping Session")
        self.log("=" * 60)
        
        start_time = datetime.datetime.now()
        
        # Run all scrapers concurrently
        tasks = [
            self.scrape_stocks(),
            self.scrape_mutual_funds(),
            self.scrape_gold()
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # After all data is scraped, regenerate analysis data in database
        await self.regenerate_analysis_data()
        
        # Optional: Also generate CSV files for backup/compatibility
        # await self.regenerate_analysis_files()
        
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Generate summary report
        self.log("=" * 60)
        self.log("Daily Scraping Session Complete")
        self.log(f"Duration: {duration:.2f} seconds")
        self.log("Results Summary:")
        
        for asset_type, data in self.results.items():
            status = data["status"]
            self.log(f"  {asset_type.title()}: {status.upper()}")
            if status == "error" and data["result"]:
                self.log(f"    Error: {data['result'].get('error', 'Unknown error')}")
        
        self.log("=" * 60)
        
        return self.results

def main():
    """Main function to run the daily scraper"""
    scraper = DailyScraper()
    
    try:
        # Run the scraper
        results = asyncio.run(scraper.run_all_scrapers())
        
        # Check if any scraping failed
        failed_scrapers = [name for name, data in results.items() if data["status"] == "error"]
        
        if failed_scrapers:
            scraper.log(f"Some scrapers failed: {', '.join(failed_scrapers)}")
            sys.exit(1)  # Exit with error code for cron monitoring
        else:
            scraper.log("All scrapers completed successfully!")
            sys.exit(0)  # Exit successfully
            
    except Exception as e:
        scraper.log(f"Critical error in daily scraper: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
