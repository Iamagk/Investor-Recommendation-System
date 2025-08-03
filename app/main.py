# app/main.py

from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import datetime
from contextlib import asynccontextmanager
from app.routes import stocks, gold, mutual_funds, load_analysis, recommend, ml_predict, portfolio
from app.routes import backtest  # Add backtest router
from data.fetch_gold import get_gold_rates
from app.services.gold_service import store_gold_rates

# Import database connection
from app.db import get_postgres_connection, Base, engine
from app.models import analysis_models
from pymongo import MongoClient

# Create database tables
Base.metadata.create_all(bind=engine)

# Import scraping and storage functions
from app.services.stock_service import get_nifty_500_stocks
from app.services.mutual_funds_service import get_mutual_fund_data
from app.scrapers.stock_scraper_db import insert_stock_data
from app.scrapers.mutual_fund_scraper_db import insert_mutual_fund_data

# Import comprehensive scrapers
from app.scrapers.stock_scraper import scrape_all_stocks
from app.scrapers.mutual_funds_scraper import get_all_mutual_funds
from app.scrapers.gold_scraper_db import insert_gold_data

# Global task status tracking
background_task_status = {
    "stocks": {"status": "idle", "last_run": None, "result": None},
    "mutual_funds": {"status": "idle", "last_run": None, "result": None},
    "gold": {"status": "idle", "last_run": None, "result": None}
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown"""
    # Startup
    print("Starting Smart Investment Recommender API...")
    print("Server started without automatic scraping")
    
    yield  # Application runs here
    
    # Shutdown
    print("Shutting down Smart Investment Recommender API...")
    print("Application shutdown complete")

app = FastAPI(
    title="Smart Sector Investment Recommender",
    description="An AI-powered investment recommendation system that analyzes stock sectors, mutual funds, and gold investments to provide personalized investment advice based on market trends and user preferences.",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://127.0.0.1:3000",
        "http://localhost:5173",  # Vite default port
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Core recommendation router - primary functionality
app.include_router(recommend.router, prefix="/recommend", tags=["recommendations"])

# Portfolio optimization router for frontend integration
app.include_router(portfolio.router, prefix="/api/portfolio", tags=["portfolio"])

# Backtest router for comprehensive analysis
app.include_router(backtest.router, prefix="/backtest", tags=["backtest"])

# Additional routers for comprehensive functionality
app.include_router(stocks.router, prefix="/stocks", tags=["stocks"])
app.include_router(mutual_funds.router, prefix="/mutual-funds", tags=["mutual-funds"])
app.include_router(gold.router, prefix="/gold", tags=["gold"])
app.include_router(load_analysis.router, prefix="/analysis", tags=["analysis"])
app.include_router(ml_predict.router, prefix="/ml", tags=["machine-learning"])

@app.get("/")
async def root():
    return {
        "message": "Welcome to Smart Sector Investment Recommender API",
        "version": "2.0.0",
        "features": [
            "ML-powered investment recommendations",
            "Technical analysis integration",
            "Real-time price data",
            "Conversational explanations",
            "Multi-asset portfolio optimization"
        ],
        "endpoints": {
            "documentation": "/docs",
            "openapi": "/openapi.json",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.datetime.now().isoformat(),
        "services": {
            "api": "running",
            "database": "connected",
            "scraping": "manual_only"
        }
    }

# Background task endpoints for manual control
@app.post("/scrape/stocks")
async def scrape_stocks_manual(background_tasks: BackgroundTasks):
    """Manually trigger stock data scraping"""
    if background_task_status["stocks"]["status"] == "running":
        return {"message": "Stock scraping is already in progress", "status": "running"}
    
    background_tasks.add_task(scrape_stocks_background)
    return {"message": "Stock scraping task started", "status": "initiated"}

@app.post("/scrape/mutual-funds")
async def scrape_mutual_funds_manual(background_tasks: BackgroundTasks):
    """Manually trigger mutual fund data scraping"""
    if background_task_status["mutual_funds"]["status"] == "running":
        return {"message": "Mutual fund scraping is already in progress", "status": "running"}
    
    background_tasks.add_task(scrape_mutual_funds_background)
    return {"message": "Mutual fund scraping task started", "status": "initiated"}

@app.post("/scrape/gold")
async def scrape_gold_manual(background_tasks: BackgroundTasks):
    """Manually trigger gold data scraping"""
    if background_task_status["gold"]["status"] == "running":
        return {"message": "Gold scraping is already in progress", "status": "running"}
    
    background_tasks.add_task(scrape_gold_background)
    return {"message": "Gold scraping task started", "status": "initiated"}

@app.post("/scrape/all")
async def scrape_all_manual(background_tasks: BackgroundTasks):
    """Manually trigger comprehensive scraping for all asset types"""
    running_tasks = [k for k, v in background_task_status.items() 
                    if v["status"] == "running"]
    
    if running_tasks:
        return {
            "message": f"Some scraping tasks are already running: {', '.join(running_tasks)}", 
            "status": "partial_running"
        }
    
    background_tasks.add_task(scrape_all_background)
    return {"message": "Comprehensive scraping task started", "status": "initiated"}

@app.get("/scrape/status")
async def get_scraping_status():
    """Get the current status of all scraping tasks"""
    return {
        "status": "active",
        "timestamp": datetime.datetime.now().isoformat(),
        "tasks": background_task_status,
        "note": "Automatic scraping disabled - use manual endpoints or external scheduler"
    }

# Background task functions
async def scrape_stocks_background():
    """Background task for stock scraping"""
    try:
        background_task_status["stocks"]["status"] = "running"
        background_task_status["stocks"]["last_run"] = datetime.datetime.now().isoformat()
        
        result = await scrape_all_stocks()
        
        background_task_status["stocks"]["result"] = result
        background_task_status["stocks"]["status"] = "completed"
        
    except Exception as e:
        background_task_status["stocks"]["result"] = {"status": "error", "message": str(e)}
        background_task_status["stocks"]["status"] = "error"
        print(f"Error in stock scraping background task: {e}")

async def scrape_mutual_funds_background():
    """Background task for mutual fund scraping"""
    try:
        background_task_status["mutual_funds"]["status"] = "running"
        background_task_status["mutual_funds"]["last_run"] = datetime.datetime.now().isoformat()
        
        result = await get_all_mutual_funds()
        
        background_task_status["mutual_funds"]["result"] = result
        background_task_status["mutual_funds"]["status"] = "completed"
        
    except Exception as e:
        background_task_status["mutual_funds"]["result"] = {"status": "error", "message": str(e)}
        background_task_status["mutual_funds"]["status"] = "error"
        print(f"Error in mutual fund scraping background task: {e}")

async def scrape_gold_background():
    """Background task for gold scraping with enhanced error handling"""
    try:
        background_task_status["gold"]["status"] = "running" 
        background_task_status["gold"]["last_run"] = datetime.datetime.now().isoformat()
        
        gold_data = get_gold_rates()
        
        if gold_data is not None and not gold_data.empty:
            # Store in both databases
            postgres_result = store_gold_rates(gold_data)
            mongo_result = insert_gold_data(gold_data)
            
            result = {
                "status": "success",
                "postgres_result": postgres_result,
                "mongo_result": mongo_result,
                "rates_stored": len(gold_data)
            }
        else:
            result = {"status": "error", "message": "No gold data retrieved or empty DataFrame"}
        
        background_task_status["gold"]["result"] = result
        background_task_status["gold"]["status"] = "completed"
        
    except Exception as e:
        background_task_status["gold"]["result"] = {"status": "error", "message": str(e)}
        background_task_status["gold"]["status"] = "error"
        print(f"Error in gold scraping background task: {e}")

async def scrape_all_background():
    """Background task for comprehensive scraping of all asset types"""
    try:
        print("Starting comprehensive scraping task...")
        
        # Run all scraping tasks concurrently
        stocks_task = scrape_stocks_background()
        mfs_task = scrape_mutual_funds_background()
        gold_task = scrape_gold_background()
        
        # Wait for all tasks to complete
        await asyncio.gather(stocks_task, mfs_task, gold_task, return_exceptions=True)
        
        print("Comprehensive scraping task completed!")
        
    except Exception as e:
        print(f"Error in comprehensive scraping background task: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)