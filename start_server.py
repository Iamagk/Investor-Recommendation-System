#!/usr/bin/env python3
"""
Simple server startup script for the investment recommender
"""
import subprocess
import sys
import os

def start_server():
    """Start the FastAPI server"""
    project_root = "/Users/allenngeorge/Projects/investment_recommender"
    python_path = f"{project_root}/venv/bin/python"
    
    print("🚀 Starting Investment Recommender Server...")
    print(f"📁 Project: {project_root}")
    print(f"🐍 Python: {python_path}")
    print("🌐 Server will be available at: http://localhost:8000")
    print("📚 API docs will be available at: http://localhost:8000/docs")
    print("\n" + "="*60)
    
    try:
        # Change to project directory
        os.chdir(project_root)
        
        # Start the server
        cmd = [python_path, "-m", "uvicorn", "app.main:app", "--reload", "--port", "8000"]
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        print("\n💡 Try running manually:")
        print("   cd /Users/allenngeorge/Projects/investment_recommender")
        print("   source venv/bin/activate")
        print("   python -m uvicorn app.main:app --reload --port 8000")

if __name__ == "__main__":
    start_server()
