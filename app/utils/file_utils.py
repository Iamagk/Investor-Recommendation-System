"""
Utility functions for file operations
"""
import glob
import os
from pathlib import Path
from typing import Optional

def get_latest_analysis_csv(analysis_type: str) -> Optional[str]:
    """
    Get the path to the latest analysis CSV file of the specified type.
    
    Args:
        analysis_type: One of 'stock', 'comprehensive', or 'enhanced'
        
    Returns:
        Path to the latest CSV file or None if not found
    """
    # Define patterns for different analysis types
    patterns = {
        'stock': 'stock_sector_analysis_*.csv',
        'comprehensive': 'comprehensive_sector_analysis_*.csv',
        'enhanced': 'enhanced_sector_scores_*.csv'
    }
    
    if analysis_type not in patterns:
        return None
    
    # Get project root directory
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data"
    
    # Find all matching files
    pattern = patterns[analysis_type]
    files = list(data_dir.glob(pattern))
    
    if not files:
        return None
    
    # Return the most recently modified file
    latest_file = max(files, key=os.path.getmtime)
    return str(latest_file)

def get_all_latest_analysis_csvs() -> dict:
    """
    Get paths to all latest analysis CSV files.
    
    Returns:
        Dictionary with keys 'stock', 'comprehensive', 'enhanced' and file paths as values
    """
    return {
        'stock': get_latest_analysis_csv('stock'),
        'comprehensive': get_latest_analysis_csv('comprehensive'),
        'enhanced': get_latest_analysis_csv('enhanced')
    }

def cleanup_old_csvs(keep_count: int = 2):
    """
    Clean up old CSV files, keeping only the most recent ones.
    
    Args:
        keep_count: Number of recent files to keep for each type
    """
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data"
    
    patterns = [
        'stock_sector_analysis_*.csv',
        'comprehensive_sector_analysis_*.csv',
        'enhanced_sector_scores_*.csv'
    ]
    
    for pattern in patterns:
        files = list(data_dir.glob(pattern))
        if len(files) > keep_count:
            # Sort by modification time, oldest first
            files.sort(key=os.path.getmtime)
            old_files = files[:-keep_count]  # All except the newest keep_count files
            
            for old_file in old_files:
                try:
                    old_file.unlink()
                    print(f"Removed old CSV: {old_file.name}")
                except Exception as e:
                    print(f"Failed to remove {old_file.name}: {e}")
