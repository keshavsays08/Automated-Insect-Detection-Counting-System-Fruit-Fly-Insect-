#!/usr/bin/env python3
"""
Download fruit fly detection dataset from Roboflow
"""

from roboflow import Roboflow
from pathlib import Path

def download_dataset():
    """Download dataset from Roboflow"""
    
    print("Downloading Fruit Fly Detection Dataset...")
    print("This may take a few minutes...\n")
    
    try:
        rf = Roboflow(api_key="ic9y7IwOHICt7zFS52PY")
        project = rf.workspace("keshavs-workspace-sscce").project("fruit-fly-detection-2rfnl")
        dataset = project.version(1).download("yolov8")
        
        print(f"✓ Dataset downloaded successfully!")
        print(f"✓ Location: {dataset.location}")
        print("\nDataset structure:")
        print("  ├── train/")
        print("  ├── val/")
        print("  ├── test/")
        print("  └── data.yaml")
        
    except Exception as e:
        print(f"✗ Error downloading dataset: {e}")
        print("\nManual download:")
        print("1. Visit: https://roboflow.com/workspace/your-workspace/project/fruit-fly-detection")
        print("2. Click 'Download'")
        print("3. Select 'YOLOv8' format")
        print("4. Extract to 'dataset/' folder")

if __name__ == "__main__":
    download_dataset()