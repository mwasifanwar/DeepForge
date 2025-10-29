#!/usr/bin/env python3
"""
Script to train all models with default configuration
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import train_all_models

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train_all.py <data_path>")
        sys.exit(1)
    
    data_path = sys.argv[1]
    train_all_models(data_path)