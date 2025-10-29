#!/usr/bin/env python3
"""
Script for single image prediction
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import predict_image

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_single.py <image_path> [model_type]")
        print("Model types: cnn, knn, random_forest, svm, ensemble (default)")
        sys.exit(1)
    
    image_path = sys.argv[1]
    model_type = sys.argv[2] if len(sys.argv) > 2 else 'ensemble'
    
    predict_image(image_path, model_type)