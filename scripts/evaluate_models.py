#!/usr/bin/env python3
"""
Script to evaluate all trained models on test data
"""

import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.paths import Paths
from config.model_config import ModelConfig
from data.data_loader import DataLoader
from models.cnn_model import CNNModel
from models.knn_model import KNNModel
from models.random_forest_model import RandomForestModel
from models.svm_model import SVMModel
from training.trainer import ModelTrainer
from utils.logger import setup_logger

logger = setup_logger()

def evaluate_all_models(data_path):
    """Evaluate all trained models on test data"""
    paths = Paths()
    config = ModelConfig()
    data_loader = DataLoader(config)
    trainer = ModelTrainer(config, paths)
    
    logger.info("Loading test data...")
    
    # Load test data for ML models
    X_test, y_test = data_loader.load_images_for_ml(data_path, test_size=0, flatten=True)
    
    # Load test data for CNN
    _, _, test_gen = data_loader.create_tf_data_generators(data_path)
    
    # Load all models
    models = {}
    
    try:
        logger.info("Loading CNN model...")
        cnn_model = CNNModel(config)
        cnn_model.load(paths.CNN_MODEL_PATH)
        models['cnn'] = cnn_model
    except Exception as e:
        logger.warning(f"Could not load CNN model: {e}")
    
    try:
        logger.info("Loading KNN model...")
        knn_model = KNNModel(config)
        knn_model.load(paths.KNN_MODEL_PATH)
        models['knn'] = knn_model
    except Exception as e:
        logger.warning(f"Could not load KNN model: {e}")
    
    try:
        logger.info("Loading Random Forest model...")
        rf_model = RandomForestModel(config)
        rf_model.load(paths.RF_MODEL_PATH)
        models['random_forest'] = rf_model
    except Exception as e:
        logger.warning(f"Could not load Random Forest model: {e}")
    
    try:
        logger.info("Loading SVM model...")
        svm_model = SVMModel(config)
        svm_model.load(paths.SVM_MODEL_PATH)
        models['svm'] = svm_model
    except Exception as e:
        logger.warning(f"Could not load SVM model: {e}")
    
    if not models:
        logger.error("No models could be loaded!")
        return
    
    # Evaluate all models
    logger.info("Evaluating models...")
    
    comparison_results = {}
    for name, model in models.items():
        logger.info(f"Evaluating {name}...")
        
        if name == 'cnn':
            results = trainer.evaluate_model(model, test_gen, test_gen.labels, model_type='cnn')
        else:
            results = trainer.evaluate_model(model, X_test, y_test, model_type='ml')
        
        comparison_results[name] = results
    
    # Create comparison summary
    summary_data = []
    for name, results in comparison_results.items():
        summary_data.append({
            'Model': name,
            'Accuracy': results['accuracy'],
            'Precision': results['precision'],
            'Recall': results['recall'],
            'F1-Score': results['f1_score'],
            'AUC-ROC': results.get('auc_roc', 'N/A')
        })
    
    summary_df = pd.DataFrame(summary_data)
    logger.info("\nModel Evaluation Summary:")
    logger.info(f"\n{summary_df.to_string(index=False)}")
    
    # Save summary to file
    summary_file = paths.RESULTS_DIR / "model_evaluation_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    logger.info(f"Summary saved to: {summary_file}")
    
    return summary_df

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python evaluate_models.py <test_data_path>")
        sys.exit(1)
    
    test_data_path = sys.argv[1]
    evaluate_all_models(test_data_path)