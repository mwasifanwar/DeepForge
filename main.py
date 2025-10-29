#!/usr/bin/env python3
"""
DeepForge: Advanced Multi-Model Deepfake Detection Framework
Main entry point for training and inference
"""

import argparse
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.paths import Paths
from config.model_config import ModelConfig
from data.data_loader import DataLoader
from models.cnn_model import CNNModel
from models.knn_model import KNNModel
from models.random_forest_model import RandomForestModel
from models.svm_model import SVMModel
from training.trainer import ModelTrainer
from inference.predictor import DeepFakePredictor
from utils.logger import setup_logger

logger = setup_logger()

def train_all_models(data_path, hyperparameter_tune=False):
    """Train all models with the provided dataset"""
    paths = Paths()
    config = ModelConfig()
    data_loader = DataLoader(config)
    trainer = ModelTrainer(config, paths)
    
    logger.info("Starting comprehensive model training...")
    logger.info(f"Dataset path: {data_path}")
    
    try:
        # Load data for traditional ML models
        logger.info("Loading and preprocessing data for ML models...")
        X_train, X_test, y_train, y_test = data_loader.load_images_for_ml(
            data_path, test_size=0.2, flatten=True
        )
        
        # Train CNN model
        logger.info("=" * 50)
        logger.info("TRAINING CNN MODEL")
        logger.info("=" * 50)
        
        train_gen, val_gen, test_gen = data_loader.create_tf_data_generators(
            data_path, augmentation=True
        )
        
        cnn_model = CNNModel(config)
        cnn_model.build_model()
        cnn_model.summary()
        
        # Calculate class weights if imbalanced
        class_weight = data_loader.get_class_weights(y_train)
        logger.info(f"Class weights: {class_weight}")
        
        trainer.train_cnn(cnn_model, train_gen, val_gen, paths.CNN_MODEL_PATH, class_weight)
        
        # Evaluate CNN
        if test_gen:
            cnn_metrics = trainer.evaluate_model(cnn_model, test_gen, test_gen.labels, model_type='cnn')
        
        # Train traditional ML models
        ml_models = {
            'knn': KNNModel(config),
            'random_forest': RandomForestModel(config),
            'svm': SVMModel(config)
        }
        
        for name, model in ml_models.items():
            logger.info("=" * 50)
            logger.info(f"TRAINING {name.upper()} MODEL")
            logger.info("=" * 50)
            
            model.build_model()
            trainer.train_ml_model(
                model, X_train, y_train, 
                getattr(paths, f"{name.upper()}_MODEL_PATH"),
                hyperparameter_tune=hyperparameter_tune
            )
            
            # Evaluate ML model
            ml_metrics = trainer.evaluate_model(model, X_test, y_test, model_type='ml')
        
        logger.info("All models trained and evaluated successfully!")
        
        # Compare all models
        logger.info("=" * 50)
        logger.info("MODEL COMPARISON")
        logger.info("=" * 50)
        
        all_models = {'cnn': cnn_model}
        all_models.update(ml_models)
        
        comparison_df = trainer.compare_models(all_models, X_test, y_test)
        logger.info("\nModel Comparison Results:")
        logger.info(f"\n{comparison_df}")
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise

def predict_image(image_path, model_type='ensemble'):
    """Predict a single image using the trained models"""
    paths = Paths()
    config = ModelConfig()
    
    predictor = DeepFakePredictor(config, paths)
    
    try:
        predictor.load_models()
        results = predictor.predict_single_image(image_path, model_type)
        
        print(f"\n{'='*60}")
        print(f"PREDICTION RESULTS: {Path(image_path).name}")
        print(f"{'='*60}")
        
        for model_name, result in results.items():
            if 'error' not in result:
                print(f"{model_name.upper():<15}: {result['prediction']:<6} "
                      f"(Confidence: {result['confidence']:.3f})")
            else:
                print(f"{model_name.upper():<15}: ERROR - {result['error']}")
        
        print(f"{'='*60}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise

def batch_predict(image_dir, model_type='ensemble', output_file=None):
    """Predict multiple images in a directory"""
    paths = Paths()
    config = ModelConfig()
    
    predictor = DeepFakePredictor(config, paths)
    
    try:
        predictor.load_models()
        results = predictor.batch_predict(image_dir, model_type, output_file)
        
        logger.info(f"Batch prediction completed. Processed {len(results)} images.")
        
        if output_file:
            logger.info(f"Results saved to: {output_file}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error during batch prediction: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(
        description="DeepForge: Advanced Multi-Model Deepfake Detection Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all models
  python main.py --mode train --data_path /path/to/dataset
  
  # Train with hyperparameter tuning
  python main.py --mode train --data_path /path/to/dataset --hyperparameter_tune
  
  # Predict single image with ensemble
  python main.py --mode predict --image_path /path/to/image.jpg
  
  # Predict with specific model
  python main.py --mode predict --image_path /path/to/image.jpg --model_type cnn
  
  # Batch predict directory
  python main.py --mode batch_predict --image_dir /path/to/images --output_file results.json
        """
    )
    
    parser.add_argument('--mode', choices=['train', 'predict', 'batch_predict'], required=True,
                       help='Operation mode: train, predict, or batch_predict')
    parser.add_argument('--data_path', help='Path to dataset directory for training')
    parser.add_argument('--image_path', help='Path to single image for prediction')
    parser.add_argument('--image_dir', help='Path to image directory for batch prediction')
    parser.add_argument('--model_type', default='ensemble',
                       choices=['cnn', 'knn', 'random_forest', 'svm', 'ensemble'],
                       help='Model type for prediction (default: ensemble)')
    parser.add_argument('--output_file', help='Output file for batch prediction results')
    parser.add_argument('--hyperparameter_tune', action='store_true',
                       help='Perform hyperparameter tuning during training')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'train':
            if not args.data_path:
                print("ERROR: Please provide --data_path for training")
                sys.exit(1)
            train_all_models(args.data_path, args.hyperparameter_tune)
            
        elif args.mode == 'predict':
            if not args.image_path:
                print("ERROR: Please provide --image_path for prediction")
                sys.exit(1)
            predict_image(args.image_path, args.model_type)
            
        elif args.mode == 'batch_predict':
            if not args.image_dir:
                print("ERROR: Please provide --image_dir for batch prediction")
                sys.exit(1)
            batch_predict(args.image_dir, args.model_type, args.output_file)
            
    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()