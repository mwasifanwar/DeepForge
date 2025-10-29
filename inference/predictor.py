import numpy as np
from PIL import Image
import tensorflow as tf
import joblib
from pathlib import Path
import logging
from utils.logger import setup_logger

logger = setup_logger()

class DeepFakePredictor:
    def __init__(self, config, paths):
        self.config = config
        self.paths = paths
        self.models = {}
        self.logger = logging.getLogger(__name__)
        
    def load_models(self, model_types=None):
        """Load specified trained models"""
        if model_types is None:
            model_types = ['cnn', 'knn', 'random_forest', 'svm']
        
        try:
            for model_type in model_types:
                if model_type == 'cnn':
                    self.models['cnn'] = tf.keras.models.load_model(self.paths.CNN_MODEL_PATH)
                    self.logger.info("CNN model loaded successfully")
                
                elif model_type == 'knn':
                    self.models['knn'] = joblib.load(self.paths.KNN_MODEL_PATH)
                    self.logger.info("KNN model loaded successfully")
                
                elif model_type == 'random_forest':
                    self.models['random_forest'] = joblib.load(self.paths.RF_MODEL_PATH)
                    self.logger.info("Random Forest model loaded successfully")
                
                elif model_type == 'svm':
                    self.models['svm'] = joblib.load(self.paths.SVM_MODEL_PATH)
                    self.logger.info("SVM model loaded successfully")
            
            self.logger.info(f"All specified models loaded: {list(self.models.keys())}")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            raise
    
    def preprocess_image(self, image_path, for_cnn=True):
        """Preprocess image for prediction"""
        try:
            img = Image.open(image_path)
            img = img.resize(self.config.IMAGE_SIZE)
            img_array = np.array(img) / 255.0
            
            if for_cnn:
                return np.expand_dims(img_array, axis=0)
            else:
                return img_array.flatten().reshape(1, -1)
                
        except Exception as e:
            self.logger.error(f"Error preprocessing image {image_path}: {e}")
            raise
    
    def predict_single_image(self, image_path, model_type='ensemble'):
        """Predict single image using specified model(s)"""
        if not self.models:
            self.load_models()
        
        results = {}
        
        if model_type == 'ensemble':
            # Get predictions from all loaded models
            for name, model in self.models.items():
                try:
                    if name == 'cnn':
                        processed_img = self.preprocess_image(image_path, for_cnn=True)
                        pred_proba = model.predict(processed_img, verbose=0)[0][0]
                    else:
                        processed_img = self.preprocess_image(image_path, for_cnn=False)
                        if hasattr(model, 'predict_proba'):
                            pred_proba = model.predict_proba(processed_img)[0][1]
                        else:
                            pred = model.predict(processed_img)[0]
                            pred_proba = 1.0 if pred == 1 else 0.0
                    
                    prediction = 'FAKE' if pred_proba > 0.5 else 'REAL'
                    confidence = pred_proba if prediction == 'FAKE' else 1 - pred_proba
                    
                    results[name] = {
                        'prediction': prediction,
                        'confidence': float(confidence),
                        'probability': float(pred_proba)
                    }
                    
                except Exception as e:
                    self.logger.error(f"Error predicting with {name}: {e}")
                    results[name] = {
                        'prediction': 'ERROR',
                        'confidence': 0.0,
                        'probability': 0.0
                    }
            
            # Ensemble prediction (weighted voting)
            if results:
                fake_votes = sum(1 for r in results.values() if r['prediction'] == 'FAKE')
                total_models = len(results)
                ensemble_confidence = sum(r['confidence'] for r in results.values()) / total_models
                
                ensemble_pred = 'FAKE' if fake_votes >= (total_models / 2) else 'REAL'
                results['ensemble'] = {
                    'prediction': ensemble_pred,
                    'confidence': float(ensemble_confidence),
                    'votes': f"{fake_votes}/{total_models} for FAKE"
                }
            
        else:
            # Single model prediction
            if model_type in self.models:
                model = self.models[model_type]
                try:
                    if model_type == 'cnn':
                        processed_img = self.preprocess_image(image_path, for_cnn=True)
                        pred_proba = model.predict(processed_img, verbose=0)[0][0]
                    else:
                        processed_img = self.preprocess_image(image_path, for_cnn=False)
                        if hasattr(model, 'predict_proba'):
                            pred_proba = model.predict_proba(processed_img)[0][1]
                        else:
                            pred = model.predict(processed_img)[0]
                            pred_proba = 1.0 if pred == 1 else 0.0
                    
                    prediction = 'FAKE' if pred_proba > 0.5 else 'REAL'
                    confidence = pred_proba if prediction == 'FAKE' else 1 - pred_proba
                    
                    results[model_type] = {
                        'prediction': prediction,
                        'confidence': float(confidence),
                        'probability': float(pred_proba)
                    }
                    
                except Exception as e:
                    self.logger.error(f"Error predicting with {model_type}: {e}")
                    results[model_type] = {
                        'prediction': 'ERROR',
                        'confidence': 0.0,
                        'probability': 0.0
                    }
            else:
                raise ValueError(f"Model type {model_type} not available")
        
        return results
    
    def batch_predict(self, image_dir, model_type='ensemble', output_file=None):
        """Predict multiple images in a directory"""
        if not self.models:
            self.load_models()
        
        results = {}
        image_path = Path(image_dir)
        
        if not image_path.exists():
            raise ValueError(f"Directory {image_dir} does not exist")
        
        image_files = list(image_path.glob('*.*'))
        self.logger.info(f"Found {len(image_files)} images in directory")
        
        for img_file in image_files:
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                try:
                    results[str(img_file)] = self.predict_single_image(str(img_file), model_type)
                except Exception as e:
                    self.logger.error(f"Error processing {img_file}: {e}")
                    results[str(img_file)] = {'error': str(e)}
        
        # Save results to file if specified
        if output_file:
            import json
            # Convert numpy types to Python native types for JSON serialization
            def convert_types(obj):
                if isinstance(obj, (np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.int32, np.int64)):
                    return int(obj)
                elif isinstance(obj, dict):
                    return {k: convert_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_types(item) for item in obj]
                else:
                    return obj
            
            serializable_results = convert_types(results)
            
            with open(output_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            self.logger.info(f"Results saved to {output_file}")
        
        return results
    
    def get_model_info(self):
        """Get information about loaded models"""
        info = {}
        for name, model in self.models.items():
            if name == 'cnn':
                info[name] = {
                    'type': 'CNN',
                    'input_shape': model.input_shape,
                    'layers': len(model.layers),
                    'trainable_params': model.count_params()
                }
            else:
                info[name] = {
                    'type': model.__class__.__name__,
                    'parameters': model.get_params() if hasattr(model, 'get_params') else 'N/A'
                }
        return info