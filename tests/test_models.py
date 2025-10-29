import unittest
import numpy as np
import tempfile
import os
from pathlib import Path

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.model_config import ModelConfig
from models.cnn_model import CNNModel
from models.knn_model import KNNModel
from models.random_forest_model import RandomForestModel
from models.svm_model import SVMModel

class TestModels(unittest.TestCase):
    
    def setUp(self):
        self.config = ModelConfig()
        self.sample_data = np.random.random((100, 128 * 128 * 3))
        self.sample_labels = np.random.randint(0, 2, 100)
    
    def test_cnn_model_build(self):
        """Test CNN model building"""
        model = CNNModel(self.config)
        built_model = model.build_model()
        self.assertIsNotNone(built_model)
        self.assertTrue(model.model is not None)
    
    def test_knn_model(self):
        """Test KNN model training and prediction"""
        model = KNNModel(self.config)
        model.build_model()
        
        # Test training
        model.train(self.sample_data, self.sample_labels)
        self.assertTrue(model.is_trained)
        
        # Test prediction
        predictions = model.predict(self.sample_data[:10])
        self.assertEqual(len(predictions), 10)
    
    def test_random_forest_model(self):
        """Test Random Forest model"""
        model = RandomForestModel(self.config)
        model.build_model()
        model.train(self.sample_data, self.sample_labels)
        
        predictions = model.predict(self.sample_data[:10])
        self.assertEqual(len(predictions), 10)
    
    def test_svm_model(self):
        """Test SVM model"""
        model = SVMModel(self.config)
        model.build_model()
        model.train(self.sample_data, self.sample_labels)
        
        predictions = model.predict(self.sample_data[:10])
        self.assertEqual(len(predictions), 10)
    
    def test_model_saving_loading(self):
        """Test model serialization"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test KNN model saving/loading
            model = KNNModel(self.config)
            model.build_model()
            model.train(self.sample_data, self.sample_labels)
            
            model_path = Path(temp_dir) / "test_model.joblib"
            model.save(model_path)
            self.assertTrue(model_path.exists())
            
            # Create new model and load
            new_model = KNNModel(self.config)
            new_model.load(model_path)
            self.assertTrue(new_model.is_trained)

if __name__ == '__main__':
    unittest.main()