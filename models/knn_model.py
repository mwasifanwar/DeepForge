from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import joblib
import numpy as np
from .base_model import BaseModel
import logging

logger = logging.getLogger(__name__)

class KNNModel(BaseModel):
    def __init__(self, config):
        super().__init__(config, name="KNN")
    
    def build_model(self):
        """Build KNN model"""
        config = self.config.KNN_CONFIG
        
        self.model = KNeighborsClassifier(
            n_neighbors=config['n_neighbors'],
            weights=config['weights'],
            algorithm=config['algorithm'],
            n_jobs=-1
        )
        
        self.logger.info("KNN model built successfully")
        return self.model
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the KNN model"""
        if self.model is None:
            self.build_model()
        
        self.logger.info("Starting KNN model training...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        self.logger.info("KNN model training completed")
        
        return self.model
    
    def hyperparameter_tune(self, X_train, y_train):
        """Perform hyperparameter tuning"""
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree']
        }
        
        grid_search = GridSearchCV(
            KNeighborsClassifier(),
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        self.logger.info("Starting KNN hyperparameter tuning...")
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        self.is_trained = True
        
        self.logger.info(f"Best parameters: {grid_search.best_params_}")
        self.logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_params_
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None or not self.is_trained:
            raise ValueError("Model not trained")
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if self.model is None or not self.is_trained:
            raise ValueError("Model not trained")
        
        return self.model.predict_proba(X)[:, 1]
    
    def save(self, filepath):
        """Save the model"""
        if self.model is None or not self.is_trained:
            raise ValueError("Model not trained")
        
        joblib.dump(self.model, filepath)
        self.logger.info(f"KNN model saved to {filepath}")
    
    def load(self, filepath):
        """Load the model"""
        self.model = joblib.load(filepath)
        self.is_trained = True
        self.logger.info(f"KNN model loaded from {filepath}")