from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
from .base_model import BaseModel
import logging

logger = logging.getLogger(__name__)

class SVMModel(BaseModel):
    def __init__(self, config):
        super().__init__(config, name="SVM")
        self.scaler = StandardScaler()
    
    def build_model(self):
        """Build SVM model"""
        config = self.config.SVM_CONFIG
        
        self.model = SVC(
            kernel=config['kernel'],
            probability=config['probability'],
            random_state=config['random_state'],
            C=config['C'],
            gamma=config['gamma'],
            verbose=False
        )
        
        self.logger.info("SVM model built successfully")
        return self.model
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the SVM model"""
        if self.model is None:
            self.build_model()
        
        self.logger.info("Starting SVM model training...")
        
        # Scale features for SVM
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        self.logger.info("SVM model training completed")
        
        return self.model
    
    def hyperparameter_tune(self, X_train, y_train):
        """Perform hyperparameter tuning"""
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.1, 0.01],
            'kernel': ['linear', 'rbf', 'poly']
        }
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        grid_search = GridSearchCV(
            SVC(probability=True, random_state=self.config.RANDOM_STATE),
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        self.logger.info("Starting SVM hyperparameter tuning...")
        grid_search.fit(X_train_scaled, y_train)
        
        self.model = grid_search.best_estimator_
        self.is_trained = True
        
        self.logger.info(f"Best parameters: {grid_search.best_params_}")
        self.logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_params_
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None or not self.is_trained:
            raise ValueError("Model not trained")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if self.model is None or not self.is_trained:
            raise ValueError("Model not trained")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def save(self, filepath):
        """Save the model and scaler"""
        if self.model is None or not self.is_trained:
            raise ValueError("Model not trained")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler
        }
        
        joblib.dump(model_data, filepath)
        self.logger.info(f"SVM model saved to {filepath}")
    
    def load(self, filepath):
        """Load the model and scaler"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.is_trained = True
        self.logger.info(f"SVM model loaded from {filepath}")