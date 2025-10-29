from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import joblib
import numpy as np
from .base_model import BaseModel
import logging

logger = logging.getLogger(__name__)

class RandomForestModel(BaseModel):
    def __init__(self, config):
        super().__init__(config, name="RandomForest")
    
    def build_model(self):
        """Build Random Forest model"""
        config = self.config.RF_CONFIG
        
        self.model = RandomForestClassifier(
            n_estimators=config['n_estimators'],
            max_depth=config['max_depth'],
            min_samples_split=config['min_samples_split'],
            min_samples_leaf=config['min_samples_leaf'],
            random_state=config['random_state'],
            n_jobs=config['n_jobs'],
            verbose=0
        )
        
        self.logger.info("Random Forest model built successfully")
        return self.model
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the Random Forest model"""
        if self.model is None:
            self.build_model()
        
        self.logger.info("Starting Random Forest model training...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        self.logger.info("Random Forest model training completed")
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            top_features = np.argsort(self.model.feature_importances_)[-10:]
            self.logger.info(f"Top 10 feature indices by importance: {top_features}")
        
        return self.model
    
    def hyperparameter_tune(self, X_train, y_train):
        """Perform hyperparameter tuning"""
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=self.config.RANDOM_STATE),
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        self.logger.info("Starting Random Forest hyperparameter tuning...")
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
    
    def get_feature_importance(self):
        """Get feature importance scores"""
        if self.model is None or not self.is_trained:
            raise ValueError("Model not trained")
        
        return self.model.feature_importances_
    
    def save(self, filepath):
        """Save the model"""
        if self.model is None or not self.is_trained:
            raise ValueError("Model not trained")
        
        joblib.dump(self.model, filepath)
        self.logger.info(f"Random Forest model saved to {filepath}")
    
    def load(self, filepath):
        """Load the model"""
        self.model = joblib.load(filepath)
        self.is_trained = True
        self.logger.info(f"Random Forest model loaded from {filepath}")