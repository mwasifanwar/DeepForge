from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """Abstract base class for all models"""
    
    def __init__(self, config, name=None):
        self.config = config
        self.name = name or self.__class__.__name__
        self.model = None
        self.logger = logging.getLogger(__name__)
        self.is_trained = False
    
    @abstractmethod
    def build_model(self):
        """Build the model architecture"""
        pass
    
    @abstractmethod
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, X):
        """Make predictions"""
        pass
    
    @abstractmethod
    def save(self, filepath):
        """Save the model"""
        pass
    
    @abstractmethod
    def load(self, filepath):
        """Load the model"""
        pass
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        predictions = self.predict(X_test)
        
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        self.logger.info(f"{self.name} Evaluation Metrics:")
        for metric, value in metrics.items():
            self.logger.info(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def summary(self):
        """Print model summary"""
        if self.model:
            if hasattr(self.model, 'summary'):
                return self.model.summary()
            else:
                self.logger.info(f"{self.name} model parameters: {self.model.get_params()}")
        else:
            self.logger.warning("Model not built yet")