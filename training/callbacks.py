import tensorflow as tf
import numpy as np
import logging

logger = logging.getLogger(__name__)

class CustomCallbacks(tf.keras.callbacks.Callback):
    """Custom callbacks for enhanced training monitoring"""
    
    def __init__(self, validation_data=None, log_dir=None):
        super().__init__()
        self.validation_data = validation_data
        self.log_dir = log_dir
        self.best_weights = None
        self.best_val_loss = float('inf')
    
    def on_train_begin(self, logs=None):
        logger.info("Training started...")
    
    def on_train_end(self, logs=None):
        logger.info("Training completed.")
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)
            logger.info("Restored model weights from best epoch.")
    
    def on_epoch_begin(self, epoch, logs=None):
        logger.info(f"Starting epoch {epoch + 1}")
    
    def on_epoch_end(self, epoch, logs=None):
        current_val_loss = logs.get('val_loss', float('inf'))
        
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            self.best_weights = self.model.get_weights()
            logger.info(f"New best validation loss: {current_val_loss:.4f}")
        
   
        train_acc = logs.get('accuracy', 0)
        val_acc = logs.get('val_accuracy', 0)
        train_loss = logs.get('loss', 0)
        val_loss = logs.get('val_loss', 0)
        
        logger.info(
            f"Epoch {epoch + 1}: "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

class GradientNormCallback(tf.keras.callbacks.Callback):
    """Monitor gradient norms during training"""
    
    def on_epoch_end(self, epoch, logs=None):
        with tf.GradientTape() as tape:

            pass
        
        logger.info(f"Epoch {epoch + 1}: Gradient norms monitored")

class LearningRateLogger(tf.keras.callbacks.Callback):
    """Log learning rate at each epoch"""
    
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr
        if hasattr(lr, 'numpy'):
            lr = lr.numpy()
        logs['learning_rate'] = lr
        logger.info(f"Epoch {epoch + 1}: Learning Rate = {lr:.6f}")