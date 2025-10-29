import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
from .base_model import BaseModel
import logging

logger = logging.getLogger(__name__)

class CNNModel(BaseModel):
    def __init__(self, config):
        super().__init__(config, name="CNN")
        self.history = None
    
    def build_model(self):
        """Build advanced CNN model with regularization"""
        config = self.config.CNN_CONFIG
        
        model = models.Sequential()
        
        # First Conv Block
        model.add(layers.Conv2D(
            config['filters'][0], 
            config['kernel_sizes'][0], 
            activation=config['activation'],
            padding='same',
            input_shape=(*self.config.IMAGE_SIZE, 3),
            kernel_regularizer=regularizers.l2(0.001)
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(config['pool_sizes'][0]))
        model.add(layers.Dropout(config['dropout_rates'][0]))
        
        # Second Conv Block
        model.add(layers.Conv2D(
            config['filters'][1],
            config['kernel_sizes'][1],
            activation=config['activation'],
            padding='same',
            kernel_regularizer=regularizers.l2(0.001)
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(config['pool_sizes'][1]))
        model.add(layers.Dropout(config['dropout_rates'][1]))
        
        # Third Conv Block
        model.add(layers.Conv2D(
            config['filters'][2],
            config['kernel_sizes'][2],
            activation=config['activation'],
            padding='same',
            kernel_regularizer=regularizers.l2(0.001)
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(config['pool_sizes'][2]))
        model.add(layers.Dropout(config['dropout_rates'][2]))
        
        # Fourth Conv Block
        model.add(layers.Conv2D(
            config['filters'][3],
            config['kernel_sizes'][3],
            activation=config['activation'],
            padding='same',
            kernel_regularizer=regularizers.l2(0.001)
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.GlobalAveragePooling2D())
        model.add(layers.Dropout(config['dropout_rates'][3]))
        
        # Dense Layers
        model.add(layers.Dense(
            config['dense_units'][0],
            activation=config['activation'],
            kernel_regularizer=regularizers.l2(0.001)
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(config['dropout_rates'][4]))
        
        model.add(layers.Dense(
            config['dense_units'][1],
            activation=config['activation'],
            kernel_regularizer=regularizers.l2(0.001)
        ))
        model.add(layers.Dropout(config['dropout_rates'][4]))
        
        # Output Layer
        model.add(layers.Dense(1, activation=config['final_activation']))
        
        # Custom optimizer with learning rate scheduling
        optimizer = optimizers.Adam(learning_rate=config['learning_rate'])
        
        # Compile model
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall', 'auc']
        )
        
        self.model = model
        self.logger.info("CNN model built successfully")
        return model
    
    def train(self, train_generator, val_generator=None, class_weight=None):
        """Train the CNN model"""
        if self.model is None:
            self.build_model()
        
        callbacks = []
        
        if val_generator:
            callbacks.extend([
                EarlyStopping(
                    monitor=self.config.TRAINING_CONFIG['monitor_metric'],
                    patience=self.config.TRAINING_CONFIG['early_stopping_patience'],
                    restore_best_weights=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor=self.config.TRAINING_CONFIG['monitor_metric'],
                    patience=self.config.TRAINING_CONFIG['reduce_lr_patience'],
                    factor=self.config.TRAINING_CONFIG['reduce_lr_factor'],
                    verbose=1
                )
            ])
        
        self.logger.info("Starting CNN model training...")
        
        if val_generator:
            self.history = self.model.fit(
                train_generator,
                epochs=self.config.EPOCHS,
                validation_data=val_generator,
                callbacks=callbacks,
                class_weight=class_weight,
                verbose=1
            )
        else:
            self.history = self.model.fit(
                train_generator,
                epochs=self.config.EPOCHS,
                callbacks=callbacks,
                class_weight=class_weight,
                verbose=1
            )
        
        self.is_trained = True
        self.logger.info("CNN model training completed")
        return self.history
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not built or loaded")
        
        predictions = self.model.predict(X)
        return (predictions > 0.5).astype(int).flatten()
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if self.model is None:
            raise ValueError("Model not built or loaded")
        
        return self.model.predict(X).flatten()
    
    def save(self, filepath):
        """Save the model"""
        if self.model is None:
            raise ValueError("Model not built")
        
        self.model.save(filepath)
        self.logger.info(f"CNN model saved to {filepath}")
    
    def load(self, filepath):
        """Load the model"""
        self.model = tf.keras.models.load_model(filepath)
        self.is_trained = True
        self.logger.info(f"CNN model loaded from {filepath}")