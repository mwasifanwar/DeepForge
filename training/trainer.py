import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import joblib
import pandas as pd
from datetime import datetime
import logging
from utils.logger import setup_logger
from utils.visualization import plot_training_history, plot_confusion_matrix, plot_roc_curve

logger = setup_logger()

class ModelTrainer:
    def __init__(self, config, paths):
        self.config = config
        self.paths = paths
        self.logger = logging.getLogger(__name__)
        
    def train_cnn(self, model, train_generator, val_generator, model_path, class_weight=None):
        """Train CNN model with advanced callbacks"""
        callbacks = [
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
            ),
            ModelCheckpoint(
                model_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            CSVLogger(
                self.paths.LOGS_DIR / 'cnn_training_log.csv',
                append=True
            )
        ]
        
        self.logger.info("Starting CNN model training...")
        self.logger.info(f"Training samples: {train_generator.samples}")
        self.logger.info(f"Validation samples: {val_generator.samples}")
        
        history = model.fit(
            train_generator,
            epochs=self.config.EPOCHS,
            validation_data=val_generator,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1
        )
        
     
        model.save(model_path)
        
        
        plot_training_history(history, save_path=self.paths.RESULTS_DIR / 'cnn_training_history.png')
        
        self.logger.info("CNN model training completed and saved")
        return history
    
    def train_ml_model(self, model, X_train, y_train, model_path, hyperparameter_tune=False):
        """Train traditional ML models"""
        model_name = model.__class__.__name__
        self.logger.info(f"Training {model_name}...")
        
        if hyperparameter_tune and hasattr(model, 'hyperparameter_tune'):
            self.logger.info(f"Performing hyperparameter tuning for {model_name}")
            best_params = model.hyperparameter_tune(X_train, y_train)
            self.logger.info(f"Best parameters for {model_name}: {best_params}")
        else:
            model.train(X_train, y_train)
        
       
        model.save(model_path)
        
        self.logger.info(f"{model_name} training completed and saved to {model_path}")
        return model
    
    def evaluate_model(self, model, X_test, y_test, model_type='ml'):
        """Comprehensive model evaluation"""
        self.logger.info(f"Evaluating {model.name if hasattr(model, 'name') else model.__class__.__name__}...")
        
        if model_type == 'cnn':
            
            loss, accuracy, precision, recall, auc = model.model.evaluate(X_test, verbose=0)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
        else:
           
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            auc = roc_auc_score(y_test, y_pred_proba)
        
       
        report = classification_report(y_test, y_pred, target_names=['Real', 'Fake'])
        cm = confusion_matrix(y_test, y_pred)
        
        
        from sklearn.metrics import f1_score
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        self.logger.info(f"Accuracy: {accuracy:.4f}")
        self.logger.info(f"Precision: {precision:.4f}")
        self.logger.info(f"Recall: {recall:.4f}")
        self.logger.info(f"F1-Score: {f1:.4f}")
        self.logger.info(f"AUC-ROC: {auc:.4f}")
        self.logger.info("Classification Report:")
        self.logger.info(f"\n{report}")
        
       
        model_name = model.name if hasattr(model, 'name') else model.__class__.__name__
        plot_confusion_matrix(cm, classes=['Real', 'Fake'], 
                             save_path=self.paths.RESULTS_DIR / f'{model_name}_confusion_matrix.png')
        
        
        if hasattr(model, 'predict_proba'):
            plot_roc_curve(y_test, y_pred_proba, 
                          save_path=self.paths.RESULTS_DIR / f'{model_name}_roc_curve.png')
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc,
            'classification_report': report,
            'confusion_matrix': cm
        }
    
    def compare_models(self, models, X_test, y_test):
        """Compare performance of multiple models"""
        comparison_results = {}
        
        for name, model in models.items():
            self.logger.info(f"Evaluating {name}...")
            results = self.evaluate_model(model, X_test, y_test)
            comparison_results[name] = results
        
       
        comparison_df = pd.DataFrame({
            name: {
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1_score'],
                'AUC-ROC': results['auc_roc']
            }
            for name, results in comparison_results.items()
        }).T
        
        
        comparison_df.to_csv(self.paths.RESULTS_DIR / 'model_comparison.csv')
        self.logger.info("Model comparison completed and saved")
        
        return comparison_df