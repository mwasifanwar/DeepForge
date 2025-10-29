import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Set style
plt.style.use('default')
sns.set_palette("husl")

def plot_training_history(history, save_path=None):
    """Plot training history for CNN models"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    # Plot training & validation accuracy
    axes[0].plot(history.history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history.history:
        axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot training & validation loss
    axes[1].plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot precision and recall
    if 'precision' in history.history:
        axes[2].plot(history.history['precision'], label='Training Precision')
        if 'val_precision' in history.history:
            axes[2].plot(history.history['val_precision'], label='Validation Precision')
        axes[2].set_title('Model Precision')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Precision')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    if 'recall' in history.history:
        axes[3].plot(history.history['recall'], label='Training Recall')
        if 'val_recall' in history.history:
            axes[3].plot(history.history['val_recall'], label='Validation Recall')
        axes[3].set_title('Model Recall')
        axes[3].set_xlabel('Epoch')
        axes[3].set_ylabel('Recall')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training history plot saved to {save_path}")
    
    plt.show()

def plot_confusion_matrix(cm, classes, normalize=False, save_path=None):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
    
    plt.show()

def plot_roc_curve(y_true, y_pred_proba, save_path=None):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"ROC curve saved to {save_path}")
    
    plt.show()
    
    return roc_auc

def plot_feature_importance(feature_importances, feature_names=None, top_k=20, save_path=None):
    """Plot feature importance for tree-based models"""
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(len(feature_importances))]
    
    # Create DataFrame for easier plotting
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importances
    })
    
    # Sort and select top features
    importance_df = importance_df.sort_values('importance', ascending=True).tail(top_k)
    
    plt.figure(figsize=(10, 8))
    plt.barh(importance_df['feature'], importance_df['importance'])
    plt.xlabel('Feature Importance')
    plt.title(f'Top {top_k} Most Important Features')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature importance plot saved to {save_path}")
    
    plt.show()
    
    return importance_df

def plot_prediction_distribution(y_pred_proba, y_true=None, save_path=None):
    """Plot distribution of prediction probabilities"""
    plt.figure(figsize=(10, 6))
    
    if y_true is not None:
        real_probs = y_pred_proba[y_true == 0]
        fake_probs = y_pred_proba[y_true == 1]
        
        plt.hist(real_probs, bins=50, alpha=0.7, label='Real Images', color='blue')
        plt.hist(fake_probs, bins=50, alpha=0.7, label='Fake Images', color='red')
    else:
        plt.hist(y_pred_proba, bins=50, alpha=0.7, color='purple')
    
    plt.xlabel('Prediction Probability')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Probabilities')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Prediction distribution plot saved to {save_path}")
    
    plt.show()

def create_model_comparison_plot(comparison_df, save_path=None):
    """Create comprehensive model comparison plot"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    for i, metric in enumerate(metrics):
        if metric in comparison_df.columns:
            comparison_df[metric].plot(kind='bar', ax=axes[i], color=sns.color_palette("husl"))
            axes[i].set_title(f'{metric} Comparison')
            axes[i].set_ylabel(metric)
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for j, v in enumerate(comparison_df[metric]):
                axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Model comparison plot saved to {save_path}")
    
    plt.show()