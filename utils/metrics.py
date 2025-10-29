import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           classification_report, precision_recall_curve)
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """Calculate comprehensive evaluation metrics"""
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
    
    # AUC-ROC if probabilities are available
    if y_pred_proba is not None:
        metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba)
    
    # Additional metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['true_negative'] = tn
    metrics['false_positive'] = fp
    metrics['false_negative'] = fn
    metrics['true_positive'] = tp
    
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # Classification report
    metrics['classification_report'] = classification_report(
        y_true, y_pred, target_names=['Real', 'Fake'], output_dict=True
    )
    
    return metrics

def calculate_confidence_intervals(metrics, y_true, y_pred, n_bootstrap=1000):
    """Calculate bootstrap confidence intervals for metrics"""
    n_samples = len(y_true)
    bootstrap_metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': []
    }
    
    for _ in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_true_bootstrap = y_true[indices]
        y_pred_bootstrap = y_pred[indices]
        
        # Calculate metrics for bootstrap sample
        bootstrap_metrics['accuracy'].append(accuracy_score(y_true_bootstrap, y_pred_bootstrap))
        bootstrap_metrics['precision'].append(precision_score(y_true_bootstrap, y_pred_bootstrap, zero_division=0))
        bootstrap_metrics['recall'].append(recall_score(y_true_bootstrap, y_pred_bootstrap, zero_division=0))
        bootstrap_metrics['f1_score'].append(f1_score(y_true_bootstrap, y_pred_bootstrap, zero_division=0))
    
    # Calculate confidence intervals
    confidence_intervals = {}
    for metric, values in bootstrap_metrics.items():
        sorted_values = np.sort(values)
        lower = sorted_values[int(0.025 * len(sorted_values))]
        upper = sorted_values[int(0.975 * len(sorted_values))]
        confidence_intervals[metric] = (lower, upper)
    
    return confidence_intervals

def plot_metrics(metrics_dict, save_path=None):
    """Plot comparison of metrics across models"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Extract metrics for plotting
    model_names = list(metrics_dict.keys())
    metric_names = ['accuracy', 'precision', 'recall', 'f1_score']
    
    # Create data for plotting
    plot_data = []
    for model_name, metrics in metrics_dict.items():
        for metric in metric_names:
            if metric in metrics:
                plot_data.append({
                    'Model': model_name,
                    'Metric': metric.replace('_', ' ').title(),
                    'Value': metrics[metric]
                })
    
    df = pd.DataFrame(plot_data)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    sns.barplot(data=df, x='Model', y='Value', hue='Metric')
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Metrics plot saved to {save_path}")
    
    plt.show()
    
    return df

def find_optimal_threshold(y_true, y_pred_proba):
    """Find optimal classification threshold using precision-recall curve"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    # Calculate F1-score for each threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    # Find threshold with maximum F1-score
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_f1 = f1_scores[optimal_idx]
    
    logger.info(f"Optimal threshold: {optimal_threshold:.4f} (F1-score: {optimal_f1:.4f})")
    
    return optimal_threshold, optimal_f1