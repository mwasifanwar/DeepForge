import logging
import sys
from pathlib import Path
import datetime

def setup_logger(name=__name__, log_level=logging.INFO, log_file=None):
    """Setup logger with console and file handlers"""
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger

def get_logger(name=__name__):
    """Get logger instance"""
    return logging.getLogger(name)

class TrainingLogger:
    """Custom logger for training progress"""
    
    def __init__(self, log_dir=None):
        self.log_dir = Path(log_dir) if log_dir else None
        self.setup_training_log()
    
    def setup_training_log(self):
        """Setup training-specific logging"""
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            log_file = self.log_dir / f"training_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            
            self.logger = setup_logger('training', log_file=log_file)
        else:
            self.logger = setup_logger('training')
    
    def log_epoch(self, epoch, train_loss, train_acc, val_loss=None, val_acc=None):
        """Log epoch results"""
        if val_loss is not None:
            self.logger.info(
                f"Epoch {epoch:3d} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            )
        else:
            self.logger.info(
                f"Epoch {epoch:3d} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}"
            )
    
    def log_metrics(self, metrics_dict, stage='validation'):
        """Log evaluation metrics"""
        self.logger.info(f"{stage.upper()} Metrics:")
        for metric, value in metrics_dict.items():
            if isinstance(value, (int, float)):
                self.logger.info(f"  {metric}: {value:.4f}")
            else:
                self.logger.info(f"  {metric}: {value}")