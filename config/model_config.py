class ModelConfig:
    # Image configuration
    IMAGE_SIZE = (128, 128)
    BATCH_SIZE = 32
    EPOCHS = 15
    VALIDATION_SPLIT = 0.2
    RANDOM_STATE = 42
    
    # CNN Architecture
    CNN_CONFIG = {
        'filters': [32, 64, 128, 256],
        'kernel_sizes': [(3, 3)] * 4,
        'pool_sizes': [(2, 2)] * 4,
        'dense_units': [512, 256],
        'dropout_rates': [0.25, 0.25, 0.25, 0.5, 0.5],
        'learning_rate': 0.001,
        'activation': 'relu',
        'final_activation': 'sigmoid'
    }
    
    # Traditional ML models configuration
    KNN_CONFIG = {
        'n_neighbors': 5,
        'weights': 'uniform',
        'algorithm': 'auto'
    }
    
    RF_CONFIG = {
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': 42,
        'n_jobs': -1
    }
    
    SVM_CONFIG = {
        'kernel': 'linear',
        'probability': True,
        'random_state': 42,
        'C': 1.0,
        'gamma': 'scale'
    }
    
    # Training configuration
    TRAINING_CONFIG = {
        'early_stopping_patience': 10,
        'reduce_lr_patience': 5,
        'reduce_lr_factor': 0.5,
        'monitor_metric': 'val_loss'
    }