import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from pathlib import Path
from .preprocessing import preprocess_image
from config.model_config import ModelConfig

class DataLoader:
    def __init__(self, config=ModelConfig):
        self.config = config
        
    def create_tf_data_generators(self, train_dir, val_dir, test_dir=None):
        """Create TensorFlow data generators with augmentation"""
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            validation_split=0.2
        )
        
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.config.IMAGE_SIZE,
            batch_size=self.config.BATCH_SIZE,
            class_mode='binary',
            subset='training'
        )
        
        val_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.config.IMAGE_SIZE,
            batch_size=self.config.BATCH_SIZE,
            class_mode='binary',
            subset='validation'
        )
        
        test_generator = None
        if test_dir:
            test_generator = val_datagen.flow_from_directory(
                test_dir,
                target_size=self.config.IMAGE_SIZE,
                batch_size=self.config.BATCH_SIZE,
                class_mode='binary'
            )
            
        return train_generator, val_generator, test_generator
    
    def load_images_for_ml(self, data_dir):
        """Load and preprocess images for traditional ML models"""
        X, y = [], []
        data_path = Path(data_dir)
        
        for class_name in ['Real', 'Fake']:
            class_dir = data_path / class_name
            class_label = 1 if class_name == 'Fake' else 0
            
            for img_path in class_dir.glob('*.*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    features = preprocess_image(str(img_path), self.config.IMAGE_SIZE)
                    X.append(features)
                    y.append(class_label)
        
        return np.array(X), np.array(y)