import numpy as np
from PIL import Image, ImageEnhance
import cv2
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def preprocess_image(image_path, target_size=(128, 128), flatten=True, enhance=False):
    """
    Preprocess image for model input
    
    Args:
        image_path: Path to image file
        target_size: Target size for resizing
        flatten: Whether to flatten the image for ML models
        enhance: Whether to apply image enhancement
    
    Returns:
        Preprocessed image array
    """
    try:

        img = Image.open(image_path)
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        if enhance:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.2)
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.1)
        
        img = img.resize(target_size)
        
        img_array = np.array(img) / 255.0

        if flatten:
            img_array = img_array.flatten()
            
        return img_array
        
    except Exception as e:
        logger.error(f"Error preprocessing image {image_path}: {e}")
        raise

def augment_image(image_array, augmentation_type='all'):
    """
    Apply data augmentation to image
    
    Args:
        image_array: Input image array
        augmentation_type: Type of augmentation ('flip', 'rotate', 'all')
    
    Returns:
        Augmented image array
    """
    if augmentation_type in ['flip', 'all']:
        
        if np.random.random() > 0.5:
            image_array = np.fliplr(image_array)
    
    if augmentation_type in ['rotate', 'all']:
     
        angle = np.random.uniform(-10, 10)
        rows, cols, ch = image_array.shape
        rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        image_array = cv2.warpAffine(image_array, rotation_matrix, (cols, rows))
    
    if augmentation_type in ['brightness', 'all']:
     
        brightness_factor = np.random.uniform(0.8, 1.2)
        image_array = np.clip(image_array * brightness_factor, 0, 1)
    
    return image_array

def extract_handcrafted_features(image_path, target_size=(128, 128)):
    """
    Extract handcrafted features for traditional ML models
    
    Args:
        image_path: Path to image file
        target_size: Target size for resizing
    
    Returns:
        Feature vector
    """
    img = preprocess_image(image_path, target_size, flatten=False, enhance=False)
    
    features = []
    
    for channel in range(3):
        features.append(np.mean(img[:, :, channel]))
        features.append(np.std(img[:, :, channel]))
    
 
    gray_img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    
    features.append(np.mean(sobelx))
    features.append(np.std(sobelx))
    features.append(np.mean(sobely))
    features.append(np.std(sobely))
    
  
    edges = cv2.Canny(gray_img, 50, 150)
    features.append(np.sum(edges > 0) / edges.size)
    
    return np.array(features)