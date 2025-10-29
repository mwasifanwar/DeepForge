import unittest
import numpy as np
import tempfile
from pathlib import Path
from PIL import Image

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.model_config import ModelConfig
from data.data_loader import DataLoader
from data.preprocessing import preprocess_image, augment_image

class TestData(unittest.TestCase):
    
    def setUp(self):
        self.config = ModelConfig()
        self.data_loader = DataLoader(self.config)
        
        # Create temporary directory with sample images
        self.temp_dir = tempfile.TemporaryDirectory()
        self.create_sample_images()
    
    def create_sample_images(self):
        """Create sample images for testing"""
        base_dir = Path(self.temp_dir.name)
        
        # Create directory structure
        train_dir = base_dir / "Train"
        real_dir = train_dir / "Real"
        fake_dir = train_dir / "Fake"
        
        real_dir.mkdir(parents=True, exist_ok=True)
        fake_dir.mkdir(parents=True, exist_ok=True)
        
        # Create sample images
        for i in range(5):
            # Real images
            img_array = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(real_dir / f"real_{i}.jpg")
            
            # Fake images
            img_array = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(fake_dir / f"fake_{i}.jpg")
    
    def test_preprocess_image(self):
        """Test image preprocessing"""
        sample_image = list(Path(self.temp_dir.name).glob("**/*.jpg"))[0]
        
        # Test with flatten
        processed = preprocess_image(str(sample_image), flatten=True)
        self.assertEqual(processed.shape, (128 * 128 * 3,))
        
        # Test without flatten
        processed = preprocess_image(str(sample_image), flatten=False)
        self.assertEqual(processed.shape, (128, 128, 3))
    
    def test_data_loading(self):
        """Test data loading functionality"""
        data_path = Path(self.temp_dir.name) / "Train"
        
        X, y = self.data_loader.load_images_for_ml(str(data_path), test_size=0, flatten=True)
        
        self.assertEqual(len(X), 10)  # 5 real + 5 fake
        self.assertEqual(len(y), 10)
        self.assertEqual(X.shape[1], 128 * 128 * 3)
    
    def test_data_generators(self):
        """Test TensorFlow data generators"""
        data_path = Path(self.temp_dir.name) / "Train"
        
        train_gen, val_gen, _ = self.data_loader.create_tf_data_generators(
            str(data_path), augmentation=False
        )
        
        # Check if generators are created
        self.assertIsNotNone(train_gen)
        self.assertIsNotNone(val_gen)
        
        # Check batch shapes
        batch_x, batch_y = next(iter(train_gen))
        self.assertEqual(batch_x.shape[1:], (128, 128, 3))
        self.assertEqual(batch_y.shape[0], self.config.BATCH_SIZE)
    
    def tearDown(self):
        self.temp_dir.cleanup()

if __name__ == '__main__':
    unittest.main()