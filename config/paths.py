from pathlib import Path

class Paths:
    def __init__(self, base_dir="."):
        self.BASE_DIR = Path(base_dir)
        self.DATA_DIR = self.BASE_DIR / "data"
        self.MODELS_DIR = self.BASE_DIR / "saved_models"
        self.LOGS_DIR = self.BASE_DIR / "logs"

        self.MODELS_DIR.mkdir(exist_ok=True)
        self.LOGS_DIR.mkdir(exist_ok=True)
        
    @property
    def CNN_MODEL_PATH(self):
        return self.MODELS_DIR / "cnn_model.h5"
    
    @property
    def KNN_MODEL_PATH(self):
        return self.MODELS_DIR / "knn_model.joblib"
    
    @property
    def RF_MODEL_PATH(self):
        return self.MODELS_DIR / "random_forest_model.pkl"
    
    @property
    def SVM_MODEL_PATH(self):
        return self.MODELS_DIR / "svm_model.joblib"