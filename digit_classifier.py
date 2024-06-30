import numpy as np
from models.cnn_model import CNNModel
from models.rf_model import RandomForestModel
from models.rand_model import RandomModel

class DigitClassifier:
    def __init__(self, algorithm: str):
        self.algorithm = algorithm
        if algorithm == 'cnn':
            self.model = CNNModel()
        elif algorithm == 'rf':
            self.model = RandomForestModel()
        elif algorithm == 'rand':
            self.model = RandomModel()
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def predict(self, image: np.ndarray) -> int:
        if image.shape != (28, 28):
            raise ValueError("Input image must be of shape (28, 28)")
        
        return self.model.predict(image)
    
    def train(self, *args, **kwargs):
        self.model.train(*args, **kwargs)