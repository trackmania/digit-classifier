import numpy as np
from interface import DigitClassificationInterface

class CNNModel(DigitClassificationInterface):
    def predict(self, image: np.ndarray) -> int:
        print("Predicting with CNN model")
        return np.random.randint(0, 10)
    
    def train(self, *args, **kwargs):
        raise NotImplementedError("Training functionality is not implemented for CNNModel.")