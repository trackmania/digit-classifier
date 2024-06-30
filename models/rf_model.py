import numpy as np
from interface import DigitClassificationInterface

class RandomForestModel(DigitClassificationInterface):
    def predict(self, image: np.ndarray) -> int:
        # Flatten the image for the Random Forest model
        if image.shape != (28, 28):
            raise ValueError("Input image must be of shape (28, 28)")
        
        flattened_image = image.flatten()
        
        print("Predicting with Random Forest model")
        return np.random.randint(0, 10)
    
    def train(self, *args, **kwargs):
        raise NotImplementedError("Training functionality is not implemented for RandomForestModel.")