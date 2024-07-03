import numpy as np
from interface import DigitClassificationInterface

class RandomModel(DigitClassificationInterface):
    def predict(self, image: np.ndarray) -> int:
        # Center crop of the image
        center_crop = image[9:19, 9:19]
        
        print("Predicting with Random model using center crop")
        return np.random.randint(0, 10)
    
    def train(self, *args, **kwargs):
        raise NotImplementedError("Training functionality is not implemented for RandomModel.")