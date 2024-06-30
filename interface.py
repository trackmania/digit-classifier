from abc import ABC, abstractmethod
import numpy as np

class DigitClassificationInterface(ABC):
    @abstractmethod
    def predict(self, image: np.ndarray) -> int:
        """Predict the class of the input image."""
        pass
    
    @abstractmethod
    def train(self, *args, **kwargs):
        """Train the model."""
        pass