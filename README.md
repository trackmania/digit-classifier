# Digit Classifier
## Overview
The goal of this project is to build a model interface for classifying handwritten digits using different models: CNN, Random Forest, and a Random model.

## Project Structure
```
digit-classifier/
├── models/
│ ├── __init__.py
│ ├── cnn_model.py
│ ├── rf_model.py
│ └── rand_model.py
|
├── __init__.py
├── digit_classifier.py
├── interface.py
├── README.md
├── requirements.txt
└── .gitignore
```
Clone the repository:
```bash
git clone <repository-url>
cd <repository-directory>
```
Create a virtual environment and activate it:
```bash
python -m venv env
source env/bin/activate
```
Install the required packages:
```bash
pip install -r requirements.txt
```
Usage example:
```
import numpy as np
from digit_classifier import DigitClassifier

# Create a random 28x28 image
image = np.random.rand(28, 28)

# Instantiate the DigitClassifier with 'cnn' algorithm
classifier = DigitClassifier(algorithm='cnn')
print(classifier.predict(image))  # Predict using CNN model

# Instantiate the DigitClassifier with 'rf' algorithm
classifier = DigitClassifier(algorithm='rf')
print(classifier.predict(image))  # Predict using Random Forest model

# Instantiate the DigitClassifier with 'rand' algorithm
classifier = DigitClassifier(algorithm='rand')
print(classifier.predict(image))  # Predict using Random model
```
