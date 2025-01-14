
# Heart Disease Prediction

This project is a machine learning application for predicting the presence of heart disease based on input features. It uses a neural network classifier (MLPClassifier) implemented with scikit-learn.

## Project Overview

The project includes the following components:
- **Data Preprocessing**: Cleaning and preparing the dataset for training.
- **Model Training**: Training an MLPClassifier with `max_iter=1000` to predict heart disease.
- **Model Evaluation**: Assessing the classifier's performance using relevant metrics.

## Files

- `Heart_Diesease_Prediction.ipynb`: Jupyter Notebook containing the code and analysis for the project.
- `classifier.pkl`: A serialized MLPClassifier trained on the heart disease dataset.

## How to Use

1. Install the dependencies listed in `requirements.txt`.
2. Open the Jupyter Notebook (`Heart_Diesease_Prediction.ipynb`) to explore the code and analysis.
3. Use the provided `classifier.pkl` file to make predictions with the pre-trained model.

### Example

Load and use the classifier:
```python
import pickle

# Load the model
with open('classifier.pkl', 'rb') as f:
    classifier = pickle.load(f)

# Make predictions
predictions = classifier.predict([[feature1, feature2, ..., featureN]])
print(predictions)
```
## Requirements
The Python dependencies are listed in requirements.txt. Install them using:
 ```bash
   pip install -r requirements.txt
 ```

