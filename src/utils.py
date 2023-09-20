import os 
import sys
import pickle
import numpy as np
from typing import Dict
from src.exception import CustomException
from sklearn.metrics import r2_score


def save_object(filepath: str, obj):
    """
    A function to save file objects

    Args: 
        filepath (str): Define file location of object.
        obj: Object, the pickle file to be saved.
    """
    try:
        # get the directory name
        dir_path = os.path.dirname(filepath)
        # Make the directory 
        os.makedirs(dir_path, exist_ok=True)

        # Open directory to save the file
        with open(filepath, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
    
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train: np.ndarray, y_train: np.ndarray, 
                    X_test: np.ndarray, y_test: np.ndarray, 
                    models: Dict) -> Dict:

    # Create an empty dictionary to store test results of the models
    results = {}

    # Train and testing loop
    for name, model in models.items():
        model.fit(X_train, y_train) # Train model

        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Evaluate Train and Test dataset
        train_results = r2_score(y_train, y_train_pred)
        test_results = r2_score(y_test, y_test_pred)

        
        print('Model performance for Training set')
        print(f"Model name: {name}")
        print(f"- R2 Score: {train_results}")

        print('----------------------------------')
        print(f"Model name: {name}")
        print('Model performance for Test set')
        print(f"- R2 Score: {test_results}")
        
        # Update results dictionary
        results[name] = test_results
        

        print('='*35)
        print('\n')
    
    return results

def load_object(filepath):
    with open(filepath, 'rb') as file_obj:
       return pickle.load(file_obj)