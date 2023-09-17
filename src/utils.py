import os 
import sys
import dill

from src.exception import CustomException


def save_object(filepath: str, obj):
    """
    A function to save file objects

    Args: 
        filepath (str): Define file location of object.
        obj: Object to be saved.
    """
    try:
        # get the directory name
        dir_path = os.path.dirname(filepath)
        # Make the directory 
        os.makedirs(dir_path, exist_ok=True)

        # Open directory to save the file
        with open(dir_path, 'wb') as file_obj:
            dill.dumb(obj, file_obj)
    
    except Exception as e:
        raise CustomException(e, sys)