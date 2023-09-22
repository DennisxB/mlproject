import pandas as pd
from src.utils import load_object
from pydantic import BaseModel

class PredictionPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        # Setup model and preprocessor path
        model_path = 'artifacts\model.pkl'
        preprocessor_path = 'artifacts\preprocessor.pkl'

        # Load model and preprocessor object
        model = load_object(filepath=model_path)
        preprocessor = load_object(filepath=preprocessor_path)

        # Scale the data
        scaled_data = preprocessor.transform(features)
        
        # make inference
        prediction = model.predict(scaled_data)

        return prediction
        

class CustomData(BaseModel):
    def __init__(self, 
            gender: str,
            race_ethnicity: str,
            parental_level_of_education: str,
            lunch: str,
            test_preparation_course: str,
            reading_score: int,
            writing_score: int):
        
        # Define the data attributes
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_frame(self) -> pd.DataFrame:
        
        # Create a dict for the input variables
        custom_data_input_dict = {
            "gender" : [self.gender],
            "race_ethnicity": [self.race_ethnicity],
           "parental_level_of_education": [self.parental_level_of_education],
            "lunch": [self.lunch],
            "test_preparation_course": [self.test_preparation_course],
            "math_score": [self.math_score],
            "reading_score": [self.reading_score],
            "writing_score": [self.writing_score]
        }

        return pd.DataFrame(custom_data_input_dict)
    