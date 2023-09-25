from fastapi import FastAPI, Request, status
from fastapi.templating import Jinja2Templates
import uvicorn
from src.pipeline.predict_pipeline import CustomData, PredictionPipeline
from pydantic import BaseModel
import pandas as pd
app = FastAPI()

template = Jinja2Templates(directory="templates")

class CustomData(BaseModel): 
    gender: str
    race_ethnicity: str
    parental_level_of_education: str
    lunch: str
    test_preparation_course: str
    reading_score: int
    writing_score: int

    def get_data_as_frame(self) -> pd.DataFrame:
        
        # Create a dict for the input variables
        custom_data_input_dict = {
            "gender" : [self.gender],
            "race_ethnicity": [self.race_ethnicity],
           "parental_level_of_education": [self.parental_level_of_education],
            "lunch": [self.lunch],
            "test_preparation_course": [self.test_preparation_course],
            "reading_score": [self.reading_score],
            "writing_score": [self.writing_score]
        }

        return pd.DataFrame(custom_data_input_dict)

@app.get('/')
def index(req: Request):
    return template.TemplateResponse(
        name="index.html",
        context={"request": req}
    )

@app.post('/predict', status_code=status.HTTP_201_CREATED)
def predict_datapoint(features: CustomData):
    
    data = CustomData(gender=features.gender,
        race_ethnicity=features.race_ethnicity,
        parental_level_of_education=features.parental_level_of_education,
        lunch=features.lunch,
        test_preparation_course=features.test_preparation_course,
        reading_score=features.reading_score,
        writing_score=features.writing_score)

    pred_df = data.get_data_as_frame()
    #print(pred_df)
    #print("Before Prediction")

    predict_pipeline=PredictionPipeline()
    #print("Mid Prediction")
    results=predict_pipeline.predict(pred_df)
    #print("after Prediction")
    #print(results)
    return {"Math Score Prediction": results[0]}

if __name__ == "__main__":
    uvicorn.run("app:app", reload=True)