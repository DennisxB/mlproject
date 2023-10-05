from fastapi import FastAPI, Request, status, Form, Depends
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import uvicorn
from src.pipeline.predict_pipeline import CustomData, PredictionPipeline
from pydantic import BaseModel
import pandas as pd
app = FastAPI()

template = Jinja2Templates(directory="templates")

# class CustomData(BaseModel): 
#     gender: str
#     race_ethnicity: str
#     parental_level_of_education: str
#     lunch: str
#     test_preparation_course: str
#     reading_score: int
#     writing_score: int

#     @classmethod
#     def get_data_as_frame(cls,
#                         gender: str = Form(...),
#                         race_ethnicity: str = Form(...),
#                         parental_level_of_education: str = Form(...),
#                         lunch: str = Form(...),
#                         test_preparation_course: str = Form(...),
#                         reading_score: int = Form(...),
#                         writing_score: int = Form(...)):
        
#         return cls(
#             gender = gender,
#             race_ethnicity = race_ethnicity,
#             parental_level_of_education = parental_level_of_education,
#             lunch = lunch,
#             test_preparation_course = test_preparation_course,
#             reading_score = reading_score,
#             writing_score = writing_score,
#         )

@app.get('/')
def index(req: Request):
    return template.TemplateResponse(
        name="index.html",
        context={"request": req}
    )

@app.get('/predict', response_class=HTMLResponse)
def get_prediction(req: Request):
    return template.TemplateResponse(
        name="home.html",
        context={"request": req}
    )

@app.post('/predict', response_class=HTMLResponse, status_code=status.HTTP_201_CREATED)
def predict_datapoint(req: Request, features: CustomData = Depends(CustomData.get_data_as_frame)):
    
    features_dict = features.dict()
    print(features_dict)

    pred_df = pd.DataFrame(features_dict, index=[0])
    # data = CustomData(gender=features.gender,
    #     race_ethnicity=features.race_ethnicity,
    #     parental_level_of_education=features.parental_level_of_education,
    #     lunch=features.lunch,
    #     test_preparation_course=features.test_preparation_course,
    #     reading_score=features.reading_score,
    #     writing_score=features.writing_score)
    # print(data)
    # pred_df = data.get_data_as_frame()
    print(pred_df)
    #print("Before Prediction")

    predict_pipeline=PredictionPipeline()
    #print("Mid Prediction")
    results=predict_pipeline.predict(pred_df)
    #print("after Prediction")
    # print(results)
    return template.TemplateResponse(
        name="home.html",
        context={"request": req, "results": str(round(results[0], 2))}
    )


if __name__ == "__main__":
    uvicorn.run("app:app", reload=True)