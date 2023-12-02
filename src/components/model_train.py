# Import libraries and dependencies
import os 
import sys
#import seaborn as sns

from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge,Lasso
#from catboost import CatBoostRegressor
#from xgboost import XGBRegressor

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_models


from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path  = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):    
        self.model_trainer_config =  ModelTrainerConfig()
    
    def  initiate_model_trainer(self, train_arr, test_arr):
        try: 
            logging.info("Split training and test input data")
            # Split the train and test data
            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )
        
            # List of models to train
            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                #"XGBRegressor": XGBRegressor(), 
                #"CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            # Train model and get results
            model_results = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, 
                                            y_test=y_test, models=models)

            # Get the best model score from dict
            best_model_score = max(sorted(model_results.values()))
            best_model_score

            # Get the best model name from dict
            best_model_name = list(models.keys())[
            list(model_results.values()).index(best_model_score)
            ]

            # Best model
            best_model = models[best_model_name]

            # Get models greater than 60%
            if best_model_score < 0.6:
                raise CustomException("No best model found, r2 score < 0.6")
            logging.info(f"Best model on training and testing : {[best_model_name]}")

            # Save the best model
            save_object(
                filepath=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Make a prediction
            predicted = best_model.predict(X_test)

            # Get the r2_score result of the prediction
            r2_score_results = r2_score(y_test, predicted) 

            # return the results
            return r2_score_results
        
        except Exception as e:
            raise CustomException(e, sys)