import os
import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self) -> None:
        self.data_transformation_config = DataTransformationConfig()

    # function responsible for data transformation
    def get_data_transformer_object(self):
        try:
            # Define numerical features 
            numerical_features = ["writing_score", "reading_score"]
            
            # Define categorical features
            categorical_features = [
                "gender", 
                "race_ethnicity",
                "parental_level_of_education", 
                "lunch",  
                "test_preparation_course"
            ]

            # Create numerical pipeline 
            num_pipeline = Pipeline(
                steps=[
                ("simple_impute", SimpleImputer(strategy="median")),
                ("standard_scaler", StandardScaler())
                ]
            )

            # Create numerical pipeline 
            cat_pipeline = Pipeline(
                steps=[
                ("simple_impute", SimpleImputer(strategy="most frequent")),
                ("one_hot_encoding", OneHotEncoder()),
                ("standard_scaler", StandardScaler())
                ]
            )

            # Logging the pipeline process
            logging.info(f"Categorical features: {categorical_features}")
            logging.info(f"Numerical features: {numerical_features}")
            
            # Create the preprocesser pipeline
            preprocesser = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_features),
                    ("cat_pipeline", cat_pipeline, categorical_features),
                ]
            )
            
            # Logginh the preprocessor pipeline process
            logging.info(f"Preprocessor pipeline is created.")

            return preprocesser

        
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try: 
            # Setup train and test DataFrame
            train_df = pd.read_csv(train_path)
            test_df = pd.read(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            # Instantiate the preprocessor object
            preprocessor_obj = self.get_data_transformer_object()

            # Setup target feature name            
            target_feature_name = "math_score"

            # Split the independent and dependent variables for the train dataframe
            input_feature_train_df = train_df.drop(columns=[target_feature_name], axis=1),
            target_feature_train_df = train_df[target_feature_name]
            
            # Split the independent and dependent values for the test dataframe
            input_feature_test_df = test_df.drop(columns=[target_feature_name], axis=1),
            target_feature_test_df = test_df[target_feature_name]

            logging.info("Applying the preprocessing object on training and testing dataframe.")

            # Applying the preprocessor to train and validation data
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info("Saved preprocessing object.")

            # Save the pickle object
            save_object(

                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            raise CustomException(e, sys)


        

