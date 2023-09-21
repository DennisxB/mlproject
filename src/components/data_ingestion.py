import os 
import sys 
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass # for creating class variables

from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_train import ModelTrainer
from src.components.model_train import ModelTrainerConfig

@dataclass # directly defining class variables
class DataIngestionConfig:
    """
    A class that defines the path to store train, test and raw data
    """
    # Set the path for the train data
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    # Set the path for the test data
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    # Set the path for the raw data
    raw_data_path: str = os.path.join('artifacts', 'data.csv')


class DataIngestion:
    def __init__(self):
        # Instantiate DataIngestionConfig to the class variable
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")

        try:
            # Read the data from csv
            df = pd.read_csv("notebooks/data/stud.csv")                                                   
            logging.info('Read the dataset as dataframe')

            # Create the Artifacts directory
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save the raw(full) data to the required path
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data ingestion is completed")

            return(
                self.ingestion_config.train_data_path, 
                self.ingestion_config.test_data_path
            ) 
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    #data_transformation.initiate_data_transformation(train_data, test_data)
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr=train_arr, test_arr=test_arr))