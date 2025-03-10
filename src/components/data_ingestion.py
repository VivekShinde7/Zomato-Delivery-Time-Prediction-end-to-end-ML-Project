from src.constant import *
from src.config.configuration import *
import os, sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    train_data_path:str = TRAIN_FILE_PATH
    test_data_path:str = TEST_FILE_PATH
    raw_data_path:str = RAW_FILE_PATH


class DataIngestion:
    def __init__(self):
        self.data_ingestion_congif = DataIngestionConfig()

    
    def initiate_data_ingestion(self):
        logging.info("="*50)
        logging.info("Initiate Data Ingestion config")
        logging.info("="*50)

        try:
            df = pd.read_csv(DATASET_PATH)

            logging.info(f"Download data {DATASET_PATH}")

            logging.info('Dataset read as pandas Dataframe')

            os.makedirs(os.path.dirname(self.data_ingestion_congif.raw_data_path ), exist_ok=True)

            df.to_csv(self.data_ingestion_congif.raw_data_path, index=False)

            logging.info("train test split")

            train_set, test_set = train_test_split(df, test_size=0.20, random_state=42)

            os.makedirs(os.path.dirname(self.data_ingestion_congif.train_data_path), exist_ok= True)
            train_set.to_csv(self.data_ingestion_congif.train_data_path, header= True, index=False)

            logging.info(f"train data path, {TRAIN_FILE_PATH}")

            os.makedirs(os.path.dirname(self.data_ingestion_congif.test_data_path), exist_ok= True)
            test_set.to_csv(self.data_ingestion_congif.test_data_path, header= True, index=False)

            logging.info(f"test data path, {TEST_FILE_PATH}")

            logging.info("data ingestion complete")

            return(
                self.data_ingestion_congif.train_data_path,
                self.data_ingestion_congif.test_data_path
            )


        except Exception as e:
            logging.info('Exception occured at Data Ingestion stage')
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    # data transformation
    data_transformation = DataTransformation()
    train_arr, test_arr,_ = data_transformation.inititate_data_transformation(train_data, test_data)

    # model training
    model_trainer = ModelTrainer()
    print(model_trainer.inititate_model_training(train_arr, test_arr))






