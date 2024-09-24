from src.constant import *
from src.logger import logging
from src.exception import CustomException
import os, sys
from src.config.configuration import *
import os, sys
import pandas as pd
import numpy as np
import pickle
from src.utils import load_model
from sklearn.pipeline import Pipeline

PREDICTION_FOLDER = "batch_prediction"
PREDICTION_CSV = "prediction_csv"
PREDICTION_FILE = "output.csv"
FEATURE_ENG_FOLDER = "feature_eng"


ROOT_DIR = os.getcwd()
BATCH_PREDICTION= os.path.join(ROOT_DIR, PREDICTION_FOLDER,PREDICTION_CSV)
FEATURE_ENG = os.path.join(ROOT_DIR, PREDICTION_FOLDER, FEATURE_ENG_FOLDER)

class batch_prediction:
    def __init__(self, input_file_path, model_file_path, transformer_file_path, feature_engineering_path) -> None:
        self.input_file_path= input_file_path
        self.model_file_path = model_file_path
        self.transformer_file_path= transformer_file_path
        self.feature_engineering_path= feature_engineering_path

    def start_batch_prediction(self):
        try:

            #load the feature engineering pipeline
            with open(self.feature_engineering_path, 'rb') as f:
                feature_pipeline = pickle.load(f)

            #load the data transformation pipeline
            with open(self.transformer_file_path, 'rb') as f:
                processor = pickle.load(f)

            #load the model 
            model = load_model(file_path=self.model_file_path)

            # create a feature eng pipeline

            feature_engineering_pipeline= Pipeline([
                ("feature_engineering", feature_pipeline())
            ])

            df= pd.read_csv(self.input_file_path)

            df.to_csv("df_zomato_delivery_prediction.csv")


            # apply feature engineeing pipeline steps

            df = feature_engineering_pipeline.transform(df)

            df.to_csv("feature_engineering.csv")

            os.makedirs(FEATURE_ENG, exist_ok=True)

            file_path = os.path.join(FEATURE_ENG, "batch_feature_eng.csv")

            df.to_csv(file_path, index=False)

            # drop time taken
            df= df.drop("Time_taken (min)", axis=1)

            df.to_csv("time_taken_droped.csv")

            transform_data = processor.transform(df)

            file_path = os.path.join(FEATURE_ENG, 'processor.csv')

            predictions = model.predict(transform_data)

            df_prediction = pd.DataFrame(predictions, columns=["prediction"])

            BATCH_PREDICTION_PATH = BATCH_PREDICTION

            os.makedirs(BATCH_PREDICTION_PATH, exist_ok=True)
            csv_path = os.path.join(BATCH_PREDICTION_PATH, "output.csv")

            df_prediction.to_csv(csv_path, index=False)

        except Exception as e:
            CustomException(e, sys)


