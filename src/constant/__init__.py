import os, sys
from datetime import datetime
 
# artifact ->pipeline folder -> timestamp -> output

def get_current_time_stamp():
    return f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

CURRENT_TIME_STAMP= get_current_time_stamp()

ROOT_DIR_KEY = os.getcwd()
DATA_DIR= "Data"
DATA_DIR_KEY = "finalTrain.csv"

# Zomato-Delivery-Time-Prediction\DATA_DIR\DATA_DIR_KEY

ARTIFACT_DIR_KEY = "Aritfact"

# Data ingestion related variable
DATA_INGESTION_KEY = "data_ingestion"
DATA_INGESTION_RAW_DATA_DIR = "raw_data_dir"
DATA_INGESTION__INGESTED_DATA_DIR_KEY= "ingested_dir"
RAW_DATA_DIR_KEY="raw.csv"
TRAIN_DATA_DIR_KEY ="train.csv"
TEST_DATA_DIR_KEY = "test.csv"

# Data transformation related variable
DATA_TRANSFORMATION_ARTIFACT = "data_transformation"
DATA_PREPROCESSED_DIR= "processor"
DATA_TRANSFORMATION_PREPROCESSING_OBJ= "processor.pkl"
DATA_TRANSFORMED_DIR = "transformation"
TRANSFORMED_TRAIN_DIR_KEY = "train.csv"
TRANSFORMED_TEST_DIR_KEY = "test.csv"
#### artifact/ datatransformation/ processor->processor.pkl and transformation -> train.csv and test.csv

#model Training
MODEL_TRAINER_KEY = "model_trainer"
MODEL_OBJECT = "model.pkl"



