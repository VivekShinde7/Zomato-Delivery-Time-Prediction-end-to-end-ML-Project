from src.constant import *
import os, sys

ROOT_DIR = ROOT_DIR_KEY

# data ingestion 

DATASET_PATH = os.path.join(ROOT_DIR, DATA_DIR, DATA_DIR_KEY)

RAW_FILE_PATH = os.path.join(ROOT_DIR, ARTIFACT_DIR_KEY, DATA_INGESTION_KEY, CURRENT_TIME_STAMP, 
                             DATA_INGESTION_RAW_DATA_DIR, RAW_DATA_DIR_KEY)

TRAIN_FILE_PATH= os.path.join(ROOT_DIR, ARTIFACT_DIR_KEY, DATA_INGESTION_KEY, CURRENT_TIME_STAMP,
                              DATA_INGESTION__INGESTED_DATA_DIR_KEY, TRAIN_DATA_DIR_KEY)


TEST_FILE_PATH = os.path.join(ROOT_DIR, ARTIFACT_DIR_KEY, DATA_INGESTION_KEY, CURRENT_TIME_STAMP,
                              DATA_INGESTION__INGESTED_DATA_DIR_KEY, TEST_DATA_DIR_KEY)


# data transformation steps
PREPROCESSING_OBJ_PATH = os.path.join(ROOT_DIR,ARTIFACT_DIR_KEY,
                                      DATA_TRANSFORMATION_ARTIFACT,DATA_PREPROCESSED_DIR,
                                      DATA_TRANSFORMATION_PREPROCESSING_OBJ)


TRANSFORMED_TRAIN_FILE_PATH= os.path.join(ROOT_DIR, ARTIFACT_DIR_KEY,DATA_TRANSFORMATION_ARTIFACT,
                                        DATA_TRANSFORMED_DIR,TRANSFORMED_TRAIN_DIR_KEY)

TRANSFORMED_TEST_FILE_PATH = os.path.join(ROOT_DIR, ARTIFACT_DIR_KEY,DATA_TRANSFORMATION_ARTIFACT,
                                          DATA_TRANSFORMED_DIR,TRANSFORMED_TEST_DIR_KEY) 


FEATURE_ENG_OBJ_PATH = os.path.join(ROOT_DIR,ARTIFACT_DIR_KEY,
                                      DATA_TRANSFORMATION_ARTIFACT,DATA_PREPROCESSED_DIR,
                                      'feature_eng.pkl')


# model training 
MODEL_FILE_PATH = os.path.join(ROOT_DIR, ARTIFACT_DIR_KEY, MODEL_TRAINER_KEY, MODEL_OBJECT)




