from src.constants import *
import os, sys

ROOT_DIR = ROOT_DIR_KEY

# Data Ingestion related paths...

DATASET_PATH = os.path.join(ROOT_DIR, DATA_DIR, DATA_DIR_KEY)

RAW_FILE_PATH = os.path.join(ROOT_DIR, ARTIFACT_DIR_KEY, CURRENT_TIME_STAMP, DATA_INGESTION_KEY, DATA_INGESTION_RAW_DATA_DIR, RAW_DATA_DIR_KEY)
# artifact/{current_time_stamp}/data_ingestion/raw_data_dir/raw.csv

TRAIN_FILE_PATH = os.path.join(ROOT_DIR, ARTIFACT_DIR_KEY, CURRENT_TIME_STAMP, DATA_INGESTION_KEY, DATA_INGESTION_INGESTED_DATA_DIR_KEY, TRAIN_DATA_DIR_KEY)
# artifact/{current_time_stamp}/data_ingestion/ingested_dir/train.csv

TEST_FILE_PATH = os.path.join(ROOT_DIR, ARTIFACT_DIR_KEY , CURRENT_TIME_STAMP, DATA_INGESTION_KEY, DATA_INGESTION_INGESTED_DATA_DIR_KEY, TEST_DATA_DIR_KEY)
# artifact/{current_time_stamp}/data_ingestion/ingested_dir/test.csv



# Data Transformation realated paths...

PROCESSED_TRAIN_FILE_PATH = os.path.join(ROOT_DIR, ARTIFACT_DIR_KEY, CURRENT_TIME_STAMP, DATA_TRANSFORMATION_KEY, TRANSFORMED_DATA_DIR, TRANSFORMED_TRAIN_DATA_FILE)
# artifact/{current_time_stamp}/data_transformation/transformed_data/train.csv

PROCESSED_TEST_FILE_PATH = os.path.join(ROOT_DIR, ARTIFACT_DIR_KEY, CURRENT_TIME_STAMP, DATA_TRANSFORMATION_KEY, TRANSFORMED_DATA_DIR, TRANSFORMED_TEST_DATA_FILE)
# artifact/{current_time_stamp}/data_transformation/transformed_data/test.csv

PREPROCESSOR_FILE_PATH = os.path.join(ROOT_DIR, ARTIFACT_DIR_KEY, CURRENT_TIME_STAMP, DATA_TRANSFORMATION_KEY, PREPROCESSOR_DIR, PREPROCESSOR_FILE)
# artifact/{current_time_stamp}/data_transformation/preprocessor/preprocessor.pkl

FEATURE_ENGG_FILE_PATH = os.path.join(ROOT_DIR, ARTIFACT_DIR_KEY, CURRENT_TIME_STAMP, DATA_TRANSFORMATION_KEY, PREPROCESSOR_DIR, 'feature_engg.pkl')



# Model Training related paths...

MODEL_FILE_PATH = os.path.join(ROOT_DIR, ARTIFACT_DIR_KEY, CURRENT_TIME_STAMP, MODEL_TRAINING_KEY, MODEL_OBJECT)