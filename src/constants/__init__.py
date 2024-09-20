import os, sys
from datetime import datetime


CURRENT_TIME_STAMP = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

ROOT_DIR_KEY = os.getcwd()
DATA_DIR = 'data'
DATA_DIR_KEY = 'finalTrain.csv'

ARTIFACT_DIR_KEY = 'artifact'

# Data Ingestion Constants

DATA_INGESTION_KEY = 'data_ingestion'
DATA_INGESTION_RAW_DATA_DIR = 'raw_data_dir'           # contains raw.csv which contains data which is extracted from source
DATA_INGESTION_INGESTED_DATA_DIR_KEY = 'ingested_dir'  # contains train.csv, test.csv.
RAW_DATA_DIR_KEY = 'raw.csv'
TRAIN_DATA_DIR_KEY = 'train.csv'
TEST_DATA_DIR_KEY = 'test.csv'

# - artifact/data_ingestion/ingested_dir/train.csv
# - artifact/data_ingestion/ingested_dir/test.csv
# - artifact/data_ingestion/raw_data_dir/raw.csv



# Data Transformation Constants

DATA_TRANSFORMATION_KEY = 'data_transformation'
PREPROCESSOR_DIR = 'preprocessor'
PREPROCESSOR_FILE = 'preprocessor.pkl'
TRANSFORMED_DATA_DIR = 'transformed_data'
TRANSFORMED_TRAIN_DATA_FILE = 'train.csv'
TRANSFORMED_TEST_DATA_FILE = 'test.csv'

# - artifact/data_transformation/transformed_data/train.csv
# - artifact/data_transformation/transformed_data/test.csv
# - artifact/data_transformation/preprocessor/preprocessor.pkl



# Model Training Constants

MODEL_TRAINING_KEY = 'model_training'
MODEL_OBJECT = 'model.pkl'