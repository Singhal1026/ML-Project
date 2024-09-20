import logging
import os, sys
from datetime import datetime

LOG_DIR = 'logs'
LOG_DIR_PATH = os.path.join(os.getcwd(), LOG_DIR)

os.makedirs(LOG_DIR_PATH, exist_ok=True)

CURRENT_TIME_STAMP = f'{datetime.now():%Y-%m-%d_%H-%M-%S}'

FILE_NAME = f'log_{CURRENT_TIME_STAMP}.log'
LOG_FILE_PATH = os.path.join(LOG_DIR_PATH, FILE_NAME)

logging.basicConfig(
    filename = LOG_FILE_PATH,
    filemode = 'w',
    format = '[%(asctime)s] - %(name)s - %(levelname)s - %(message)s',
    level = logging.INFO
)
