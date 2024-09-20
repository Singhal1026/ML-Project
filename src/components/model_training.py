from src.constants import *
from src.config.configuration import *
from src.logger import logging
from src.exceptions import CustomException
from src.utils import evaluate_model, save_obj
from dataclasses import dataclass
import os, sys 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
import pandas as pd
import numpy as np



@dataclass
class ModelTrainerConfig:
    trained_model_file_path = MODEL_FILE_PATH


class ModelTraining:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_training(self, train_arr, test_arr):
        try:
            X_train, y_train, X_test, y_test = (train_arr[:, :-1], train_arr[:, -1], test_arr[:, :-1], test_arr[:, -1])

            models = {
                "XGBRegressor" : XGBRegressor(),
                "DecisionTreeRegressor" : DecisionTreeRegressor(),
                "GradientBoostingRegressor" : GradientBoostingRegressor(),
                "RandomForestRegressor" : RandomForestRegressor(),
                "SVR" : SVR()
            }

            model_report, best_model_obj = evaluate_model(X_train, y_train, X_test, y_test, models)
            logging.info(model_report)
            print(model_report)

            best_model_name = sorted(model_report, key=lambda x: model_report[x], reverse=True)[0]
            best_model_score = model_report[best_model_name]
        
            print(f"Best model : {best_model_name} r2_score : {best_model_score}")

            logging.info(f"Best model : {best_model_name} r2_score : {best_model_score}")

            save_obj(file_path = self.model_trainer_config.trained_model_file_path, obj = best_model_obj)


        except Exception as e:
            raise CustomException(e, sys)

