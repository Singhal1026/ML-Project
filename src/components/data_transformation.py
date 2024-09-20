from src.constants import *
from src.logger import logging
from src.exceptions import CustomException
from src.config import *
from src.config.configuration import *
from src.utils import save_obj
import os, sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline

class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        logging.info('****************************Feature Engineering Start****************************')

    def distance(self, df, lat1, lon1, lat2, lon2):
        p = np.pi/180
        a = 0.5-np.cos((df[lat2]-df[lat1])*p)/2 + np.cos(df[lat1]*p) * np.cos(df[lat2]*p) * (1 - np.cos((df[lon2] - df[lon1]) * p))
        df['distance'] = 12734 * np.arccos(np.clip(a, -1, 1))

    def transform_data(self, df):
        try:
            # df = df.copy()
            df.drop(['ID'], axis=1, inplace=True)
            self.distance(df, 'Restaurant_latitude', 'Restaurant_longitude', 'Delivery_location_latitude', 'Delivery_location_longitude')
            
            df.drop(['Delivery_person_ID', 'Restaurant_latitude', 'Restaurant_longitude', 'Delivery_location_latitude', 'Delivery_location_longitude', 'Order_Date', 'Time_Orderd', 'Time_Order_picked'], axis=1, inplace=True)

            logging.info('Columns dropped and distance calculated successfully')

            return df

        except Exception as e:
            raise CustomException(e, sys)
        
    def fit(self, X, y=None):
        # No fitting required for this transformer, so just return self
        return self

        
    def transform(self, X:pd.DataFrame, y=None):
        try:
            transformed_df = self.transform_data(X)
            return transformed_df

        except Exception as e:
            logging.error(e)
            raise CustomException(e, sys)
        

@dataclass
class DataTransformationConfig:
    processed_obj_file_path = PREPROCESSOR_FILE_PATH
    processed_train_file_path = PROCESSED_TRAIN_FILE_PATH
    processed_test_file_path = PROCESSED_TEST_FILE_PATH
    feature_engg_file_path = FEATURE_ENGG_FILE_PATH


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    
    def get_data_transformed_obj(self):
        try:
            Road_traffic_density = ['Low', 'Medium', 'High', 'Jam']
            Weather_condition = ['Sunny', 'Cloudy', 'Fog', 'Sandstorms', 'Windy', 'Stormy']

            categorical_columns=['Type_of_order','Type_of_vehicle','Festival','City']
            ordinal_encode=['Road_traffic_density','Weather_conditions']
            numerical_columns=['Delivery_person_Age','Delivery_person_Ratings','Vehicle_condition','multiple_deliveries','distance']

            num_pipeline = Pipeline(steps=[
                ('impute', SimpleImputer(strategy='constant', fill_value=0)),
                ('scaler', StandardScaler(with_mean=False))
            ])

            cat_pipeline = Pipeline(steps=[
                ('impute', SimpleImputer(strategy='most_frequent')),
                ('oneHot', OneHotEncoder(handle_unknown='ignore')),
                ('scaler', StandardScaler(with_mean=False))
            ])

            ord_pipeline = Pipeline(steps=[
                ('impute', SimpleImputer(strategy='most_frequent')),
                ('ordinal', OrdinalEncoder(categories=[Road_traffic_density, Weather_condition], handle_unknown='use_encoded_value', unknown_value=-1)),
                ('scaler', StandardScaler(with_mean=False))
            ])

            preprocessor = ColumnTransformer([
                ('numerical_pipeline', num_pipeline, numerical_columns),
                ('categorical_pipeline', cat_pipeline, categorical_columns),
                ('ordinal_pipeline', ord_pipeline, ordinal_encode)
            ], remainder='passthrough')

            logging.info('Column Transformation Completed')

            return preprocessor 
        

        except Exception as e:
            raise CustomException(e, sys)
        

    def get_feature_engg_obj(self):
        try:
            feature_engineering = Pipeline(steps=[('fe', FeatureEngineering())])

            return feature_engineering

        except Exception as e:
            raise CustomException(e, sys)
        

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            fe_obj = self.get_feature_engg_obj()

            train_df = fe_obj.fit_transform(train_df)
            test_df = fe_obj.transform(test_df)

            # train_df.to_csv('train.csv', index=False)
            # test_df.to_csv('test.csv', index=False)

            preprocessor_obj = self.get_data_transformed_obj()

            target_column_name = 'Time_taken (min)'

            X_train = train_df.drop(columns=[target_column_name])
            y_train = train_df[target_column_name]

            X_test = test_df.drop(columns=[target_column_name])
            y_test = test_df[target_column_name]

            X_train_transformed = preprocessor_obj.fit_transform(X_train)
            X_test_transformed = preprocessor_obj.transform(X_test)

            y_train = np.array(y_train).reshape(-1, 1)
            y_test = np.array(y_test).reshape(-1, 1)

            train_arr = np.concatenate([X_train_transformed, y_train], axis=1)
            test_arr = np.concatenate([X_test_transformed, y_test], axis=1)

            df_train = pd.DataFrame(train_arr)
            df_test = pd.DataFrame(test_arr)

            os.makedirs(os.path.dirname(self.data_transformation_config.processed_train_file_path), exist_ok=True)
            df_train.to_csv(self.data_transformation_config.processed_train_file_path, index=False, header=True)

            os.makedirs(os.path.dirname(self.data_transformation_config.processed_test_file_path), exist_ok=True)
            df_test.to_csv(self.data_transformation_config.processed_test_file_path, index=False, header=True)

            save_obj(file_path=self.data_transformation_config.feature_engg_file_path, obj=fe_obj)
            save_obj(file_path=self.data_transformation_config.processed_obj_file_path, obj=fe_obj)

            logging.info('Data tranformation completed')

            return train_arr, test_arr, self.data_transformation_config.processed_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)

