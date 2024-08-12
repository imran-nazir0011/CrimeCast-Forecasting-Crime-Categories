import os
import sys

from dataclasses import dataclass

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging

# from src.utils import save_objects

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path:str=os.path.join('artifacts','preprocessor.pk1')

class DataTransformation:

    def __init__(self):

        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformer_object(self):
        
        try:
            numerical_columns=['Latitude','Longitude','Area_ID','Reporting_District_no','Part 1-2','Victim_Age','Premise_Code','Weapon_Used_Code','Hours_Occurred','Reported_Year','Reported_Month','Reported_Weekday','Occurred_Day','Occurred_Weekday','Modus_Operandi_num_code','Time_Difference_Log']
            
            categorical_columns=['Location', 'Victim_Sex', 'Victim_Descent', 'Status']

            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )

            logging.info(f'Categorical Features:{categorical_columns}')
            logging.info(f'Numerical Features : {numerical_columns}')

            preprocessor=ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,numerical_columns),
                    ('cat_pipeline',cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
    

    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info('Read train and test data completed')

            logging.info('Obtaining Preprocessing object')
            
            preprocessor_obj=self.get_data_transformer_object()

            target='Crime_Category'

            
        
        except Exception as e:
            pass