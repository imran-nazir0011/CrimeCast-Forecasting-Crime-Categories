import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass


@dataclass
class DataCleaning:
    def __init__(self):
        pass

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logging.info('Data Cleaning has initiated')
            
            # Remove rows with Latitude or Longitude equal to 0
            df = df[(df['Latitude'] != 0) & (df['Longitude'] != 0)]
            
            # Fill missing values for specific columns
            df['Weapon_Used_Code'] = df['Weapon_Used_Code'].fillna(0.0)
            df['Weapon_Description'] = df['Weapon_Description'].fillna('Not Reported')
            df['Premise_Description'] = df['Premise_Description'].fillna('Not Reported')
            logging.info(f'Data Cleaning completed. Shape of data: {df.shape}')
            return df

        except Exception as e:
            raise CustomException(e, sys)

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logging.info('Feature Engineering has initiated')

            # Function to normalize date formats
            def normalize_date(date_str):
                try:
                    date_obj = datetime.strptime(date_str, '%m/%d/%Y %I:%M:%S %p')
                except ValueError:
                    try:
                        date_obj = datetime.strptime(date_str, '%m-%d-%Y %H:%M')
                    except ValueError:
                        return pd.NaT  # Return NaT if both formats fail
                return date_obj.strftime('%m-%d-%Y %H:%M')

            # Apply the function to both date columns
            df['Date_Reported'] = pd.to_datetime(df['Date_Reported'].apply(normalize_date))
            df['Date_Occurred'] = pd.to_datetime(df['Date_Occurred'].apply(normalize_date))

            # Function to convert Time_Occurred to hours
            def convert_to_hours(time):
                hours = time // 100          # Integer division to get the hour
                minutes = time % 100         # Modulus to get the minutes
                return hours + minutes / 60   # Return total hours as a decimal

            # Apply the function to the Time_Occurred column
            df['Hours_Occurred'] = df['Time_Occurred'].apply(convert_to_hours)

            # Extract features from Date_Reported
            df['Reported_Year'] = df['Date_Reported'].dt.year
            df['Reported_Month'] = df['Date_Reported'].dt.month
            df['Reported_Day'] = df['Date_Reported'].dt.day
            df['Reported_Weekday'] = df['Date_Reported'].dt.weekday

            # Extract features from Date_Occurred
            df['Occurred_Year'] = df['Date_Occurred'].dt.year
            df['Occurred_Month'] = df['Date_Occurred'].dt.month
            df['Occurred_Day'] = df['Date_Occurred'].dt.day
            df['Occurred_Weekday'] = df['Date_Occurred'].dt.weekday

            # Calculate time difference in hours
            df['Time_Difference'] = (df['Date_Reported'] - df['Date_Occurred']).dt.total_seconds() / 3600  # in hours
           
            # Add feature: Modus_Operandi_num_code
            df['Modus_Operandi_num_code'] = df['Modus_Operandi'].apply(lambda x: len(str(x).split()))

            # Log-transform the Time_Difference
            df['Time_Difference_Log'] = np.log(df['Time_Difference'] + 1)

            # Map Crime_Category to 0 to 5 classes
            crime_category=df['Crime_Category'].value_counts().index.tolist()
            labels_map={}
            for i in range(len(crime_category)):
                labels_map[crime_category[i]]=i
            df['Crime_Category'] = df['Crime_Category'].map(labels_map)
     

            # Drop features that were removed during cleaning
            features_to_remove = [
                'Reported_Day', 'Occurred_Year', 'Occurred_Month', 
                'Date_Reported', 'Date_Occurred', 'Area_Name', 
                'Premise_Description', 'Weapon_Description', 
                'Status_Description', 'Time_Difference', 'Modus_Operandi', 
                'Cross_Street', 'Time_Occurred'
            ]
            
            df = df.drop(columns=features_to_remove, axis=1)
            
            # Drop duplicates
            df = df.drop_duplicates()
            
            # Reset index
            df.reset_index(drop=True, inplace=True)
            
            logging.info(f'Feature Engineering completed. Shape of data: {df.shape}')
            return df

        except Exception as e:
            raise CustomException(e, sys)
