import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.components.data_cleaning import DataCleaning  # Update the import path as needed

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')
    cleaned_data_path: str = os.path.join('artifacts', 'cleaned_data.csv')  # New path for cleaned data

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        self.data_cleaning = DataCleaning()  # Instantiate the DataCleaning class
    
    def initiate_data_ingestion(self):
        logging.info('Data Ingestion has initiated')
        try:
            # Load raw data
            df = pd.read_csv('notebook/data/crime_data.csv')
            logging.info('Raw data has been loaded')
            
            # Save the raw data to the raw_data_path
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info('Raw data has been saved to path')

            # Clean the data using DataCleaning class
            cleaned_df = self.data_cleaning.clean_data(df)
            engineered_df = self.data_cleaning.feature_engineering(cleaned_df)
            logging.info('Data Cleaning and Feature Engineering completed')

            # Save the cleaned data to the cleaned_data_path
            os.makedirs(os.path.dirname(self.ingestion_config.cleaned_data_path), exist_ok=True)
            engineered_df.to_csv(self.ingestion_config.cleaned_data_path, index=False, header=True)
            logging.info('Cleaned data has been saved to path')

            # Split the data into training and testing sets
            logging.info('Train Test Split has initiated')
            train_set, test_set = train_test_split(engineered_df, test_size=0.2, random_state=42)
            
            # Save the train and test data to their respective paths
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info("Data Ingestion and splitting completed")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.cleaned_data_path  # Return the path to the cleaned data
            )

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == '__main__':
    data_ingestion = DataIngestion()
    train_data, test_data, cleaned_data = data_ingestion.initiate_data_ingestion()
    print(f"Training data saved to: {train_data}")
    print(f"Testing data saved to: {test_data}")
    print(f"Cleaned data saved to: {cleaned_data}")
