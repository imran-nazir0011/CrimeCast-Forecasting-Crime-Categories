import os 
import sys
import pandas as pd
from src.components.data_cleaning import DataCleaning
from src.exception import CustomException
from src.logger import logging

from src.utils import load_data

class PredictPipeline:
    def __init__(self):
        self.data_clean=DataCleaning()
    
    def predict(self,features):
        
        logging.info('cleaning the features and performing feature engineering')
        features=self.data_clean.clean_data(features)
        features=self.data_clean.feature_engineering(features)
        model_path=os.path.join('artifacts','model.pkl')
        preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
        logging.info('before loading model and preprocessor')

        
        preprocessor=load_data(file_path=preprocessor_path)
        model=load_data(file_path=model_path)

        logging.info('after loading model and preprocessor')

        cleaned__data=preprocessor.transform(features)
    
        prediction=model.predict(cleaned__data)
        label_path=os.path.join('artifacts','labels.csv')
        label=pd.read_csv(label_path)
        predicted_label=label.loc[label['Code']==prediction[0],'Crime_Category']
        logging.info(f'The predicted Crime Category  is {predicted_label.iloc[0]}')

        return predicted_label.iloc[0]


class CustomData:
    def __init__(self,Location:str,Cross_Street:str,Latitude:float,Longitude:float,Date_Reported:str,
       Date_Occurred:str, Time_Occurred:int, Area_ID:int, Area_Name:str,
       Reporting_District_no:int, Part_1_2:int, Modus_Operandi:str, Victim_Age:int,
       Victim_Sex:str, Victim_Descent:str, Premise_Code:int, Premise_Description:str,
       Weapon_Used_Code:float, Weapon_Description:str, Status:str,
       Status_Description:str):

       self.Location=Location
       self.Cross_Street=Cross_Street 
       self.Latitude=Latitude 
       self.Longitude=Longitude
       self.Date_Occurred=Date_Occurred 
       self.Date_Reported=Date_Reported
       self.Time_Occurred=Time_Occurred 
       self.Area_ID =Area_ID
       self.Area_Name=Area_Name
       self.Reporting_District_no =Reporting_District_no
       self.Part_1_2=Part_1_2
       self.Modus_Operandi =Modus_Operandi
       self.Victim_Age=Victim_Age
       self.Victim_Sex =Victim_Sex
       self.Victim_Descent =Victim_Descent
       self.Premise_Code =Premise_Code
       self.Premise_Description =Premise_Description
       self.Weapon_Used_Code = Weapon_Used_Code
       self.Weapon_Description  = Weapon_Description
       self.Status=Status
       self.Status_Description = Status_Description

    def get_feature_as_dataframe(self):

        try :
            feature_dict = {
                'Location': [self.Location],
                'Cross_Street': [self.Cross_Street],
                'Latitude': [self.Latitude],
                'Longitude': [self.Longitude],
                'Date_Reported': [self.Date_Reported],
                'Date_Occurred': [self.Date_Occurred],
                'Time_Occurred': [self.Time_Occurred],
                'Area_ID': [self.Area_ID],
                'Area_Name': [self.Area_Name],
                'Reporting_District_no': [self.Reporting_District_no],
                'Part 1-2': [self.Part_1_2],  # Use self.Part_1_2 for 'Part 1-2'
                'Modus_Operandi': [self.Modus_Operandi],
                'Victim_Age': [self.Victim_Age],
                'Victim_Sex': [self.Victim_Sex],
                'Victim_Descent': [self.Victim_Descent],
                'Premise_Code': [self.Premise_Code],
                'Premise_Description': [self.Premise_Description],
                'Weapon_Used_Code': [self.Weapon_Used_Code],
                'Weapon_Description': [self.Weapon_Description],
                'Status': [self.Status],
                'Status_Description': [self.Status_Description] 
            }

            return pd.DataFrame(feature_dict)
        
        except Exception as e:
            raise CustomException(e,sys)