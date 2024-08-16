import pandas as pd
import numpy as np
import json
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import PredictPipeline, CustomData
from src.components.data_ingestion import DataIngestionConfig
from src.logger import logging

# Load data and extract unique values for dropdowns
df = pd.read_csv(DataIngestionConfig().raw_data_path)
string_columns = df.select_dtypes(include='object').columns.tolist()
unique_values = {}
for col in string_columns:
    unique_values[col] = df[col].dropna().unique()

application = Flask(__name__)
app = application

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoints():
    if request.method == 'GET':
        # Pass unique values to the template
        return render_template('predict.html', unique_values=unique_values)
    else:
        Location = request.form.get('Location')
        Cross_Street = request.form.get('Cross_Street')
        Latitude = float(request.form.get('Latitude'))  # Default to 0.0 if not present
        Longitude = float(request.form.get('Longitude'))  # Default to 0.0 if not present
        Date_Reported = request.form.get('Date_Reported')
        Date_Occurred = request.form.get('Date_Occurred')
        Time_Occurred = request.form.get('Time_Occurred')  # HH:MM format
        Area_ID = int(request.form.get('Area_ID'))  # Default to 0 if not present
        Area_Name = request.form.get('Area_Name')
        Reporting_District_no = int(request.form.get('Reporting_District_no'))  # Default to 0 if not present
        Part_1_2 = int(request.form.get('Part_1_2'))  # Default to 0 if not present
        Modus_Operandi = request.form.get('Modus_Operandi')
        Victim_Age = int(request.form.get('Victim_Age'))  # Default to 0 if not present
        Victim_Sex = request.form.get('Victim_Sex')
        Victim_Descent = request.form.get('Victim_Descent')
        Premise_Code = int(request.form.get('Premise_Code'))  # Default to 0 if not present
        Premise_Description = request.form.get('Premise_Description')
        Weapon_Used_Code = float(request.form.get('Weapon_Used_Code'))  # Default to 0.0 if not present
        Weapon_Description = request.form.get('Weapon_Description')
        Status = request.form.get('Status')
        Status_Description = request.form.get('Status_Description')

        # Convert Time Occurred from HH:MM to HHMM
        if Time_Occurred:
            Time_Occurred = Time_Occurred.replace(':', '')  # Remove colon
            Time_Occurred = int(Time_Occurred)  # Convert to integer
        else:
            Time_Occurred = 0  # Default to 0 if not provided

        data = CustomData(
            Location=Location,
            Cross_Street=Cross_Street,
            Latitude=Latitude,
            Longitude=Longitude,
            Date_Reported=Date_Reported,
            Date_Occurred=Date_Occurred,
            Time_Occurred=Time_Occurred,
            Area_ID=Area_ID,
            Area_Name=Area_Name,
            Reporting_District_no=Reporting_District_no,
            Part_1_2=Part_1_2,
            Modus_Operandi=Modus_Operandi,
            Victim_Age=Victim_Age,
            Victim_Sex=Victim_Sex,
            Victim_Descent=Victim_Descent,
            Premise_Code=Premise_Code,
            Premise_Description=Premise_Description,
            Weapon_Used_Code=Weapon_Used_Code,
            Weapon_Description=Weapon_Description,
            Status=Status,
            Status_Description=Status_Description
        )
        f = data.get_feature_as_dataframe()
        obj = PredictPipeline()
        result = obj.predict(f)
        return render_template('predict.html', result=result, unique_values=unique_values)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
