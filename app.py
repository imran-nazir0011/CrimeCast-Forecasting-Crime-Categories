import pandas as pd
import numpy as np
import logging
import traceback
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import PredictPipeline, CustomData
from src.components.data_ingestion import DataIngestionConfig

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load data and extract unique values for dropdowns
df = pd.read_csv(DataIngestionConfig().raw_data_path)

df['Premise_Description'] = df['Premise_Description'].fillna('NOT APPLICABLE')
df['Weapon_Description'] = df['Weapon_Description'].fillna('NOT APPLICABLE')
df['Weapon_Used_Code'] = df['Weapon_Used_Code'].fillna(float(0.0))

columns = df.select_dtypes(include='object').columns.tolist()
columns.append('Reporting_District_no')
columns.append('Part 1-2')
unique_values = {}
for col in columns:
    unique_list = df[col].dropna().unique().tolist()
    unique_values[col] = sorted(unique_list)

Area = df[['Area_ID', 'Area_Name']].drop_duplicates().set_index('Area_Name').to_dict()['Area_ID']
Premise = df[['Premise_Code', 'Premise_Description']].drop_duplicates().set_index('Premise_Description').to_dict()['Premise_Code']
Weapon = df[['Weapon_Used_Code', 'Weapon_Description']].drop_duplicates().set_index('Weapon_Description').to_dict()['Weapon_Used_Code']
STATUS = df[['Status', 'Status_Description']].drop_duplicates().set_index('Status_Description').to_dict()['Status']

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoints():
    if request.method == 'GET':
        return render_template('predict.html', unique_values=unique_values)
    else:
        try:
            Location = request.form.get('Location')
            Cross_Street = request.form.get('Cross_Street')
            Latitude = float(request.form.get('Latitude'))
            Longitude = float(request.form.get('Longitude'))
            Date_Occurred = request.form.get('Date_Occurred')
            Date_Reported = request.form.get('Date_Reported')
            Time_Occurred = request.form.get('Time_Occurred')  # HH:MM format
            Area_Name = request.form.get('Area_Name')
            Reporting_District_no = int(request.form.get('Reporting_District_no'))
            Part_1_2 = int(request.form.get('Part_1_2'))
            Modus_Operandi = request.form.get('Modus_Operandi')
            Victim_Age = int(request.form.get('Victim_Age'))
            Victim_Sex = request.form.get('Victim_Sex')
            Victim_Descent = request.form.get('Victim_Descent')
            Premise_Description = request.form.get('Premise_Description')
            Weapon_Description = request.form.get('Weapon_Description')
            Status_Description = request.form.get('Status_Description')

            Area_ID = int(Area[Area_Name])
            Premise_Code = int(Premise[Premise_Description])
            Weapon_Used_Code = float(Weapon[Weapon_Description])
            Status = STATUS[Status_Description]

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
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            logging.error(traceback.format_exc())
            return render_template('predict.html', result=f"Error occurred : {e}", unique_values=unique_values)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
