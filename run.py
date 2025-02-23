import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import PredictPipeline, CustomData
from src.components.data_ingestion import DataIngestionConfig

# Flask API URL (Change if needed)
FLASK_API_URL = "http://127.0.0.1:5000/predict"  

# Load dataset from Data Ingestion Config
@st.cache_data
def load_data():
    df = pd.read_csv(DataIngestionConfig().raw_data_path)  # Update path based on config
    return df

df = load_data()

# Create Mapping Dictionaries
Area = df[['Area_ID', 'Area_Name']].drop_duplicates().set_index('Area_Name').to_dict()['Area_ID']
Premise = df[['Premise_Code', 'Premise_Description']].drop_duplicates().set_index('Premise_Description').to_dict()['Premise_Code']
Weapon = df[['Weapon_Used_Code', 'Weapon_Description']].drop_duplicates().set_index('Weapon_Description').to_dict()['Weapon_Used_Code']
STATUS = df[['Status', 'Status_Description']].drop_duplicates().set_index('Status_Description').to_dict()['Status']

# Extract unique values for dropdowns (Excluding mapped columns)
location_options = sorted(df["Location"].dropna().unique().tolist())
reporting_district_options = sorted(df["Reporting_District_no"].dropna().astype(str).unique().tolist())
victim_sex_options = sorted(df["Victim_Sex"].dropna().unique().tolist())
victim_descent_options = sorted(df["Victim_Descent"].dropna().unique().tolist())
street_options =sorted(df['Cross_Street'].dropna().unique().tolist())

# Sidebar Navigation
st.sidebar.title("LA Police Crime Prediction App")
page = st.sidebar.radio("Navigate", ["Predict Crime"])

if page == "Predict Crime" :

    st.title("Crime Data Prediction 🔍")
    st.write("Fill in the details below to predict crime occurrence.")

    with st.form("crime_prediction_form"):
        # --- Location Details ---
        st.subheader("📍Crime Area & Location Details")
        latitude = st.number_input("Latitude", min_value=33.7037, max_value=34.3373, step=0.0001)
        longitude = st.number_input("Longitude", min_value=-118.6682, max_value=-118.1553, step=0.0001)
        location = st.selectbox("Location",["Select Location"] + location_options)
        cross_street=st.selectbox("Cross street",["Select Cross Street"]+street_options)
        area_name = st.selectbox("Area Name", ["Select Area"] + list(Area.keys()))
        reporting_district_no = st.selectbox("Reporting District No", reporting_district_options)
        
        # --- Time & Date Details ---
        st.subheader("📅 Time & Date Details")
        date_reported = st.date_input("Date Reported", datetime.date.today())
        date_occurred = st.date_input("Date Occurred", datetime.date.today())
        time_occurred = st.time_input("Time Occurred", datetime.time(12, 0))
        # --- Crime type ---
        st.subheader("🚔 Crime Type")
        part_1_2 = st.radio("Part 1 or Part 2 Crime?", [1, 2])
        
        # --- Victim Details ---
        st.subheader("🧑 Victim Details")
        victim_age = st.number_input("Victim Age", min_value=0, step=1)
        victim_sex = st.selectbox("Victim Sex", victim_sex_options)
        victim_descent = st.selectbox("Victim Descent", victim_descent_options)

        # --- Additional Crime Details ---
        st.subheader("⚠️ Additional Crime Details")
        premise_description = st.selectbox("Premise Description", ["Select Premise"]+list(Premise.keys()) )
        weapon_description = st.selectbox("Weapon Description", ["Select Weapon"]+list(Weapon.keys()) )
        status_description = st.selectbox("Status Description", ["Select Status"]+list(STATUS.keys()) )

        # --- Submit Button ---
        submit_button = st.form_submit_button("🚀 Predict Crime")

    # --- PREDICTION REQUEST ---
    if submit_button:
        # Create CustomData instance
        crime_data = CustomData(
            Location=location,
            Cross_Street=cross_street,
            Latitude = latitude,
            Longitude=longitude,
            Date_Reported=date_reported.strftime("%Y-%m-%d"),
            Date_Occurred=date_occurred.strftime("%Y-%m-%d"),
            Time_Occurred=str(time_occurred),
            Area_Name=area_name,
            Area_ID=Area.get(area_name, None),  # Map Area_Name → Area_ID
            Reporting_District_no=reporting_district_no,
            Part_1_2=part_1_2,
            Victim_Age=victim_age,
            Victim_Sex=victim_sex,
            Victim_Descent=victim_descent,
            Premise_Description = premise_description,
            Premise_Code=Premise.get(premise_description, None),  # Map Premise_Description → Premise_Code
            Weapon_Description = weapon_description,
            Weapon_Used_Code=Weapon.get(weapon_description, None),  # Map Weapon_Description → Weapon_Used_Code
            Status_Description = status_description,
            Status=STATUS.get(status_description, None)  # Map Status_Description → Status
        )

        # Convert to DataFrame for preprocessing
        crime_df = crime_data.get_feature_as_dataframe()


        # Load Predict Pipeline and make prediction
        predict_pipeline = PredictPipeline()
        prediction_result = predict_pipeline.predict(crime_df)

        st.success(f"🎯 **Prediction Result:** {prediction_result}")