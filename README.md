# Crime Classification Web App

## Overview
This project involves classifying crime categories using LA Police crime data. The LightGBM model, which achieved an accuracy of over 87%, is utilized for predictions. The project covers end-to-end data processing, model training, and deployment via a web application.

## Project Structure
- **Data Ingestion**: Ingests and preprocesses raw crime data, saving it as CSV files.
- **Data Cleaning**: Cleans the data by removing invalid entries, handling missing values, and normalizing data.
- **Feature Engineering**: Extracts and transforms features from the cleaned data to improve model performance.
- **Model Training**: Trains and evaluates various machine learning models, selecting LightGBM based on performance metrics.
- **Prediction Pipeline**: Creates a pipeline for making predictions with the trained model.
- **Web Application**: Hosts a Flask web app to provide a user interface for predictions.

## Setup
1. **Clone the Repository**
   ```bash
   git clone 'https://github.com/imran-nazir0011/CrimeCast-Forecasting-Crime-Categories'
   cd 'CrimeCast-Forecasting-Crime-Categories'
