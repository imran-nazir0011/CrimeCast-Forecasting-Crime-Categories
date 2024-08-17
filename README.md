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
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   The `setup.py` file will be executed automatically during the installation process.

3. **Run the Application**
   ```bash
   python app.py
   ```
   The web app will be accessible at `http://0.0.0.0:8000`.

## Model Evaluation
- **Training and Evaluation**: Multiple models were trained and evaluated on the dataset.
- **Selected Model**: LightGBM was chosen for its high accuracy of over 87%.
- **Model Persistence**: The trained LightGBM model and other relevant objects are saved as pickle files.

## Pickle Files
- **Saving Objects**: Models and other essential objects are serialized and saved as pickle files.
- **Loading Objects**: The web application loads these pickle files to make predictions.

## Exception Handling
- **CustomException**: Custom exceptions are implemented to manage and log errors encountered during data processing and model operations.
- **Logging**: Comprehensive logging is used to track data processing, model training, and prediction activities.

## Utilities
- **Model Evaluation**: Includes functions for evaluating model performance and comparing different models.
- **Pickle Management**: Utility functions for saving and loading objects as pickle files to facilitate easy model management and deployment.

## Contributing
Contributions to the project are welcome. Please open issues or submit pull requests in accordance with the coding guidelines and testing procedures provided.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgements
- **LightGBM**: For its high-performance classification capabilities.
- **Flask**: For enabling the creation of the web application.
- **LA Police Dataset**: For providing the data used in this project.

## Additional Information

### Data Ingestion
- **Path**: `artifacts/raw.csv`
- **Description**: Raw crime data is ingested and saved as CSV files. This data is then used for further processing and model training.

### Data Cleaning
- **Cleaning Steps**: Removes invalid entries, handles missing values, normalizes data, and drops unnecessary columns.
- **Path for Cleaned Data**: `artifacts/cleaned_data.csv`

### Feature Engineering
- **Features Created**: Date normalization, time conversion, feature extraction, and new feature creation.
- **Features Removed**: Includes redundant or irrelevant features based on domain knowledge and data analysis.

### Model Training
- **Models Evaluated**: Various models were evaluated, including LightGBM.
- **Best Model**: LightGBM was selected due to its superior performance metrics.

### Prediction Pipeline
- **Usage**: The pipeline is used for making predictions on new data using the trained model.
- **Integration**: Integrated with the Flask web app to provide real-time predictions.

### Web Application
- **Framework**: Flask
- **Functionality**: Provides a user interface for inputting crime data and receiving predictions from the model.
- **Deployment**: Can be hosted locally or on a server to make predictions accessible via a web interface.
