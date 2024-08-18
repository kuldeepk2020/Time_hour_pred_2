import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import gdown

# URL to download the dataset
url = 'https://drive.google.com/uc?export=download&id=1qnndKXlJnFHVkzP1bcIJIbCclVsIOpSh'

# Path to save the downloaded file
file_path = 'Data_Set.csv'

# Function to download and load data
def load_data_from_gdrive(url):
    gdown.download(url, file_path, quiet=False)
    data = pd.read_csv(file_path)
    return data

# Function to train XGBoost model
def train_model(data):
    model = xgb.XGBRegressor()
    X = data[['Time_hour', 'Time_hour_ctc', 'itemnumber', 'itemcooktime', 'itemid', 'day_shift']]   
    y = data['total_sales']  # Assuming 'total_sales' is your target variable
    model.fit(X, y)
    return model

# Function to predict using the trained model
def predict(model, input_data):
    prediction = model.predict(input_data)
    return prediction

# Streamlit app
st.title('Time Hours Sales Prediction')

try:
    # Load data from Google Drive
    data = load_data_from_gdrive(url)

    # Display data preview
    st.write('Data Preview:', data.head(2))

    # Train model
    model = train_model(data)

    # Input for prediction
    Time_hour = st.slider('Time_hour', min_value=0, max_value=23, value=4)
    Time_hour_ctc = st.slider('Time_hour_ctc', min_value=0, max_value=23, value=3)
    itemnumber = st.slider('itemnumber', min_value=0, max_value=10000, value=1)
    itemcooktime = st.slider('itemcooktime', min_value=0, max_value=10000, value=50)
    itemid = st.slider('ITEM_ID', min_value=0, max_value=10000, value=1)
    day_shift = st.slider('day_shift', min_value=0, max_value=3, value=1)

    # Predict
    input_data = np.array([[Time_hour, Time_hour_ctc, itemnumber, itemcooktime, itemid, day_shift]])
    prediction = predict(model, input_data)
    st.write('Total_Sales:', prediction)

except Exception as e:
    st.error(f"Error loading or processing file: {e}")





