import joblib
import pandas as pd
import streamlit as st

# Load the trained model
model = joblib.load('random_forest_model.joblib')

# Create a Streamlit app
st.title('Apartment Price Prediction App')

# Load the original DataFrame with categorical columns
# This is used to map dummy variables back to the original categorical values
original_df = pd.read_csv('apartment_data.csv')  # Replace with your actual dataset

# Unique values for 'district' and 'garage'
district_options = original_df['district'].unique()
garage_options = original_df['garage'].unique()

# User inputs for 'district' and 'garage'
selected_district = st.selectbox('Select District', district_options)
selected_garage = st.selectbox('Select Garage', garage_options)

# Map all unique categorical values to dummy variables
all_district_dummies = pd.get_dummies(original_df['district'], prefix='district')
all_garage_dummies = pd.get_dummies(original_df['garage'], prefix='garage')

# Ensure that the input_data has columns for all dummy variables
input_data = pd.DataFrame({
    'area_m2': [st.slider('Area (m2)', min_value=0, max_value=500, value=100)],
    'age_building': [st.slider('Age of Building', min_value=0, max_value=50, value=10)],
    'building_floor': [st.slider('Building Floor', min_value=1, max_value=20, value=5)],
    'apartment_floor': [st.slider('Apartment Floor', min_value=1, max_value=20, value=5)],
    'number_windows': [st.slider('Number of Windows', min_value=1, max_value=10, value=5)],
    **dict.fromkeys(all_district_dummies.columns, 0),
    **dict.fromkeys(all_garage_dummies.columns, 0)
})

# Set the value for the selected district and garage
input_data[f'district_{selected_district}'] = 1
input_data[f'garage_{selected_garage}'] = 1

# Make predictions
prediction = model.predict(input_data)

# Display the input_data and prediction
st.subheader('Input Data for Prediction:')
st.write(input_data)

st.subheader('Predicted Apartment Price:')
st.write(prediction[0])

# It was really hard to come up with a method to turn the dummied variables back into their normal and then back into the get_dummies columns for the model to understand them 
# This project was was a challenge but I have learned alot from it, basically every step from the beginning is reflected on the next step of the process, every step must be bulletproof
# I hope to be able to independently create such useful tools in the future and one thing I want to learn more about is scraping techniques as well as methods of efficiency in it.