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

# Map the selected categorical values to dummy variables
district_dummies = pd.get_dummies(original_df['district']).loc[:, selected_district]
garage_dummies = pd.get_dummies(original_df['garage']).loc[:, selected_garage]

# Numeric inputs (you can customize these based on your features)
area = st.slider('Area (m2)', min_value=0, max_value=500, value=100)
age_of_building = st.slider('Age of Building', min_value=0, max_value=50, value=10)
building_floor = st.slider('Building Floor', min_value=1, max_value=20, value=5)
apartment_floor = st.slider('Apartment Floor', min_value=1, max_value=20, value=5)
num_windows = st.slider('Number of Windows', min_value=1, max_value=10, value=5)

# Prepare the input data for prediction
input_data = pd.DataFrame({
    'area_m2': [area],
    'age_building': [age_of_building],
    'building_floor': [building_floor],
    'apartment_floor': [apartment_floor],
    'number_windows': [num_windows],
    selected_district: [district_dummies.iloc[0]],
    selected_garage: [garage_dummies.iloc[0]]
})

# Make predictions
prediction = model.predict(input_data)

# Display the prediction
st.subheader('Predicted Apartment Price:')
st.write(prediction[0])