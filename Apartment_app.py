import joblib
import pandas as pd
import streamlit as st

# Load the trained model
model = joblib.load('random_forest_model.joblib')

# Load the original DataFrame with categorical columns
# This is used to map dummy variables back to the original categorical values
original_df = pd.read_csv('apartment_data.csv')  # Replace with your actual dataset

# Create a Streamlit app
st.title('Apartment Price Prediction App')

# Numeric inputs
area = st.slider('Area', min_value=0, max_value=500, value=100)
age_of_building = st.slider('Age of Building', min_value=0, max_value=50, value=10)
building_floor = st.slider('Building Floor', min_value=1, max_value=20, value=5)
apartment_floor = st.slider('Apartment Floor', min_value=1, max_value=20, value=5)
num_windows = st.slider('Number of Windows', min_value=1, max_value=10, value=5)

# Categorical inputs
district_options = original_df['District'].unique()
selected_district = st.selectbox('Select District', district_options)

garage_options = original_df['Garage'].unique()
selected_garage = st.selectbox('Select Garage', garage_options)

# Map the selected categorical values to dummy variables
district_dummies = pd.get_dummies(original_df['District']).loc[:, selected_district]
garage_dummies = pd.get_dummies(original_df['Garage']).loc[:, selected_garage]

# Prepare the input data for prediction
input_data = pd.DataFrame({
    'Area': [area],
    'Age_of_Building': [age_of_building],
    'Building_Floor': [building_floor],
    'Apartment_Floor': [apartment_floor],
    'Number_of_Windows': [num_windows],
    selected_district: [district_dummies.iloc[0]],
    selected_garage: [garage_dummies.iloc[0]]
})

# Make predictions
prediction = model.predict(input_data)

# Display the prediction
st.subheader('Predicted Apartment Price:')
st.write(prediction[0])