import streamlit as st
import joblib  # To load the trained model
import pandas as pd
import numpy as np

# Load the trained model
model = joblib.load('model.pkl')

# Front-end - User input form
st.title('Wine Type Prediction')

# Add subtitle with your name and ID
st.subheader("Craig Chadiwa: R215904U")

# Create three columns to display inputs in one row
col1, col2, col3 = st.columns(3)

# Inputs for the user (e.g., features like alcohol, acidity, etc.)
with col1:
    alcohol = st.number_input('Alcohol', min_value=0.0, max_value=20.0, step=0.1)

with col2:
    fixed_acidity = st.number_input('Fixed Acidity', min_value=0.0, max_value=20.0, step=0.1)

with col3:
    volatile_acidity = st.number_input('Volatile Acidity', min_value=0.0, max_value=2.0, step=0.01)

# Create another row with 3 more inputs
col4, col5, col6 = st.columns(3)

with col4:
    citric_acid = st.number_input('Citric Acid', min_value=0.0, max_value=2.0, step=0.01)

with col5:
    residual_sugar = st.number_input('Residual Sugar', min_value=0.0, max_value=20.0, step=0.1)

with col6:
    chlorides = st.number_input('Chlorides', min_value=0.0, max_value=1.0, step=0.01)

# Create another row with 3 more inputs
col7, col8, col9 = st.columns(3)

with col7:
    free_sulfur_dioxide = st.number_input('Free Sulfur Dioxide', min_value=0.0, max_value=100.0, step=0.1)

with col8:
    total_sulfur_dioxide = st.number_input('Total Sulfur Dioxide', min_value=0.0, max_value=200.0, step=0.1)

with col9:
    density = st.number_input('Density', min_value=0.0, max_value=2.0, step=0.0001)

# Create another row with 2 more inputs
col10, col11, col12 = st.columns(3)

with col10:
    pH = st.number_input('pH', min_value=0.0, max_value=14.0, step=0.01)

with col11:
    sulphates = st.number_input('Sulphates', min_value=0.0, max_value=2.0, step=0.01)

with col12:
    quality = st.number_input('Quality', min_value=0, max_value=10, step=1)

# Create a dataframe with the user inputs
user_input = pd.DataFrame([[alcohol, fixed_acidity, volatile_acidity, citric_acid, residual_sugar, 
                            chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, 
                            sulphates, quality]],  # Include quality as an input
                          columns=['alcohol', 'fixed_acidity', 'volatile_acidity', 'citric_acid', 
                                   'residual_sugar', 'chlorides', 'free_sulfur_dioxide', 
                                   'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'quality'])

# Prediction button
if st.button('Predict Wine Type'):
    try:
        # Make prediction (0 for white wine, 1 for red wine)
        prediction = model.predict(user_input)
        
        # Display the predicted wine type
        wine_type = "Red Wine" if prediction[0] == 1 else "White Wine"
        st.write(f'Predicted Wine Type: {wine_type}')
        
    except Exception as e:
        st.error(f"Error: {e}")
