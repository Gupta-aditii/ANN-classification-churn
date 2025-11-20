import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler , LabelEncoder , OneHotEncoder
import pickle

# load the trained model
model = tf.keras.models.load_model('model.h5')

# ----------------------------------------------------
# LOAD THE ENCODERS & SCALER
# ----------------------------------------------------

# Load OneHotEncoder for Geography
with open('onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo = pickle.load(file)   # FIXED NAME

# DEBUG CHECK (THIS IS WHERE YOU ADD IT)
print("Loaded geo encoder:", onehot_encoder_geo)

# Load LabelEncoder for Gender (you said keep name 'gebder')
with open('label_encoder_gebder.pkl','rb') as file:
    label_encoder_gebder = pickle.load(file)

# Load Scaler
with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)

# ----------------------------------------------------
# STREAMLIT APP
# ----------------------------------------------------

st.title("Customer Churn Prediction")

# user input fields
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gebder.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# ----------------------------------------------------
# PREPARE INPUT DATA
# ----------------------------------------------------

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gebder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode Geography
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
)

# Combine with input_data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale input
input_data_scaled = scaler.transform(input_data)

# ----------------------------------------------------
# PREDICTION
# ----------------------------------------------------

prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is unlikely to churn.")
