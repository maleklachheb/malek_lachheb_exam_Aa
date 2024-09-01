# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 15:34:27 2024

@author: asus
"""

# Model deployment using Streamlit

import streamlit as st
import pandas as pd
import joblib
import json
from sklearn import set_config

set_config(transform_output='pandas')

# Load the filepaths
FILEPATHS_FILE = 'config/filepaths.json'
with open(FILEPATHS_FILE) as f:
    FPATHS = json.load(f)

# Define the load train or test data function with caching
@st.cache_data
def load_Xy_data(fpath):
    train_path = fpath['data']['ml']['train']
    X_train, y_train = joblib.load(train_path)
    test_path = fpath['data']['ml']['test']
    X_test, y_test = joblib.load(test_path)
    return X_train, y_train, X_test, y_test

@st.cache_resource
def load_model_ml(fpath):
    model_path = fpath['models']['linear_regression']
    linreg = joblib.load(model_path)
    return linreg

### Start of App
st.title('House Prices Prediction')
st.image('image/dollar.jpg')
st.sidebar.header("House Features")

# Load training and testing data
X_train, y_train, X_test, y_test = load_Xy_data(FPATHS)

# Load the model
linreg = load_model_ml(FPATHS)

# Sidebar Sliders
Bedrooms = st.sidebar.slider('Bedrooms',
                              min_value=X_train['bedrooms'].min(),
                              max_value=X_train['bedrooms'].max(),
                              step=1, value=3)

Bathrooms = st.sidebar.slider('Bathrooms',
                               min_value=X_train['bathrooms'].min(),
                               max_value=X_train['bathrooms'].max(),
                               step=0.25, value=2.5)

sqft_lot = st.sidebar.number_input('sqft_lot',
                                     min_value=290,
                                     max_value=X_train['sqft_lot'].max(),
                                     step=150, value=2500)

# Add text for entering features
st.subheader("Select values using the sidebar on the left.\nThen check the box below to predict the price.")

st.sidebar.subheader("Enter/select House Features For Prediction")

# Define function to convert widget values to DataFrame
def get_X_to_predict():
    X_to_predict = pd.DataFrame({'bedrooms': Bedrooms,
                                 'bathrooms': Bathrooms,
                                 'sqft_lot': sqft_lot},
                                 index=['House'])
    return X_to_predict

def get_prediction(model, X_to_predict):
    return model.predict(X_to_predict)[0]

if st.sidebar.button('Predict'):
    X_to_pred = get_X_to_predict()
    new_pred = get_prediction(linreg, X_to_pred)
    st.markdown(f"> #### Model Predicted Price = ${new_pred:,.0f}")
else:
    st.empty()









