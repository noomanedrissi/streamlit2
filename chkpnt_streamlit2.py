# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 14:39:19 2026

@author: noomane.drissi
"""

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.title("Streamlit Chkpnt 2")

# 1. DATA IMPORT
uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # 2. DATA EXPLORATION & CLEANING
    st.write("### Dataset Preview", df.head(3))
    
    # Simple Cleaning: Drop duplicates & fill missing values with 0
    df = df.drop_duplicates()
    df = df.fillna(0)
    
    # Encode strings to numbers so the ML model can read them
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col].astype(str))

    # 3. TRAINING
    target = st.selectbox("Select the column to predict (Target)", df.columns)
    
    if st.button("Train Model"):
        X = df.drop(columns=[target])
        y = df[target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = RandomForestClassifier().fit(X_train, y_train)
        
        st.session_state.model = model
        st.session_state.features = X.columns.tolist()
        st.success("Model trained successfully!")

    # 4. PREDICTION FORM
    if 'model' in st.session_state:
        st.divider()
        st.subheader("Make a Prediction")
        
        inputs = {}
        # Create an input field for every feature
        for feat in st.session_state.features:
            inputs[feat] = st.number_input(f"Value for {feat}", value=0.0)
        
        if st.button("Run Prediction"):
            input_df = pd.DataFrame([inputs])
            prediction = st.session_state.model.predict(input_df)
            st.metric("Result", prediction[0])