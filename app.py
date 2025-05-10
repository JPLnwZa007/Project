# -*- coding: utf-8 -*-
"""
Created on Sat May 10 18:17:36 2025

@author: folkk
"""

# customer_segmentation.py

import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, 'kmeans_model.pkl')
scaler_path = os.path.join(BASE_DIR, 'scaler.pkl')

model = pickle.load(open(model_path, 'rb'))
scaler = pickle.load(open(scaler_path, 'rb'))


st.title("Customer Segmentation App")

# Input section
st.header("🔍 Predict Segment")
income = st.number_input("Income", min_value=0)
kids = st.slider("Number of Kids", 0, 3)
teens = st.slider("Number of Teens", 0, 3)
recency = st.number_input("Recency", min_value=0)
wines = st.number_input("Monthly Wine Spend")
fruits = st.number_input("Monthly Fruit Spend")

if st.button("Predict Segment"):
    data = [[income, kids, teens, recency, wines, fruits]]
    data_scaled = scaler.transform(data)
    segment = model.predict(data_scaled)
    st.success(f"Predicted Customer Segment: {segment[0]}")

# Visualization section
st.header("📊 Customer Segment Distribution")

try:
    df = pd.read_csv("segmented_customers.csv")

    fig, ax = plt.subplots()
    df['Segment'].value_counts().sort_index().plot(kind='bar', ax=ax)
    ax.set_xlabel("Segment")
    ax.set_ylabel("Number of Customers")
    ax.set_title("Customer Count by Segment")
    st.pyplot(fig)

except FileNotFoundError:
    st.warning("segmented_customers.csv not found. Please run the training script first.")


