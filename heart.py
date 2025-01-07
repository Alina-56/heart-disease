import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the dataset
def load_data():
    return pd.read_csv('heart.csv')

data = load_data()

# Preprocess the data
le = LabelEncoder()
data['Sex'] = le.fit_transform(data['Sex'])  # Encode 'Sex' (M=1, F=0)
data['ChestPainType'] = le.fit_transform(data['ChestPainType'])  # Encode chest pain type

# Filter features for the model
X = data[['Cholesterol', 'Sex']]
y = data['HeartDisease']

# Train a simple logistic regression model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Streamlit app
st.title("Heart Disease Prediction Based on Cholesterol Levels")
st.write("This app predicts the likelihood of heart disease based on cholesterol levels, with separate analysis for males and females.")

# User inputs
age = st.number_input("Enter your age:", min_value=0, max_value=120, value=30)
bp = st.number_input("Enter your resting blood pressure (mm Hg):", min_value=0, max_value=300, value=120)
chest_pain = st.selectbox("Do you experience chest pain?", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
cholesterol = st.slider("Enter your cholesterol level (mg/dL):", min_value=100, max_value=400, value=200)
sex = st.radio("Select your gender:", ["Male", "Female"])

# Encode user inputs
sex_encoded = 1 if sex == "Male" else 0
chest_pain_encoded = le.transform([chest_pain])[0]  # Encode chest pain

# Prediction
input_data = np.array([[cholesterol, sex_encoded]])
prediction = model.predict(input_data)[0]
predicted_prob = model.predict_proba(input_data)[0][1]

st.subheader("Prediction Result")
if prediction == 1:
    st.write(f"**You are likely to have heart disease. Probability: {predicted_prob:.2f}**")
else:
    st.write(f"**You are unlikely to have heart disease. Probability: {predicted_prob:.2f}**")

# Visualization
male_data = data[data['Sex'] == 1]
female_data = data[data['Sex'] == 0]

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Male plot
ax[0].scatter(male_data['Cholesterol'], male_data['HeartDisease'], alpha=0.5, label="Data")
ax[0].set_title("Male: Cholesterol vs Heart Disease")
ax[0].set_xlabel("Cholesterol")
ax[0].set_ylabel("Heart Disease")
ax[0].legend()

# Female plot
ax[1].scatter(female_data['Cholesterol'], female_data['HeartDisease'], alpha=0.5, label="Data", color="orange")
ax[1].set_title("Female: Cholesterol vs Heart Disease")
ax[1].set_xlabel("Cholesterol")
ax[1].set_ylabel("Heart Disease")
ax[1].legend()

st.pyplot(fig)