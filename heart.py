import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the dataset
def load_data():
    try:
        return pd.read_csv('heart.csv')
    except FileNotFoundError:
        st.error("The dataset 'heart.csv' was not found. Please ensure it is in the same directory as this script.")
        return None

# Function to remove outliers based on IQR
def remove_outliers(data):
    Q1 = data['Cholesterol'].quantile(0.25)
    Q3 = data['Cholesterol'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_data = data[(data['Cholesterol'] >= lower_bound) & (data['Cholesterol'] <= upper_bound)]
    return filtered_data

data = load_data()

if data is not None:
    # Preprocess the data
    le_sex = LabelEncoder()

    data['Sex'] = le_sex.fit_transform(data['Sex'])  # Encode 'Sex' (M=1, F=0)

    # Remove outliers
    data = remove_outliers(data)

    # Separate datasets for males and females
    male_data = data[data['Sex'] == 1]
    female_data = data[data['Sex'] == 0]

    # Train models for males and females
    male_X = male_data[['Cholesterol']]
    male_y = male_data['HeartDisease']
    female_X = female_data[['Cholesterol']]
    female_y = female_data['HeartDisease']

    male_model = LogisticRegression()
    female_model = LogisticRegression()

    male_model.fit(male_X, male_y)
    female_model.fit(female_X, female_y)

    # Streamlit app
    st.title("Heart Disease Prediction Based on Cholesterol Levels")
    st.write("This app predicts the likelihood of heart disease based on cholesterol levels, with separate analysis for males and females.")

    # User inputs
    age = st.number_input("Enter your age:", min_value=0, max_value=120, value=30)
    bp = st.number_input("Enter your resting blood pressure (mm Hg):", min_value=0, max_value=300, value=120)
    chest_pain = st.radio("Do you experience chest pain?", ["Yes", "No"])
    cholesterol = st.slider("Enter your cholesterol level (mg/dL):", min_value=100, max_value=400, value=200)
    sex = st.radio("Select your gender:", le_sex.classes_)

    # Encode user inputs
    try:
        sex_encoded = le_sex.transform([sex])[0]  # Encode sex
        chest_pain_encoded = 1 if chest_pain == "Yes" else 0  # Convert chest pain to binary

        # Select model based on gender
        if sex_encoded == 1:
            model = male_model
            gender_data = male_data
            gender = "Male"
        else:
            model = female_model
            gender_data = female_data
            gender = "Female"

        # Prediction
        input_data = np.array([[cholesterol]])
        prediction = model.predict(input_data)[0]
        predicted_prob = model.predict_proba(input_data)[0][1]

        st.subheader("Prediction Result")
        if prediction == 1:
            st.write(f"**As a {gender}, you are likely to have heart disease. Probability: {predicted_prob:.2f}**")
        else:
            st.write(f"**As a {gender}, you are unlikely to have heart disease. Probability: {predicted_prob:.2f}**")

        # Visualization for the selected gender using Logistic Regression (for probabilities)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(gender_data['Cholesterol'], gender_data['HeartDisease'], alpha=0.5, label="Data")
        cholesterol_range = np.linspace(100, 400, 50).reshape(-1, 1)
        probabilities = model.predict_proba(cholesterol_range)[:, 1]  # Get probabilities from logistic regression
        ax.plot(cholesterol_range, probabilities, label="Prediction Probability", color="red")
        ax.set_title(f"{gender}: Cholesterol vs Heart Disease Probability (Logistic Regression)")
        ax.set_xlabel("Cholesterol")
        ax.set_ylabel("Heart Disease Probability")
        ax.legend()

        st.pyplot(fig)

        # Simple Linear Regression Graph for the binary outcome
        from sklearn.linear_model import LinearRegression
        slr_model = LinearRegression()
        slr_model.fit(gender_data[['Cholesterol']], gender_data['HeartDisease'])

        predicted_values = slr_model.predict(cholesterol_range)

        slr_fig, slr_ax = plt.subplots(figsize=(10, 6))
        slr_ax.scatter(gender_data['Cholesterol'], gender_data['HeartDisease'], alpha=0.5, label="Data")
        slr_ax.plot(cholesterol_range, predicted_values, label="SLR Prediction", color="blue")
        slr_ax.set_title(f"Simple Linear Regression ({gender}): Cholesterol vs Heart Disease (SLR)")
        slr_ax.set_xlabel("Cholesterol")
        slr_ax.set_ylabel("Heart Disease (0 or 1)")
        slr_ax.legend()

        st.pyplot(slr_fig)

    except Exception as e:
        st.error(f"An error occurred during prediction or visualization: {e}")
else:
    st.stop()
