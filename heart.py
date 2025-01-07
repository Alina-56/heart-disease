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

data = load_data()

if data is not None:
    # Preprocess the data
    le_sex = LabelEncoder()

    data['Sex'] = le_sex.fit_transform(data['Sex'])  # Encode 'Sex' (M=1, F=0)

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
    chest_pain = st.radio("Do you experience chest pain?", ["Yes", "No"])
    cholesterol = st.slider("Enter your cholesterol level (mg/dL):", min_value=100, max_value=400, value=200)
    sex = st.radio("Select your gender:", le_sex.classes_)

    # Encode user inputs
    try:
        sex_encoded = le_sex.transform([sex])[0]  # Encode sex
        chest_pain_encoded = 1 if chest_pain == "Yes" else 0  # Convert chest pain to binary

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

        # Line chart visualization
        cholesterol_range = np.linspace(100, 400, 50)
        probabilities = []

        for chol in cholesterol_range:
            prob = model.predict_proba([[chol, sex_encoded]])[0][1]
            probabilities.append(prob)

        line_fig, line_ax = plt.subplots(figsize=(10, 6))
        line_ax.plot(cholesterol_range, probabilities, label="Prediction Probability", color="blue")
        line_ax.set_title("Heart Disease Probability vs Cholesterol")
        line_ax.set_xlabel("Cholesterol")
        line_ax.set_ylabel("Probability of Heart Disease")
        line_ax.legend()

        st.pyplot(line_fig)

        # Simple Linear Regression Graph
        from sklearn.linear_model import LinearRegression
        slr_model = LinearRegression()
        slr_model.fit(X_train[['Cholesterol']], y_train)

        predicted_values = slr_model.predict(np.array(cholesterol_range).reshape(-1, 1))

        slr_fig, slr_ax = plt.subplots(figsize=(10, 6))
        slr_ax.scatter(X_train['Cholesterol'], y_train, alpha=0.5, label="Training Data")
        slr_ax.plot(cholesterol_range, predicted_values, label="SLR Prediction", color="red")
        slr_ax.set_title("Simple Linear Regression: Cholesterol vs Heart Disease")
        slr_ax.set_xlabel("Cholesterol")
        slr_ax.set_ylabel("Heart Disease")
        slr_ax.legend()

        st.pyplot(slr_fig)

    except Exception as e:
        st.error(f"An error occurred during prediction or visualization: {e}")
else:
    st.stop()
