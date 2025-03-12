import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import time
import warnings

warnings.filterwarnings('ignore')

st.title("Personal Fitness Tracker")
st.write("In this web app, you can observe your predicted calories burned.")

st.sidebar.header("User Input Parameters")

def user_input_features():
    age = st.sidebar.slider("Age", 10, 100, 30)
    bmi = st.sidebar.slider("BMI", 15.0, 40.0, 20.0)
    duration = st.sidebar.slider("Duration (min)", 0, 35, 15)
    heart_rate = st.sidebar.slider("Heart Rate", 60, 130, 80)
    body_temp = st.sidebar.slider("Body Temperature (C)", 36.0, 42.0, 38.0)
    gender_button = st.sidebar.radio("Gender", ("Male", "Female"))
    gender = 1 if gender_button == "Male" else 0
    
    data_model = {
        "Age": age,
        "BMI": bmi,
        "Duration": duration,
        "Heart_Rate": heart_rate,
        "Body_Temp": body_temp,
        "Gender_male": gender  # Gender is encoded as 1 for male, 0 for female
    }
    return pd.DataFrame(data_model, index=[0])

df = user_input_features()

st.header("Your Parameters")
st.write(df)

# Progress bar animation
latest_iteration = st.empty()
bar = st.progress(0)
for i in range(100):
    bar.progress(i + 1)
    time.sleep(0.01)

# Load and preprocess data
calories = pd.read_csv("calories.csv")
exercise = pd.read_csv("exercise.csv")
exercise_df = exercise.merge(calories, on="User_ID")
exercise_df.drop(columns=["User_ID"], inplace=True)

# Add BMI column to both training and test sets
exercise_df["BMI"] = exercise_df["Weight"] / ((exercise_df["Height"] / 100) ** 2)
exercise_df["BMI"] = round(exercise_df["BMI"], 2)

# Prepare the training and testing sets
exercise_train_data, exercise_test_data = train_test_split(exercise_df, test_size=0.2, random_state=1)

selected_columns = ["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]
exercise_train_data = exercise_train_data[selected_columns]
exercise_test_data = exercise_test_data[selected_columns]

# Convert categorical variable
exercise_train_data = pd.get_dummies(exercise_train_data, drop_first=True)
exercise_test_data = pd.get_dummies(exercise_test_data, drop_first=True)

# Separate features and labels
X_train = exercise_train_data.drop("Calories", axis=1)
y_train = exercise_train_data["Calories"]
X_test = exercise_test_data.drop("Calories", axis=1)
y_test = exercise_test_data["Calories"]

# Train the model
random_reg = RandomForestRegressor(n_estimators=1000, max_features=3, max_depth=6)
random_reg.fit(X_train, y_train)

# Align prediction data columns with training data
df = df.reindex(columns=X_train.columns, fill_value=0)

# Make prediction
prediction = random_reg.predict(df)

st.write("---")
st.header("Prediction")
st.write(f"**{round(prediction[0], 2)} kilocalories**")

# Find similar results based on predicted calories
calorie_range = [prediction[0] - 10, prediction[0] + 10]
similar_data = exercise_df[(exercise_df["Calories"] >= calorie_range[0]) & (exercise_df["Calories"] <= calorie_range[1])]
st.write("Similar results:")
st.write(similar_data.sample(5) if not similar_data.empty else "No similar results found.")

# General Information
st.write("---")
st.header("General Information")
boolean_age = (exercise_df["Age"] < df["Age"].values[0]).tolist()
boolean_duration = (exercise_df["Duration"] < df["Duration"].values[0]).tolist()
boolean_body_temp = (exercise_df["Body_Temp"] < df["Body_Temp"].values[0]).tolist()
boolean_heart_rate = (exercise_df["Heart_Rate"] < df["Heart_Rate"].values[0]).tolist()
st.write(f"You are older than {round(sum(boolean_age) / len(boolean_age) * 100, 2)}% of other people.")
