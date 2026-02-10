import pandas as pd
import streamlit as st
import joblib
# Mapping dictionaries (must match training data)
sex_map = {"Male": "Male", "Female": "Female"}

cp_map = {
    "typical angina": "typical angina",
    "atypical angina": "atypical angina",
    "non-anginal pain": "non-anginal pain",
    "asymptomatic": "asymptomatic"
}

fbs_map = {"Yes": "True", "No": "False"}

restecg_map = {
    "normal": "normal",
    "abnormal": "abnormal",
    "probable": "probable"
}

exang_map = {"Yes": "True", "No": "False"}

slope_map = {
    "upsloping": "upsloping",
    "flat": "flat",
    "downsloping": "downsloping"
}

thal_map = {
    "normal": "normal",
    "fixed defect": "fixed defect",
    "reversible defect": "reversible defect"
}


# Load trained pipeline (preprocessing + model)
pipeline = joblib.load("heart_disease_pipeline.pkl")

# Page config
st.set_page_config(page_title="Heart Disease Prediction", layout="centered")

# Title
st.title(" Heart Disease Prediction App")
st.write("This app predicts whether a person has heart disease using Machine Learning.")

st.subheader("Enter Patient Details")

# Inputs
age = st.number_input("Age", min_value=1, max_value=120, value=45)
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", value=120)
chol = st.number_input("Cholesterol (mg/dl)", value=200)
thalach = st.number_input("Maximum Heart Rate Achieved", value=150)
oldpeak = st.number_input("ST Depression (oldpeak)", value=1.0)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type (cp)", ["typical angina", "atypical angina", "non-anginal pain", "asymptomatic"])
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", ["Yes", "No"])
restecg = st.selectbox("Resting ECG (restecg)", ["normal", "abnormal", "probable"])
exang = st.selectbox("Exercise Induced Angina (exang)", ["Yes", "No"])
slope = st.selectbox("Slope of Peak Exercise ST Segment", ["upsloping", "flat", "downsloping"])
ca = st.selectbox("Number of Major Vessels (ca)", [0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia (thal)", ["normal", "fixed defect", "reversible defect"])


st.markdown("---")

# Predict button
if st.button("Predict"):
    input_df = pd.DataFrame([{
        "age": age,
        "trestbps": trestbps,
        "chol": chol,
        "thalch": thalach,
        "oldpeak": oldpeak,
        "sex": sex_map[sex],
        "cp": cp_map[cp],
        "fbs": fbs_map[fbs],
        "restecg": restecg_map[restecg],
        "exang": exang_map[exang],
        "slope": slope_map[slope],
        "ca": ca,
        "thal": thal_map[thal]
    }])

    prediction = pipeline.predict(input_df)[0]

    if prediction == 1:
        st.error("⚠️ Heart Disease Detected")
    else:
        st.success("✅ No Heart Disease Detected")
