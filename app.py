import pandas as pd
import streamlit as st
import joblib

import pandas as pd
from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("heart.csv")

# Separate features and target
X = df.drop("num", axis=1)
y = df["num"]

# Define categorical and numerical columns
categorical_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
numerical_cols = ["age", "trestbps", "chol", "thalch", "oldpeak"]

# Preprocessing
from sklearn.pipeline import Pipeline as SkPipeline

numeric_transformer = SkPipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = SkPipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)


# Create pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ]
)

# Train model
pipeline.fit(X, y)

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
    st.write("Button clicked")
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

    
