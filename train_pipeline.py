import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("heart.csv")

# Create binary target
df["target"] = df["num"].apply(lambda x: 0 if x == 0 else 1)

# Drop unnecessary columns
df = df.drop(["id", "dataset", "num"], axis=1)

# Separate features and target
X = df.drop("target", axis=1)
y = df["target"]

# Identify column types
categorical_features = [
    "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"
]

numerical_features = [
    "age", "trestbps", "chol", "thalch", "oldpeak"
]

# Preprocessing for numerical data
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Full pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train pipeline
pipeline.fit(X_train, y_train)

# Save pipeline
joblib.dump(pipeline, "heart_disease_pipeline.pkl")

print("✅ Pipeline trained and saved as heart_disease_pipeline.pkl")
