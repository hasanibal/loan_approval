import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
df = pd.read_csv("loan_data.csv")

# 7 key features
selected_features = [
    "person_age", "person_income", "person_emp_exp",
    "credit_score", "loan_amnt", "loan_int_rate",
    "previous_loan_defaults_on_file"
]

X = df[selected_features]
y = df["loan_status"]

# Identify types
numeric_cols = [
    "person_age", "person_income", "person_emp_exp",
    "credit_score", "loan_amnt", "loan_int_rate"
]
categorical_cols = ["previous_loan_defaults_on_file"]

# Preprocessing
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
])

# Models with balanced classes
rf_model = Pipeline([
    ("preprocess", preprocessor),
    ("classifier", RandomForestClassifier(class_weight="balanced", n_estimators=200, random_state=42))
])

svm_model = Pipeline([
    ("preprocess", preprocessor),
    ("classifier", SVC(class_weight="balanced", probability=True))
])

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
rf_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train)

# Evaluate
rf_pred = rf_model.predict(X_test)
svm_pred = svm_model.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
print("SVM Accuracy:", accuracy_score(y_test, svm_pred))

# Save models and feature list
joblib.dump(rf_model, "rf_model.pkl")
joblib.dump(svm_model, "svm_model.pkl")
joblib.dump(selected_features, "selected_features.pkl")

print("Training complete!")
