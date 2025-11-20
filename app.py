import streamlit as st
import pandas as pd
import joblib

# Load models and selected features
rf_model = joblib.load("rf_model.pkl")
svm_model = joblib.load("svm_model.pkl")
selected_features = joblib.load("selected_features.pkl")

st.title("🏦 Loan Approval Prediction App")
st.write("Enter applicant details to predict loan approval.")

# ------------------------
# User Inputs (7 key features)
# ------------------------
person_age = st.number_input("Age", min_value=18, max_value=100, value=30)
person_income = st.number_input("Annual Income", min_value=0, value=50000)
person_emp_exp = st.number_input("Employment Experience (years)", min_value=0, value=5)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
loan_amnt = st.number_input("Loan Amount Requested", min_value=0, value=10000)
loan_int_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=13.0)
previous_loan_defaults_on_file = st.selectbox("Previous Loan Default", ["Y", "N"])

# ------------------------
# Prepare input dataframe
# ------------------------
input_data = pd.DataFrame([[
    person_age, person_income, person_emp_exp, credit_score,
    loan_amnt, loan_int_rate, previous_loan_defaults_on_file
]], columns=selected_features)

# ------------------------
# Predict button
# ------------------------
if st.button("Predict Loan Approval"):
    rf_pred = rf_model.predict(input_data)[0]
    rf_prob = rf_model.predict_proba(input_data)[0][1]

    svm_pred = svm_model.predict(input_data)[0]
    svm_prob = svm_model.predict_proba(input_data)[0][1]

    st.subheader("Predictions:")
    st.write(f"Random Forest Prediction: **{rf_pred}** (Probability Approved: {rf_prob:.2f})")
    st.write(f"SVM Prediction: **{svm_pred}** (Probability Approved: {svm_prob:.2f})")
