import streamlit as st
import pandas as pd
import joblib

# Load trained assets
model = joblib.load('churn_model.pkl')
encoders = joblib.load('encoders.pkl')
feature_columns = joblib.load('feature_columns.pkl')

st.title("📊 Customer Churn Risk Estimator")
st.write("Input customer details to predict churn likelihood.")

# Form inputs
tenure = st.slider("Tenure (Months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges ($)", value=65.0)
total_charges = st.number_input("Total Charges ($)", value=tenure * monthly_charges)

contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment_method = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])

# Create default row with sample values matching trained features
input_data = {col: 0 for col in feature_columns}
input_data['tenure'] = tenure
input_data['MonthlyCharges'] = monthly_charges
input_data['TotalCharges'] = total_charges

# Encode selected inputs
if 'Contract' in encoders:
    input_data['Contract'] = encoders['Contract'].transform([contract])[0]
if 'InternetService' in encoders:
    input_data['InternetService'] = encoders['InternetService'].transform([internet_service])[0]
if 'PaymentMethod' in encoders:
    input_data['PaymentMethod'] = encoders['PaymentMethod'].transform([payment_method])[0]

# Prediction
if st.button("Predict Churn Risk"):
    input_df = pd.DataFrame([input_data])
    probability = model.predict_proba(input_df)[0][1] * 100
    
    st.markdown("---")
    if probability > 50:
        st.error(f"⚠️ High Risk: Customer has a **{probability:.1f}%** chance of churning.")
    else:
        st.success(f"✅ Low Risk: Customer has a **{probability:.1f}%** chance of churning.")