# 📊 Customer Churn Risk Estimator

An end-to-end Machine Learning application that predicts the likelihood of customer churn for subscription-based businesses. Built with Python, Scikit-Learn, and deployed as an interactive web application using Streamlit.

🔗 **Live Demo:** [Click here to test the live app][https://YOUR_LIVE_APP_LINK.streamlit.app](https://nikhil-churn.streamlit.app/)

---

## 🚀 Overview

Customer churn is a critical metric for businesses. This project provides a real-time risk assessment tool that helps retention teams identify customers at high risk of canceling their service, enabling proactive intervention.

### Key Features
* **Interactive UI:** Input customer details (contract type, tenure, monthly charges, payment methods) via an intuitive dashboard.
* **Real-time Prediction:** Uses a trained Machine Learning pipeline to calculate and display churn probability instantly.
* **Risk Categorization:** Clear visual indicators highlighting High Risk vs. Low Risk customers.

---

## 🛠️ Tech Stack & Tools

* **Programming Language:** Python
* **Data Processing & ML:** Pandas, NumPy, Scikit-Learn
* **Model:** Random Forest Classifier
* **Web Framework:** Streamlit
* **Version Control:** Git & GitHub

---

## 📂 Project Structure

```text
customer-churn-predictor/
│
├── train.py                  # Script to clean data, train, and save ML model
├── app.py                    # Streamlit web application script
├── churn_model.pkl           # Saved trained Random Forest model
├── encoders.pkl              # Label Encoders for categorical features
├── feature_columns.pkl       # Feature list required for model input
├── requirements.txt          # Project dependencies
└── README.md                 # Documentation
