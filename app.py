import streamlit as st
import joblib
import numpy as np

model = joblib.load('heart_disease_model.pkl')

st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

st.markdown("""
    <style>
        body {
            background: linear-gradient(to right, #6a11cb, #2575fc);
            color: white;
            font-family: 'Arial', sans-serif;
        }
        .stButton button {
            background-color: #2575fc;
            color: white;
            font-size: 16px;
            border-radius: 8px;
            padding: 10px 20px;
        }
        .stButton button:hover {
            background-color: #6a11cb;
        }
        .stTextInput, .stNumberInput, .stSelectbox {
            background-color: rgba(255, 255, 255, 0.2);
            color: white;
            border: 1px solid #ffffff;
            border-radius: 8px;
            padding: 10px;
        }
        .stTextInput input, .stNumberInput input, .stSelectbox select {
            background-color: transparent;
            color: black;  /* Changed text color to black */
        }
        .stMarkdown, .stSubheader, .stTitle {
            color: #fff;
        }
        .stProgress bar {
            background-color: #2575fc;
        }
        .stSpinner {
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Heart Disease Prediction Web App")

st.sidebar.header("Enter Patient Details")

age = st.sidebar.number_input("Age", min_value=0, max_value=100, step=1)
chest_pain = st.sidebar.selectbox("Chest Pain Type", options=[0, 1, 2, 3])
resting_bp = st.sidebar.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=250)
cholesterol = st.sidebar.number_input("Cholesterol Level (mg/dL)", min_value=0, max_value=500)
exercise_angina = st.sidebar.selectbox("Exercise-Induced Angina (1: Yes, 0: No)", options=[0, 1])
st_depression = st.sidebar.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0)
max_heart_rate = st.sidebar.number_input("Maximum Heart Rate Achieved", min_value=50, max_value=250)
num_vessels = st.sidebar.selectbox("Number of Major Vessels Colored by Fluoroscopy", options=[0, 1, 2, 3])
thalassemia = st.sidebar.selectbox("Thalassemia (1: Normal, 2: Fixed Defect, 3: Reversible Defect)", options=[1, 2, 3])

predict_button = st.sidebar.button("Predict")

input_data = np.array([[age, chest_pain, resting_bp, cholesterol, exercise_angina, st_depression, max_heart_rate, num_vessels, thalassemia]])

if predict_button:
    with st.spinner('Making prediction...'):
        y_pred_prob = model.predict_proba(input_data)[0, 1]
        threshold = 0.6
        adjusted_pred = (y_pred_prob >= threshold).astype(int)

        if y_pred_prob > 0.8:
            risk_category = "High Risk"
        elif y_pred_prob > 0.5:
            risk_category = "Medium Risk"
        else:
            risk_category = "Low Risk"

        st.success("Prediction Complete!")
        st.subheader(f"Probability of having heart disease: {y_pred_prob * 100:.2f}%")
        st.subheader(f"Adjusted Prediction (Threshold {threshold}): {'Heart Disease' if adjusted_pred == 1 else 'No Heart Disease'}")
        st.subheader(f"Risk Category: {risk_category}")

        st.progress(y_pred_prob)

        if risk_category == "High Risk":
            st.warning("You are at high risk for heart disease. Please consult a doctor immediately.")
        elif risk_category == "Medium Risk":
            st.info("You are at medium risk. It's advisable to undergo further tests.")
        else:
            st.success("You are at low risk. Maintain a healthy lifestyle!")

        st.markdown("### Model Insights")
        st.markdown("This model uses various factors such as age, cholesterol levels, and exercise-induced angina to predict heart disease.")
else:
    st.info("Please fill in the details on the sidebar to predict heart disease.")
