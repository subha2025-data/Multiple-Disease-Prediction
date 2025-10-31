import streamlit as st
import pickle
import os
import numpy as np
import pandas as pd

# -------------------------------
# Function to load saved models
# -------------------------------
def load_model(model_path):
    if os.path.exists(model_path):
        with open(model_path, "rb") as file:
            return pickle.load(file)
    else:
        st.error(f"❌ {model_path} not found!")
        return None

# -------------------------------
# Load All Disease Models
# -------------------------------
parkinson_model = load_model("models/parkinson_model.pkl")
parkinson_scaler = load_model("models/parkinson_scaler.pkl")

liver_model = load_model("models/liver_model.pkl")
liver_scaler = load_model("models/liver_scaler.pkl")

kidney_model = load_model("models/kidney_model.pkl")
kidney_scaler = load_model("models/kidney_scaler.pkl")

# -------------------------------
# Sidebar Navigation
# -------------------------------
st.sidebar.title("🧠 Multiple Disease Prediction System")
app_mode = st.sidebar.radio(
    "📂 Choose Page:",
    ["🏠 Home", "🧠 Parkinson’s Disease", "🫀 Liver Disease", "🧪 Kidney Disease"]
)

# -------------------------------
# 🏠 HOME PAGE
# -------------------------------
if app_mode == "🏠 Home":
    st.title("🏥 Welcome to the Multiple Disease Prediction App")
    st.image(
        "https://cdn-icons-png.flaticon.com/512/4228/4228703.png",
        width=200,
    )
    st.markdown("""
        ### 👋 Hello!
        This web app helps you **predict the likelihood** of three major diseases using **Machine Learning**:
        - 🧠 **Parkinson’s Disease**
        - 🫀 **Liver Disease**
        - 🧪 **Kidney Disease**

        ---
        ### 💡 How It Works:
        1. Go to the sidebar (left).
        2. Choose the disease you want to predict.
        3. Enter the required medical details.
        4. Click **Predict** to see the result instantly!

        ---
        ### 🧬 Powered By:
        - Python 🐍
        - Scikit-Learn 🤖
        - Streamlit 🌐
        - Your efforts 💪
        ---
        """)
    st.success("Start by selecting a disease from the sidebar 👉")


# -------------------------------
# 🧠 Parkinson’s Disease Page
# -------------------------------
if app_mode == "🧠 Parkinson’s Disease":
    st.title("🧠 Parkinson’s Disease Prediction")

    # Feature inputs based on dataset
    st.subheader("Enter the following values:")

    MDVP_Fo = st.number_input("MDVP:Fo(Hz)", min_value=0.0)
    MDVP_Fhi = st.number_input("MDVP:Fhi(Hz)", min_value=0.0)
    MDVP_Flo = st.number_input("MDVP:Flo(Hz)", min_value=0.0)
    MDVP_Jitter_percent = st.number_input("MDVP:Jitter(%)", min_value=0.0)
    MDVP_Jitter_Abs = st.number_input("MDVP:Jitter(Abs)", min_value=0.0)
    MDVP_RAP = st.number_input("MDVP:RAP", min_value=0.0)
    MDVP_PPQ = st.number_input("MDVP:PPQ", min_value=0.0)
    Jitter_DDP = st.number_input("Jitter:DDP", min_value=0.0)
    MDVP_Shimmer = st.number_input("MDVP:Shimmer", min_value=0.0)
    MDVP_Shimmer_dB = st.number_input("MDVP:Shimmer(dB)", min_value=0.0)
    Shimmer_APQ3 = st.number_input("Shimmer:APQ3", min_value=0.0)
    Shimmer_APQ5 = st.number_input("Shimmer:APQ5", min_value=0.0)
    MDVP_APQ = st.number_input("MDVP:APQ", min_value=0.0)
    Shimmer_DDA = st.number_input("Shimmer:DDA", min_value=0.0)
    NHR = st.number_input("NHR", min_value=0.0)
    HNR = st.number_input("HNR", min_value=0.0)
    RPDE = st.number_input("RPDE", min_value=0.0)
    DFA = st.number_input("DFA", min_value=0.0)
    spread1 = st.number_input("spread1", min_value=0.0)
    spread2 = st.number_input("spread2", min_value=0.0)
    D2 = st.number_input("D2", min_value=0.0)
    PPE = st.number_input("PPE", min_value=0.0)

    input_data = np.array([[MDVP_Fo, MDVP_Fhi, MDVP_Flo, MDVP_Jitter_percent,
                            MDVP_Jitter_Abs, MDVP_RAP, MDVP_PPQ, Jitter_DDP,
                            MDVP_Shimmer, MDVP_Shimmer_dB, Shimmer_APQ3, 
                            Shimmer_APQ5, MDVP_APQ, Shimmer_DDA, NHR, HNR, 
                            RPDE, DFA, spread1, spread2, D2, PPE]])

    if st.button("Predict Parkinson’s Disease"):
        input_scaled = parkinson_scaler.transform(input_data)
        pred = parkinson_model.predict(input_scaled)

        if pred[0] == 1:
            st.error("⚠️ The person is likely to have Parkinson’s Disease.")
        else:
            st.success("✅ The person is Healthy.")

# -------------------------------
# 🫀 Liver Disease Page
# -------------------------------
elif app_mode == "🫀 Liver Disease":
    st.title("🫀 Liver Disease Prediction")

    Age = st.number_input("Age", min_value=1)
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Total_Bilirubin = st.number_input("Total Bilirubin", min_value=0.0)
    Direct_Bilirubin = st.number_input("Direct Bilirubin", min_value=0.0)
    Alkaline_Phosphotase = st.number_input("Alkaline Phosphotase", min_value=0.0)
    Alamine_Aminotransferase = st.number_input("Alamine Aminotransferase", min_value=0.0)
    Aspartate_Aminotransferase = st.number_input("Aspartate Aminotransferase", min_value=0.0)
    Total_Protiens = st.number_input("Total Proteins", min_value=0.0)
    Albumin = st.number_input("Albumin", min_value=0.0)
    Albumin_and_Globulin_Ratio = st.number_input("Albumin and Globulin Ratio", min_value=0.0)

    Gender = 1 if Gender == "Male" else 0

    input_data = np.array([[Age, Gender, Total_Bilirubin, Direct_Bilirubin,
                            Alkaline_Phosphotase, Alamine_Aminotransferase,
                            Aspartate_Aminotransferase, Total_Protiens,
                            Albumin, Albumin_and_Globulin_Ratio]])

    if st.button("Predict Liver Disease"):
        input_scaled = liver_scaler.transform(input_data)
        pred = liver_model.predict(input_scaled)

        if pred[0] == 1:
            st.error("⚠️ The person is likely to have Liver Disease.")
        else:
            st.success("✅ The person is Healthy.")

# -------------------------------
# 🧪 Kidney Disease Page
# -------------------------------
elif app_mode == "🧪 Kidney Disease":
    st.title("🧪 Kidney Disease Prediction")

    Age = st.number_input("Age", min_value=1)
    Blood_Pressure = st.number_input("Blood Pressure (bp)", min_value=0.0)
    Specific_Gravity = st.number_input("Specific Gravity", min_value=0.0)
    Albumin = st.number_input("Albumin", min_value=0.0)
    Sugar = st.number_input("Sugar", min_value=0.0)
    Blood_Glucose_Random = st.number_input("Blood Glucose Random", min_value=0.0)
    Blood_Urea = st.number_input("Blood Urea", min_value=0.0)
    Serum_Creatinine = st.number_input("Serum Creatinine", min_value=0.0)
    Sodium = st.number_input("Sodium", min_value=0.0)
    Potassium = st.number_input("Potassium", min_value=0.0)
    Hemoglobin = st.number_input("Hemoglobin", min_value=0.0)
    Packed_Cell_Volume = st.number_input("Packed Cell Volume", min_value=0.0)
    White_Blood_Cell_Count = st.number_input("White Blood Cell Count", min_value=0.0)
    Red_Blood_Cell_Count = st.number_input("Red Blood Cell Count", min_value=0.0)

    input_data = np.array([[Age, Blood_Pressure, Specific_Gravity, Albumin, Sugar,
                            Blood_Glucose_Random, Blood_Urea, Serum_Creatinine,
                            Sodium, Potassium, Hemoglobin, Packed_Cell_Volume,
                            White_Blood_Cell_Count, Red_Blood_Cell_Count]])

    if st.button("Predict Kidney Disease"):
        input_scaled = kidney_scaler.transform(input_data)
        pred = kidney_model.predict(input_scaled)

        if pred[0] == 1:
            st.error("⚠️ The person is likely to have Chronic Kidney Disease.")
        else:
            st.success("✅ The person is Healthy.")
