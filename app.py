import streamlit as st
import numpy as np
import pandas as pd
import joblib
import firebase_admin
from firebase_admin import credentials, firestore
import json

# Load model and scaler
model = joblib.load("svr_model.pkl")
scaler = joblib.load("scaler.pkl")

# Initialize Firebase using secrets
if not firebase_admin._apps:
    cred = credentials.Certificate(json.loads(st.secrets["firebase_key"]))
    firebase_admin.initialize_app(cred)
db = firestore.client()

# Mapping for height_diff
height_diff_map = {
    ('64-65', 0): 1.302222,  ('64-65', 1): 1.410000,
    ('65-66', 0): 2.017778,  ('65-66', 1): 0.504667,
    ('66-67', 0): 0.958750,  ('66-67', 1): 1.471312,
    ('67-68', 0): 0.853590,  ('67-68', 1): 0.746538,
    ('68-69', 0): -0.138879, ('68-69', 1): 0.200000,
    ('69-70', 0): -0.472102, ('69-70', 1): -0.204054,
    ('70-71', 0): -0.351444, ('70-71', 1): -0.153745,
    ('71-72', 0): -0.982286, ('71-72', 1): -0.415938,
    ('72-73', 0): -1.334286, ('72-73', 1): -1.413333,
    ('73-74', 0): -1.003000, ('73-74', 1): -0.181111,
    ('75-76', 0): -1.363333, ('75-76', 1): -2.230000,
}

st.title("👨‍👩‍👧 Predict Child Height")

with st.form("height_form"):
    father_ft = st.number_input("Father's Height (feet)", min_value=4, max_value=8, value=5, step=1)
    father_in = st.number_input("Father's Height (inches)", min_value=0, max_value=11, value=8, step=1)
    mother_ft = st.number_input("Mother's Height (feet)", min_value=4, max_value=8, value=5, step=1)
    mother_in = st.number_input("Mother's Height (inches)", min_value=0, max_value=11, value=4, step=1)
    gender = st.selectbox("Child Gender", ["Male", "Female"])
    child_num = st.number_input("Child Number (in family)", min_value=1, max_value=15, value=1)

    submitted = st.form_submit_button("Predict Height")

if submitted:
    father_height = father_ft * 12 + father_in
    mother_height = mother_ft * 12 + mother_in
    gender_encoded = 1 if gender == "Male" else 0
    midparent = (father_height + 1.08 * mother_height) / 2

    # Get bin
    bin_floor = int(midparent)
    bin_str = f"{bin_floor}-{bin_floor + 1}"
    height_diff = height_diff_map.get((bin_str, gender_encoded), 0)

    # Prepare features
    input_data = [[
        father_height,
        mother_height,
        midparent,
        gender_encoded,
        child_num,
        height_diff
    ]]
    input_scaled = scaler.transform(input_data)
    pred_inch = model.predict(input_scaled)[0]

    # Convert to ft + in
    pred_ft = int(pred_inch // 12)
    pred_rem_in = round(pred_inch % 12, 1)

    st.success(f"🧒 Predicted Child Height: {pred_ft} ft {pred_rem_in} in")

    # Store in Firebase
    db.collection("height_predictions").add({
        "father_height_in": father_height,
        "mother_height_in": mother_height,
        "gender": gender,
        "child_num": child_num,
        "predicted_height_in": pred_inch
    })
