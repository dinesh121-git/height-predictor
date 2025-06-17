import streamlit as st
import numpy as np
import joblib
import os
import firebase_admin
from firebase_admin import credentials, firestore

# --- Firebase Initialization using Environment Variables ---

# Retrieve the private key from the environment variable.
firebase_private_key = os.environ.get("FIREBASE_PRIVATE_KEY")
if firebase_private_key is None:
    raise ValueError("FIREBASE_PRIVATE_KEY environment variable not set")

# If the key contains literal "\n", replace those with actual newline characters.
if "\\n" in firebase_private_key:
    firebase_private_key = firebase_private_key.replace("\\n", "\n")

# Build the Firebase configuration dictionary.
firebase_config = {
    "type": os.environ.get("FIREBASE_TYPE"),
    "project_id": os.environ.get("FIREBASE_PROJECT_ID"),
    "private_key_id": os.environ.get("FIREBASE_PRIVATE_KEY_ID"),
    "private_key": firebase_private_key,
    "client_email": os.environ.get("FIREBASE_CLIENT_EMAIL"),
    "client_id": os.environ.get("FIREBASE_CLIENT_ID"),
    "auth_uri": os.environ.get("FIREBASE_AUTH_URI"),
    "token_uri": os.environ.get("FIREBASE_TOKEN_URI"),
    "auth_provider_x509_cert_url": os.environ.get("FIREBASE_AUTH_PROVIDER_CERT_URL"),
    "client_x509_cert_url": os.environ.get("FIREBASE_CLIENT_CERT_URL")
}

# Initialize Firebase only once.
if not firebase_admin._apps:
    cred = credentials.Certificate(firebase_config)
    firebase_admin.initialize_app(cred)
db = firestore.client()

# --- Load pre-trained model and scaler ---
model = joblib.load("svr_model.pkl")
scaler = joblib.load("scaler.pkl")

# --- Streamlit App Title ---
st.title("👨‍👩‍👧 Predict Child Height")

# --- Initialize session state variables ---
if "predicted_inch" not in st.session_state:
    st.session_state.predicted_inch = None
if "input_data" not in st.session_state:
    st.session_state.input_data = {}
if "show_actual_input" not in st.session_state:
    st.session_state.show_actual_input = False

# --- Helper function to reset the app state ---
def reset_app():
    st.session_state.predicted_inch = None
    st.session_state.input_data = {}
    st.session_state.show_actual_input = False

# --- Input Form: Only show if no prediction exists ---
if st.session_state.predicted_inch is None:
    with st.form("height_form"):
        father_ft = st.number_input("Father's Height (feet)", min_value=4, max_value=8, step=1)
        father_in = st.number_input("Father's Height (inches)", min_value=0, max_value=11)
        mother_ft = st.number_input("Mother's Height (feet)", min_value=4, max_value=8, step=1)
        mother_in = st.number_input("Mother's Height (inches)", min_value=0, max_value=11)
        gender = st.selectbox("Child Gender", ["Male", "Female"])
        child_num = st.number_input("Child Number", min_value=1, max_value=15, step=1)
        submit = st.form_submit_button("Predict Height")
    
    if submit:
        # Convert parent's heights from feet and inches to total inches.
        father_height = father_ft * 12 + father_in
        mother_height = mother_ft * 12 + mother_in
        gender_encoded = 1 if gender == "Male" else 0
        midparent = (father_height + 1.08 * mother_height) / 2

        # --- Height Difference Mapping ---
        height_diff_map = {
            ('64-65', 0): 1.302222, ('64-65', 1): 1.41,
            ('65-66', 0): 2.017778, ('65-66', 1): 0.504667,
            ('66-67', 0): 0.95875,   ('66-67', 1): 1.471312,
            ('67-68', 0): 0.85359,   ('67-68', 1): 0.746538,
            ('68-69', 0): -0.138879, ('68-69', 1): 0.2,
            ('69-70', 0): -0.472102, ('69-70', 1): -0.204054,
            ('70-71', 0): -0.351444, ('70-71', 1): -0.153745,
            ('71-72', 0): -0.982286, ('71-72', 1): -0.415938,
            ('72-73', 0): -1.334286, ('72-73', 1): -1.413333,
            ('73-74', 0): -1.003,    ('73-74', 1): -0.181111,
            ('75-76', 0): -1.363333, ('75-76', 1): -2.23,
        }

        # Create a bin string (e.g., "64-65") based on midparent height.
        bin_floor = int(midparent)
        bin_str = f"{bin_floor}-{bin_floor + 1}"
        height_diff = height_diff_map.get((bin_str, gender_encoded), 0)

        # Prepare features and scale them.
        features = [[father_height, mother_height, midparent, gender_encoded, child_num, height_diff]]
        scaled_features = scaler.transform(features)
        predicted_inch = model.predict(scaled_features)[0]

        # Save the predicted height and input details in the session state.
        st.session_state.predicted_inch = predicted_inch
        st.session_state.input_data = {
            "father_height_in": father_height,
            "mother_height_in": mother_height,
            "gender": gender,
            "child_num": child_num
        }

# --- Display Prediction and User Options ---
if st.session_state.predicted_inch is not None:
    pred_inch = st.session_state.predicted_inch
    pred_ft = int(pred_inch // 12)
    pred_in = round(pred_inch % 12, 1)
    st.subheader(f"Predicted Height: {pred_ft} ft {pred_in} in")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Yes, this is accurate"):
            try:
                # Save prediction data to a Firebase collection.
                doc_ref = db.collection("child_heights_collection").add({
                    **st.session_state.input_data,
                    "predicted_height_in": pred_inch
                })
                st.success("✅ Height saved successfully.")
                reset_app()
            except Exception as e:
                st.error("Error saving data: " + str(e))
    with col2:
        if st.button("❌ No, enter actual height"):
            st.session_state.show_actual_input = True

    if st.session_state.show_actual_input:
        st.info("Please enter your actual height below:")
        with st.form("actual_height_form"):
            act_ft = st.number_input("Actual Height (feet)", min_value=3, max_value=8, step=1, key="actual_ft")
            act_in = st.number_input("Actual Height (inches)", min_value=0, max_value=11, step=1, key="actual_in")
            submitted_actual = st.form_submit_button("Submit Actual Height")
        if submitted_actual:
            actual_total = act_ft * 12 + act_in
            try:
                doc_ref = db.collection("child_heights_collection").add({
                    **st.session_state.input_data,
                    "predicted_height_in": actual_total
                })
                st.success("📥 Actual height saved successfully.")
                reset_app()
            except Exception as e:
                st.error("Error saving data: " + str(e))

