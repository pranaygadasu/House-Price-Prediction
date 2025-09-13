# app/app.py
import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="House Price Prediction", layout="centered")

# -----------------------------
# Load model + metadata
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))           # points to app/ folder
MODEL_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "models", "house_price_pipeline.pkl"))

# attempt to load
try:
    artifact = joblib.load(MODEL_PATH)
except FileNotFoundError:
    st.error(f"Model file not found at: {MODEL_PATH}. Run train.py to create it.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model file: {e}")
    st.stop()

# Support both new artifact format and older plain-pipeline format:
if isinstance(artifact, dict) and "pipeline" in artifact:
    model = artifact["pipeline"]
    feature_names = artifact["feature_names"]
    numeric_features = artifact.get("numeric_features", [])
    categorical_features = artifact.get("categorical_features", [])
else:
    # fallback: the file is a plain pipeline (older). We need feature names to make a full-row.
    model = artifact
    # try to grab feature names if possible
    if hasattr(model, "feature_names_in_"):
        feature_names = list(model.feature_names_in_)
        numeric_features = []   # unknown — but it's ok; missing numeric_features just controls typing
        categorical_features = []
    else:
        st.error(
            "The saved model does not include the feature list. "
            "Please re-run train.py (updated version) so the app can load metadata."
        )
        st.stop()

# -----------------------------
# Streamlit UI: inputs (expand as needed)
# -----------------------------
st.title("🏠 House Price Prediction App")
st.write("Enter house details and click *Predict Price*.")

# Basic inputs (add more controls as needed)
year_built = st.number_input("Year Built", min_value=1800, max_value=2025, value=2000)
yr_sold = st.number_input("Year Sold", min_value=1900, max_value=2025, value=2010)
overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 5)
gr_liv_area = st.number_input("Above Ground Living Area (sq ft)", min_value=200, max_value=10000, value=1500)

house_age = yr_sold - year_built

# -----------------------------
# Build a full-row DataFrame matching training columns
# -----------------------------
# Create a single-row DataFrame with the exact columns used for training
input_template = pd.DataFrame(columns=feature_names)
# set a single row of NaNs
input_template.loc[0] = np.nan

# Fill the known inputs into the template (only if those columns exist)
mappings = {
    "YearBuilt": year_built,
    "YrSold": yr_sold,
    "OverallQual": overall_qual,
    "GrLivArea": gr_liv_area,
    "HouseAge": house_age
}

for col, val in mappings.items():
    if col in input_template.columns:
        input_template.at[0, col] = val

# Ensure numeric columns are numeric dtype (coerce where possible)
if isinstance(numeric_features, (list, tuple)):
    numeric_present = [c for c in numeric_features if c in input_template.columns]
    if numeric_present:
        input_template[numeric_present] = input_template[numeric_present].apply(pd.to_numeric, errors="coerce")

# Friendly note
st.info("Missing fields will be imputed using the values learned during training.")

# Show the (hidden) input dataframe if user wants to inspect
with st.expander("Show input data being sent to model"):
    st.dataframe(input_template.T, use_container_width=True)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Price"):
    try:
        pred = model.predict(input_template)[0]
        st.success(f"💰 Predicted Sale Price: ${pred:,.2f}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")