import streamlit as st
import joblib
import pandas as pd
import os

st.set_page_config(page_title="Women Safety Analytics", layout="centered")

# ------------------------------
# LOAD ALL REQUIRED PKL FILES
# ------------------------------
@st.cache_resource
def load_files():
    try:
        model = joblib.load("district_model.pkl")
        label_encoder = joblib.load("label_encoder.pkl")
        district_data = joblib.load("district_crime_data.pkl")   # dataframe from colab
        return model, label_encoder, district_data
    except Exception as e:
        st.error(f"❌ Error loading model files: {e}")
        return None, None, None

model, label_encoder, district_data = load_files()

st.title("🔒 Women Safety Analytics – District Risk Predictor")
st.markdown("Using **real crime statistics** and the **median-based ML model** from training.")

st.markdown("---")

# --------------------------------------------------------
# Generate dynamic State → District list from PKL
# --------------------------------------------------------
if district_data is not None:
    states = sorted(district_data["States/UTs"].unique())
    state = st.selectbox("Select State", states)

    districts = sorted(district_data[district_data["States/UTs"] == state]["District"].unique())
    district = st.selectbox("Select District", districts)
else:
    st.stop()

st.markdown("---")

# --------------------------------------------------------
# Predict Button
# --------------------------------------------------------
if st.button("🔍 Analyze Safety Risk", use_container_width=True):

    # Fetch REAL total crime from district_data dataframe
    row = district_data[
        (district_data["States/UTs"] == state)
        & (district_data["District"] == district)
    ]

    if row.empty:
        st.error("District not found inside the model data!")
        st.stop()

    total_crime = row["total_crime"].values[0]

    # Run ML Model
    pred_encoded = model.predict([[total_crime]])[0]
    risk_label = label_encoder.inverse_transform([pred_encoded])[0]

    st.subheader("📊 Risk Assessment Result")

    if risk_label == "High":
        st.error("🔴 **HIGH RISK AREA**")
    else:
        st.success("🟢 **LOW RISK AREA**")

    st.write(f"**State:** {state}")
    st.write(f"**District:** {district}")
    st.write(f"**Total Crimes:** {int(total_crime)}")

    # SAFETY TIPS
    st.markdown("---")
    if risk_label == "High":
        st.warning("⚠️ Recommended Safety Tips")
        tips = [
            "Avoid traveling alone late at night.",
            "Prefer well-lit and busy areas.",
            "Share live location with trusted contacts.",
            "Use verified cab/transport services.",
            "Keep emergency numbers (112) accessible.",
            "Stay alert in isolated areas."
        ]
        for t in tips:
            st.markdown(f"- {t}")
    else:
        st.success("This area is relatively safe. Continue normal precautions.")
