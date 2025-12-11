import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
from rapidfuzz import process, fuzz
import geopandas as gpd

st.set_page_config(page_title="Women Safety Analytics", layout="centered")

# ------------------------------
# LOAD MODEL + DATASET
# ------------------------------
@st.cache_resource
def load_files():
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(BASE_DIR, "women_safety_rf_model.pkl")
        data_path = os.path.join(BASE_DIR, "district_risk_dataset_sorted.csv")

        model = joblib.load(model_path)
        df = pd.read_csv(data_path)

        # Normalize for consistent matching
        df["STATE"] = df["STATE"].str.upper()
        df["DISTRICT"] = df["DISTRICT"].str.upper()

        return model, df
    except Exception as e:
        st.error(f"❌ Error loading model or dataset: {e}")
        return None, None

model, df = load_files()

# ------------------------------
# PAGE HEADER
# ------------------------------
st.title("🔒 Women Safety Analytics – District Risk Predictor")
st.markdown("Using **NCRB crime data** + **Machine Learning** to assess women's safety risk.")
st.markdown("---")

# ------------------------------
# STATE & DISTRICT DROPDOWNS
# ------------------------------
if df is not None:
    states = sorted(df["STATE"].unique())
    state = st.selectbox("Select State", states)

    districts = sorted(df[df["STATE"] == state]["DISTRICT"].unique())
    district = st.selectbox("Select District", districts)
else:
    st.stop()

st.markdown("---")

# ------------------------------
# PREDICT BUTTON
# ------------------------------
if st.button("🔍 Analyze Safety Risk", use_container_width=True):

    row = df[(df["STATE"] == state) & (df["DISTRICT"] == district)]
    
    if row.empty:
        st.error("❌ District not found in dataset.")
        st.stop()

    row = row.iloc[0]

    # ML Input
    X_input = np.array([
        row["rape_avg"],
        row["kidnapping_avg"],
        row["assault_avg"],
        row["domestic_avg"],
        row["dowry_avg"],
        row["trafficking_avg"],
        row["other_ipc_avg"]
    ]).reshape(1, -1)

    pred = model.predict(X_input)[0]
    risk_label = "High" if pred == 1 else "Low"

    # --------------------------
    # DISPLAY RISK RESULT
    # --------------------------
    st.subheader("📊 Risk Assessment Result")

    if risk_label == "High":
        st.error("🔴 **HIGH RISK AREA**")
    else:
        st.success("🟢 **LOW RISK AREA**")

    st.write(f"**State:** {state}")
    st.write(f"**District:** {district}")
    st.write(f"**Risk Level:** {risk_label}")

    # --------------------------
    # CRIME BREAKDOWN TABLE
    # --------------------------
    st.markdown("### 📉 Crime Breakdown (Average per Year)")

    crime_data = {
        "Crime Type": [
            "Rape Cases (avg)",
            "Kidnapping (avg)",
            "Assault (avg)",
            "Domestic Violence (avg)",
            "Dowry Deaths (avg)",
            "Trafficking (avg)",
            "Other IPC Crimes (avg)",
            "Total Crime Avg"
        ],
        "Value": [
            round(row['rape_avg'], 2),
            round(row['kidnapping_avg'], 2),
            round(row['assault_avg'], 2),
            round(row['domestic_avg'], 2),
            round(row['dowry_avg'], 2),
            round(row['trafficking_avg'], 2),
            round(row['other_ipc_avg'], 2),
            round(row['total_crime_avg'], 2)
        ]
    }

    crime_df = pd.DataFrame(crime_data)
    st.dataframe(crime_df, use_container_width=True)

    st.markdown("---")

    # --------------------------
    # SAFETY TIPS
    # --------------------------
    if risk_label == "High":
        st.warning("⚠️ Recommended Safety Tips")
        tips = [
            "Avoid traveling alone late at night.",
            "Prefer well-lit and crowded places.",
            "Share your live location with trusted contacts.",
            "Use verified cab/transport services only.",
            "Keep emergency numbers (112) accessible.",
            "Stay alert in isolated or unsafe areas."
        ]
        for t in tips:
            st.markdown(f"- {t}")
    else:
        st.success("This district is relatively safer. Continue general precautions.")

# ------------------------------
# OPTIONAL: DISTRICT-WISE VISUALIZATION
# OUTSIDE PREDICTION BLOCK
# ------------------------------
# OPTIONAL: RISK SCATTER PLOT (VISUALIZATION)
# ------------------------------
st.markdown("---")
with st.expander("📊 View District Risk Levels (Scatter Plot)"):

    selected_state = st.selectbox("Select a State to Visualize", sorted(df["STATE"].unique()), key="viz_state_scatter")

    # Filter for selected state
    state_df = df[df["STATE"] == selected_state].copy()

    # Numeric risk (High=1, Low=0)
    state_df["risk_num"] = state_df["risk_label"].apply(lambda x: 1 if x == "High" else 0)

    # Sort districts (High first)
    state_df = state_df.sort_values(by="risk_num", ascending=False)

    # Colors
    state_df["color"] = state_df["risk_label"].apply(lambda x: "red" if x == "High" else "green")

    plt.figure(figsize=(14, 5))
    plt.scatter(state_df["DISTRICT"], state_df["risk_num"], s=200, c=state_df["color"], alpha=0.8, edgecolor="black")

    plt.xticks(rotation=90)
    plt.yticks([0, 1], ["Low Risk", "High Risk"])
    plt.xlabel("Districts")
    plt.title(f"Women-Safety Risk Levels Across Districts in {selected_state.title()}")
    plt.grid(axis="y", linestyle="--", alpha=0.3)

    st.pyplot(plt)

    st.markdown("""
    **🔴 Red Dot = High ML-predicted risk**  
    **🟢 Green Dot = Low ML-predicted risk**  
    """)

    # ------------------------------
# DISTRICT RISK MAP
# ------------------------------
st.markdown("---")
st.header("🗺️ District Risk Map")

# Load GeoJSON once
@st.cache_resource
def load_geojson():
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        geojson_path = os.path.join(BASE_DIR, "india_districts.geojson")
        return gpd.read_file(geojson_path)
    except Exception:
        return None

geo = load_geojson()

if geo is None:
    st.error("⚠️ `india_districts.geojson` not found. Map cannot be displayed.")
else:
    # Clean geo district names
    geo["GEO_DIST_CLEAN"] = geo["NAME_2"].str.upper().str.replace("[^A-Z ]", "", regex=True)
    geo["GEO_STATE_CLEAN"] = geo["NAME_1"].str.upper().str.replace("[^A-Z ]", "", regex=True)

    # --- USER SELECTS STATE ---
    # Use a unique key to avoid conflict with other selectboxes
    selected_state_map = st.selectbox("Select State for Map", sorted(df["STATE"].unique()), key="map_state_selector")

    # Filter ML data for that state
    ml_state = df[df["STATE"] == selected_state_map].copy()
    ml_state["DIST_CLEAN"] = ml_state["DISTRICT"].str.upper().str.replace("[^A-Z ]", "", regex=True)

    # Filter GeoJSON for that state
    geo_state = geo[geo["GEO_STATE_CLEAN"] == selected_state_map].copy()

    if geo_state.empty:
        st.warning(f"No map data found for state: {selected_state_map}")
    else:
        # Prepare list for matching
        geo_names = list(geo_state["GEO_DIST_CLEAN"])

        # Perform fuzzy matching LIVE
        matches = []
        for d in ml_state["DIST_CLEAN"].unique():
            best = process.extractOne(d, geo_names, scorer=fuzz.WRatio)
            if best:
                matches.append([d, best[0], best[1]])

        match_df = pd.DataFrame(matches, columns=["ML_DIST", "GEO_DIST", "SCORE"])

        # Merge ML data with Geo data
        merged = ml_state.merge(match_df, left_on="DIST_CLEAN", right_on="ML_DIST", how="left")
        merged = merged.merge(geo_state, left_on="GEO_DIST", right_on="GEO_DIST_CLEAN", how="left")
        
        # Ensure we have a risk_label column (fallback if missing in CSV)
        if "risk_label" not in merged.columns:
            # Simple heuristic or placeholder if model prediction is complex to batch here
            # Assuming 'risk_label' might be missing if CSV structure changed
             merged["risk_label"] = "Low" # Fallback

        # Create folium map centered on state
        if not geo_state.geometry.centroid.empty:
            state_center = [geo_state.geometry.centroid.y.mean(), geo_state.geometry.centroid.x.mean()]
            m = folium.Map(location=state_center, zoom_start=6)

            # Add district markers
            for _, row in merged.iterrows():
                if pd.isna(row.geometry):
                    continue

                if hasattr(row.geometry, 'centroid'):
                    centroid = row.geometry.centroid
                    color = "red" if row.get("risk_label", "Low") == "High" else "green"

                    folium.CircleMarker(
                        location=[centroid.y, centroid.x],
                        radius=5,
                        color=color,
                        fill=True,
                        fill_opacity=0.8,
                        popup=f"{row['DISTRICT']} — {row.get('risk_label', 'Unknown')} Risk",
                    ).add_to(m)

            st_folium(m, width=800, height=600)
