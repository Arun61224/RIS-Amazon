import streamlit as st
import pandas as pd
import numpy as np

# --- Upar diya gaya 'calculate_distance_from_zips' function yahan paste karein ---
# ... (Paste the complete function 'calculate_distance_from_zips' and 'get_zip_data_df' here) ...
# Data loading ko Streamlit caching se optimize karna
@st.cache_data
def load_data(file_path):
    """Data load karta hai aur caching ka use karta hai."""
    try:
        df_raw = pd.read_csv(file_path, dtype={'Zip': str})
        return df_raw[['Zip', 'Latitude', 'Longitude']].set_index('Zip')
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def calculate_distance(zip1, zip2, df_raw):
    # Calculation logic from the Python script (with minor Streamlit adjustments)
    # ... (Implementation of spherical law of cosines goes here, similar to the function above) ...
    # Main logic
    R = 6371  # Earth radius in km
    try:
        lat1 = df_raw.loc[str(zip1), 'Latitude']
        lon1 = df_raw.loc[str(zip1), 'Longitude']
        lat2 = df_raw.loc[str(zip2), 'Latitude']
        lon2 = df_raw.loc[str(zip2), 'Longitude']
    except KeyError:
        return "Not Found" # Custom return for Streamlit error handling
    
    co_lat1 = np.radians(90 - lat1)
    co_lat2 = np.radians(90 - lat2)
    delta_lon = np.radians(lon1 - lon2)
    
    cos_c = (np.cos(co_lat1) * np.cos(co_lat2)) + \
            (np.sin(co_lat1) * np.sin(co_lat2) * np.cos(delta_lon))
    
    cos_c = np.clip(cos_c, -1, 1) # Clipping for acos stability
    angular_distance = np.arccos(cos_c)
    
    distance_km = angular_distance * R
    return distance_km

# --- Streamlit UI ---
st.title("🗺️ RIS Distance Calculator (ZIP to ZIP)")
st.caption("Calculates Great-Circle Distance (km) using Spherical Law of Cosines.")

df_zip_data = load_data("RIS checker.xlsx - RawData.csv")

if not df_zip_data.empty:
    st.sidebar.subheader("Input ZIP Codes")
    
    # User input for ZIP codes
    zip_list = df_zip_data.index.tolist()
    zip_from = st.sidebar.selectbox("Select Starting ZIP (From)", options=zip_list)
    zip_to = st.sidebar.selectbox("Select Destination ZIP (To)", options=zip_list)

    if st.sidebar.button("Calculate Distance"):
        if zip_from and zip_to:
            result = calculate_distance(zip_from, zip_to, df_zip_data)
            
            if result == "Not Found":
                st.error("One or both selected ZIP codes were not found in the raw data.")
            elif isinstance(result, (float, np.float64)):
                st.success(f"**Distance between {zip_from} and {zip_to}:**")
                st.balloons()
                st.metric(label="Distance", value=f"{result:.4f} km")
            else:
                st.error(f"An error occurred during calculation: {result}")
else:
    st.error("Could not load ZIP code data. Please check the file path and format.")

# Required dependencies file:
# requirements.txt:
# pandas
# numpy
# streamlit