import streamlit as st
import pandas as pd
import numpy as np
import io

# --- 1. Distance Calculation Logic (Same as before) ---

def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Do set of coordinates ke beech ki distance (in kilometers) calculate karta hai 
    Spherical Law of Cosines ka use karke.
    """
    R = 6371  # Earth ka radius kilometers mein
    
    # Degrees ko Radians mein convert karna aur co-latitude (90-lat) nikalna
    co_lat1 = np.radians(90 - lat1)
    co_lat2 = np.radians(90 - lat2)
    delta_lon = np.radians(lon1 - lon2)
    
    # Spherical Law of Cosines ka core calculation
    cos_c = (np.cos(co_lat1) * np.cos(co_lat2)) + \
            (np.sin(co_lat1) * np.sin(co_lat2) * np.cos(delta_lon))
    
    # Numerical stability ke liye value ko [-1, 1] range mein clip karna
    cos_c = np.clip(cos_c, -1.0, 1.0)

    # Distance calculation: ACOS(cos_c) * R
    angular_distance = np.arccos(cos_c)
    distance_km = angular_distance * R
    
    return distance_km

# --- 2. Data Loading and Caching ---

@st.cache_data
def load_raw_data(file_path):
    """ZIP data (Latitude/Longitude mapping) load karta hai aur caching ka use karta hai."""
    try:
        # File name wahi rakhein: "RIS checker.xlsx - RawData.csv"
        df_raw = pd.read_csv(file_path, dtype={'Zip': str})
        
        # Sirf relevant columns rakhna aur 'Zip' ko index banana (VLOOKUP ka equivalent)
        df_raw = df_raw[['Zip', 'Latitude', 'Longitude']].set_index('Zip')
        return df_raw
    except Exception as e:
        # Agar yeh file server par na mile toh error dikhana
        st.error(f"❌ Critical Error: Could not load the ZIP-Lat/Lon mapping file ('{file_path}'). Please ensure it is present.")
        return pd.DataFrame()

# --- 3. Main Streamlit Application UI ---

st.set_page_config(page_title="Bulk RIS Calculator", layout="wide")

st.title("📦 Bulk RIS Distance Calculator")
st.markdown("Upload your order file (CSV/Excel) to calculate distances between Origin and Destination Pincodes.")

# Server par rakhi hui Raw Data file load karna
RAW_DATA_PATH = "RIS checker.xlsx - RawData.csv"
df_zip_data = load_raw_data(RAW_DATA_PATH)

if df_zip_data.empty:
    st.stop() # Agar Raw Data load nahi hui toh aage nahi badhenge

# File Uploader
uploaded_file = st.file_uploader(
    "Upload your Order Data File (CSV or Excel) - Expected columns: 'Origin Pincode', 'Destination Pincode'", 
    type=['csv', 'xlsx']
)

if uploaded_file is not None:
    # File type ke according data read karna
    try:
        if uploaded_file.name.endswith('.csv'):
            df_orders = pd.read_csv(uploaded_file, dtype={'Origin Pincode': str, 'Destination Pincode': str})
        else: # Excel (.xlsx) file
            df_orders = pd.read_excel(uploaded_file, dtype={'Origin Pincode': str, 'Destination Pincode': str})

    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
        st.stop()
        
    st.subheader("1. Uploaded Order Data (First 5 Rows)")
    st.dataframe(df_orders.head())

    # Required columns check karna
    required_cols = ['Origin Pincode', 'Destination Pincode']
    if not all(col in df_orders.columns for col in required_cols):
        st.error(f"❌ Error: The uploaded file must contain the columns: {required_cols}")
        st.stop()
        
    st.subheader(f"2. Calculating RIS Distance for {len(df_orders)} Orders...")

    # DataFrames ko merge karne ke liye functions use karte hain (VLOOKUP ka bulk equivalent)
    with st.spinner('Matching Pincodes and calculating distances...'):
        
        # Origin Pincode ki Lat/Lon fetch karna
        df_orders = df_orders.merge(
            df_zip_data, 
            left_on='Origin Pincode', 
            right_index=True, 
            how='left', 
            suffixes=('_Origin', '_Dest') # Suffixes abhi kaam nahi aayenge, sirf Destination merge mein
        ).rename(columns={'Latitude': 'Lat_Origin', 'Longitude': 'Lon_Origin'})

        # Destination Pincode ki Lat/Lon fetch karna
        df_orders = df_orders.merge(
            df_zip_data, 
            left_on='Destination Pincode', 
            right_index=True, 
            how='left', 
            suffixes=('_Origin', '_Dest')
        ).rename(columns={'Latitude': 'Lat_Dest', 'Longitude': 'Lon_Dest'})


        # Distance Calculate karna (Pandas apply function ka use karke)
        df_orders['RIS_Distance_KM'] = df_orders.apply(lambda row: 
            calculate_distance(
                row['Lat_Origin'], row['Lon_Origin'], 
                row['Lat_Dest'], row['Lon_Dest']
            ) 
            if pd.notna(row['Lat_Origin']) and pd.notna(row['Lat_Dest'])
            else 'PINCODE_NOT_FOUND', axis=1
        )
        
    
    # Final Result Display karna
    st.subheader("3. Final Result")
    
    # Columns ko re-order karna taaki distance result front mein dikhe
    result_cols = [col for col in df_orders.columns if col not in ['RIS_Distance_KM']]
    final_df = df_orders[['RIS_Distance_KM'] + result_cols]
    
    # Sirf final result (distance columns) ko dikhana aur ZIP data columns ko hide karna
    display_cols = ['Order ID', 'ASIN', 'Origin Pincode', 'Destination Pincode', 'RIS_Distance_KM']
    if 'Order ID' not in final_df.columns:
        # Agar user file mein Order ID/ASIN nahi hai
        display_cols.remove('Order ID')
        if 'ASIN' in display_cols: display_cols.remove('ASIN')

    st.dataframe(final_df[display_cols].head(20))
    
    # Download button
    csv_export = final_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Results as CSV 💾",
        data=csv_export,
        file_name='RIS_Distance_Calculated_Results.csv',
        mime='text/csv',
    )
    
    # Summary
    not_found_count = (final_df['RIS_Distance_KM'] == 'PINCODE_NOT_FOUND').sum()
    if not_found_count > 0:
        st.warning(f"⚠️ Warning: {not_found_count} rows could not be calculated because the Pincode was not found in the Raw Data file.")
