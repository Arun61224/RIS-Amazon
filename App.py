import streamlit as st
import pandas as pd
import numpy as np
import io

# --- 1. Distance Calculation Logic ---

def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Do set of coordinates ke beech ki distance (in kilometers) calculate karta hai 
    Spherical Law of Cosines ka use karke (Great-Circle Distance).
    """
    R = 6371  # Earth ka radius kilometers mein
    
    # Degrees ko Radians mein convert karna aur co-latitude (90-lat) nikalna
    co_lat1 = np.radians(90 - lat1)
    co_lat2 = np.radians(90 - lat2)
    delta_lon = np.radians(lon1 - lon2)
    
    # Spherical Law of Cosines ka core calculation
    # cos(c) = cos(a) * cos(b) + sin(a) * sin(b) * cos(C)
    cos_c = (np.cos(co_lat1) * np.cos(co_lat2)) + \
            (np.sin(co_lat1) * np.sin(co_lat2) * np.cos(delta_lon))
    
    # Numerical stability ke liye value ko [-1, 1] range mein clip karna
    cos_c = np.clip(cos_c, -1.0, 1.0)

    # Distance calculation: ACOS(cos_c) * R
    angular_distance = np.arccos(cos_c)
    distance_km = angular_distance * R
    
    return distance_km

# --- 2. Data Loading and Caching (Critical File) ---

@st.cache_data
def load_raw_data(file_path):
    """ZIP data (Latitude/Longitude mapping) load karta hai, jismein encoding fix bhi hai."""
    try:
        # Pehle default utf-8 encoding try karna
        df_raw = pd.read_csv(file_path, dtype={'Zip': str})
    except UnicodeDecodeError:
        try:
            # Agar utf-8 fail ho toh latin-1 encoding try karna (Jo zyada Indian CSVs mein hoti hai)
            df_raw = pd.read_csv(file_path, dtype={'Zip': str}, encoding='latin-1')
        except Exception:
            # Agar dono fail ho toh error throw karna
            raise FileNotFoundError(f"Could not read the file {file_path} with standard encodings.")
    
    # Sirf relevant columns rakhna aur 'Zip' ko index banana
    df_raw = df_raw[['Zip', 'Latitude', 'Longitude']].set_index('Zip')
    return df_raw

# --- 3. Main Streamlit Application UI ---

st.set_page_config(page_title="Bulk RIS Calculator", layout="wide")

st.title("📦 Bulk RIS (Regional In Stock) Distance Calculator")
st.markdown("Upload **Order Data** file (CSV/Excel) to calculate distances based on **Origin** and **Destination Pincodes**.")

# Server par rakhi hui Raw Data file load karna
RAW_DATA_PATH = "RIS checker.xlsx - RawData.csv"

# Data loading with error handling
try:
    df_zip_data = load_raw_data(RAW_DATA_PATH)
except FileNotFoundError as e:
    st.error(f"❌ Critical Error: ZIP-Lat/Lon mapping file ('{RAW_DATA_PATH}') nahi mili ya read nahi ho saki. Please confirm the file is in the repository root and is not corrupted. ({e})")
    st.stop()
except Exception as e:
    st.error(f"❌ An unexpected error occurred during raw data loading: {e}")
    st.stop()


# File Uploader
uploaded_file = st.file_uploader(
    "1. Upload your Order Data File (CSV or Excel) - Must contain columns: 'Origin Pincode' and 'Destination Pincode'", 
    type=['csv', 'xlsx']
)

if uploaded_file is not None:
    # File type ke according data read karna
    with st.spinner('Reading file...'):
        try:
            if uploaded_file.name.endswith('.csv'):
                df_orders = pd.read_csv(uploaded_file, dtype={'Origin Pincode': str, 'Destination Pincode': str})
            else: # Excel (.xlsx) file
                df_orders = pd.read_excel(uploaded_file, dtype={'Origin Pincode': str, 'Destination Pincode': str})

        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
            st.stop()
        
    st.subheader("2. Uploaded Order Data Preview")
    st.dataframe(df_orders.head())

    # Required columns check karna
    required_cols = ['Origin Pincode', 'Destination Pincode']
    if not all(col in df_orders.columns for col in required_cols):
        st.error(f"❌ Error: The uploaded file mein yeh columns hona zaruri hai: {required_cols}")
        st.stop()
        
    st.subheader(f"3. Calculating RIS Distance for {len(df_orders)} Orders...")

    # --- Calculation Process ---
    with st.spinner('Matching Pincodes and calculating distances...'):
        
        # 1. Pincodes ko String mein convert karna taaki merge sahi ho
        df_orders['Origin Pincode'] = df_orders['Origin Pincode'].astype(str)
        df_orders['Destination Pincode'] = df_orders['Destination Pincode'].astype(str)
        
        # 2. Origin Pincode ki Lat/Lon fetch karna
        df_temp = df_orders.merge(
            df_zip_data, 
            left_on='Origin Pincode', 
            right_index=True, 
            how='left', 
        ).rename(columns={'Latitude': 'Lat_Origin', 'Longitude': 'Lon_Origin'})

        # 3. Destination Pincode ki Lat/Lon fetch karna
        df_final = df_temp.merge(
            df_zip_data, 
            left_on='Destination Pincode', 
            right_index=True, 
            how='left', 
        ).rename(columns={'Latitude': 'Lat_Dest', 'Longitude': 'Lon_Dest'})


        # 4. Distance Calculate karna (Apply function ka use karke)
        df_final['RIS_Distance_KM'] = df_final.apply(lambda row: 
            calculate_distance(
                row['Lat_Origin'], row['Lon_Origin'], 
                row['Lat_Dest'], row['Lon_Dest']
            ) 
            # Agar koi Lat/Lon value NaN (Not Found) hai toh error message dena
            if pd.notna(row['Lat_Origin']) and pd.notna(row['Lat_Dest'])
            else 'PINCODE_NOT_FOUND', axis=1
        )
    
    
    # --- Final Result Display ---
    st.subheader("4. Final Calculated Results")
    
    # Display columns: distance sabse pehle dikhani hai
    display_cols = ['RIS_Distance_KM', 'Origin Pincode', 'Destination Pincode']
    
    # Agar user ki file mein 'Order ID' ya 'ASIN' hai toh unhe bhi add karna
    if 'Order ID' in df_final.columns:
        display_cols.insert(1, 'Order ID')
    if 'ASIN' in df_final.columns:
        display_cols.insert(1, 'ASIN') 
        
    # Final DataFrame ko clean karna (sirf zaroori columns rakhna)
    final_display_df = df_final[display_cols + [col for col in df_final.columns if col not in display_cols and col not in ['Lat_Origin', 'Lon_Origin', 'Lat_Dest', 'Lon_Dest']]]

    st.dataframe(final_display_df.head(100))
    
    # Summary
    not_found_count = (final_display_df['RIS_Distance_KM'] == 'PINCODE_NOT_FOUND').sum()
    if not_found_count > 0:
        st.warning(f"⚠️ **{not_found_count}** Pincodes Raw Data mein nahi mile. Unke liye distance 'PINCODE_NOT_FOUND' hai.")
    else:
        st.success("🎉 All distances calculated successfully!")
        
    # Download button
    csv_export = final_display_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Results as CSV 💾",
        data=csv_export,
        file_name='RIS_Distance_Calculated_Results.csv',
        mime='text/csv',
    )
