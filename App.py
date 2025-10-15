import streamlit as st
import pandas as pd
import numpy as np
import io

# --- 1. Distance Calculation Logic ---

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculates Great-Circle Distance (km) using Spherical Law of Cosines."""
    R = 6371  # Earth radius in km
    
    co_lat1 = np.radians(90 - lat1)
    co_lat2 = np.radians(90 - lat2)
    delta_lon = np.radians(lon1 - lon2)
    
    cos_c = (np.cos(co_lat1) * np.cos(co_lat2)) + \
            (np.sin(co_lat1) * np.sin(co_lat2) * np.cos(delta_lon))
    
    cos_c = np.clip(cos_c, -1.0, 1.0)
    angular_distance = np.arccos(cos_c)
    
    return angular_distance * R

# --- 2. Data Loading and Caching (Critical File) ---

# Global variable to store updated ZIP data (for session-level additions)
# Yeh user ke manual entries ko session mein store karega.
if 'df_zip_data' not in st.session_state:
    st.session_state['df_zip_data'] = pd.DataFrame()

@st.cache_data(show_spinner="Loading ZIP-Lat/Lon Mapping Data...")
def load_raw_data(file_path):
    """Loads the main ZIP data file."""
    try:
        df_raw = pd.read_csv(file_path, dtype={'Zip': str}, encoding='utf-8')
    except UnicodeDecodeError:
        df_raw = pd.read_csv(file_path, dtype={'Zip': str}, encoding='latin-1')
    
    # Ensure correct column names and set 'Zip' as index
    df_raw = df_raw[['Zip', 'Latitude', 'Longitude']].set_index('Zip')
    return df_raw

# --- 3. Main Streamlit Application UI ---

st.set_page_config(page_title="Bulk RIS Calculator", layout="wide")
st.title("📦 Bulk RIS (Regional In Stock) Distance Calculator")
st.markdown("Upload **Order Data** file to calculate distances. If any Pincode is missing, you can **manually add** its details below.")

# Load Raw Data once at the start
RAW_DATA_PATH = "RIS checker.xlsx - RawData.csv"

try:
    df_initial = load_raw_data(RAW_DATA_PATH)
    # Load initial data into session state only if it's the first time
    if st.session_state['df_zip_data'].empty:
        st.session_state['df_zip_data'] = df_initial
        
except Exception as e:
    st.error(f"❌ Critical Error: ZIP-Lat/Lon mapping file ('{RAW_DATA_PATH}') could not be loaded. Please ensure it is present and correct. Error: {e}")
    st.stop()


# --- File Uploader ---
uploaded_file = st.file_uploader(
    "1. Upload your Order Data File (CSV or Excel) - Must contain: 'Origin Pincode' & 'Destination Pincode'", 
    type=['csv', 'xlsx']
)

if uploaded_file is not None:
    # --- Data Reading ---
    with st.spinner('Reading file...'):
        try:
            if uploaded_file.name.endswith('.csv'):
                df_orders = pd.read_csv(uploaded_file, dtype={'Origin Pincode': str, 'Destination Pincode': str})
            else:
                df_orders = pd.read_excel(uploaded_file, dtype={'Origin Pincode': str, 'Destination Pincode': str})

            required_cols = ['Origin Pincode', 'Destination Pincode']
            if not all(col in df_orders.columns for col in required_cols):
                st.error(f"❌ Error: The uploaded file must contain the columns: {required_cols}")
                st.stop()
                
            df_orders['Origin Pincode'] = df_orders['Origin Pincode'].astype(str).str.strip()
            df_orders['Destination Pincode'] = df_orders['Destination Pincode'].astype(str).str.strip()

        except Exception as e:
            st.error(f"❌ Error reading or validating file: {e}")
            st.stop()
        
    st.subheader(f"2. Calculating RIS Distance for {len(df_orders)} Orders...")

    # --- Calculation Process ---
    df_current_zip_data = st.session_state['df_zip_data']
    
    with st.spinner('Matching Pincodes and calculating distances...'):
        
        # 1. Merge for Origin Pincode
        df_temp = df_orders.merge(
            df_current_zip_data, 
            left_on='Origin Pincode', 
            right_index=True, 
            how='left', 
        ).rename(columns={'Latitude': 'Lat_Origin', 'Longitude': 'Lon_Origin'})

        # 2. Merge for Destination Pincode
        df_final = df_temp.merge(
            df_current_zip_data, 
            left_on='Destination Pincode', 
            right_index=True, 
            how='left', 
        ).rename(columns={'Latitude': 'Lat_Dest', 'Longitude': 'Lon_Dest'})


        # 3. Distance Calculate karna
        df_final['RIS_Distance_KM'] = df_final.apply(lambda row: 
            calculate_distance(
                row['Lat_Origin'], row['Lon_Origin'], 
                row['Lat_Dest'], row['Lon_Dest']
            ) 
            if pd.notna(row['Lat_Origin']) and pd.notna(row['Lat_Dest'])
            else 'PINCODE_NOT_FOUND', axis=1
        )
    
    
    # --- 4. Missing Pincode Identification and Manual Entry ---
    
    missing_origin = df_final[df_final['Lat_Origin'].isna()]['Origin Pincode'].unique()
    missing_dest = df_final[df_final['Lat_Dest'].isna()]['Destination Pincode'].unique()
    
    # Unique missing pincodes nikalna
    all_missing_pincodes = pd.Series(np.concatenate([missing_origin, missing_dest])).unique()
    
    # Filter out pincodes jo already manual additions mein hain
    pincodes_to_add = [p for p in all_missing_pincodes if p not in st.session_state['df_zip_data'].index]

    if pincodes_to_add:
        st.error(f"🚨 **{len(pincodes_to_add)}** Unique Pincodes are **MISSING** from the Raw Data! (e.g., {', '.join(pincodes_to_add[:5])}...)")
        
        with st.expander("➕ **Manually Add Missing Pincode Details** (Required for Calculation)", expanded=True):
            st.warning("Enter Latitude and Longitude for the missing Pincodes and click 'Update Data' to re-calculate.")
            
            # --- Dynamic Form for Missing Pincodes ---
            with st.form("missing_pincode_form"):
                
                new_zip = st.selectbox("Select Pincode to Add:", options=pincodes_to_add)
                new_lat = st.number_input(f"Latitude for {new_zip}", format="%.6f")
                new_lon = st.number_input(f"Longitude for {new_zip}", format="%.6f")
                
                submitted = st.form_submit_button("Update Data and Re-Calculate")

                if submitted and new_zip and new_lat and new_lon:
                    # Naya row banana
                    new_row = pd.DataFrame(
                        [[new_lat, new_lon]],
                        index=[new_zip],
                        columns=['Latitude', 'Longitude']
                    )
                    
                    # Session state data mein append karna
                    st.session_state['df_zip_data'] = pd.concat([st.session_state['df_zip_data'], new_row])
                    
                    st.success(f"✅ Pincode {new_zip} added successfully! Please click the 'Calculate' button again.")
                    # Streamlit will rerun the script automatically, leading to a new calculation attempt.
    
    # --- 5. Final Result Display ---
    st.subheader("5. Final Calculated Results")
    
    # Display columns: distance sabse pehle dikhani hai
    display_cols = ['RIS_Distance_KM', 'Origin Pincode', 'Destination Pincode']
    
    if 'Order ID' in df_final.columns: display_cols.insert(1, 'Order ID')
    if 'ASIN' in df_final.columns: display_cols.insert(1, 'ASIN') 
        
    final_display_df = df_final[display_cols + [col for col in df_final.columns if col not in display_cols and col not in ['Lat_Origin', 'Lon_Origin', 'Lat_Dest', 'Lon_Dest']]]
    
    # Highlighting missing values
    def highlight_missing(s):
        return ['background-color: #ffcccc' if v == 'PINCODE_NOT_FOUND' else '' for v in s]

    st.dataframe(
        final_display_df.style.apply(highlight_missing, subset=['RIS_Distance_KM']),
        use_container_width=True
    )
    
    # Summary
    not_found_count = (final_display_df['RIS_Distance_KM'] == 'PINCODE_NOT_FOUND').sum()
    if not_found_count > 0:
        st.warning(f"⚠️ **{not_found_count}** Rows still show 'PINCODE_NOT_FOUND'. Please add the missing Pincodes above.")
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
