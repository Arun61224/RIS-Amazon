import streamlit as st
import pandas as pd
import numpy as np
import io

# --- 1. Distance Calculation Logic (Same) ---

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculates Great-Circle Distance (km) using Spherical Law of Cosines."""
    R = 6371 
    co_lat1 = np.radians(90 - lat1)
    co_lat2 = np.radians(90 - lat2)
    delta_lon = np.radians(lon1 - lon2)
    
    cos_c = (np.cos(co_lat1) * np.cos(co_lat2)) + \
            (np.sin(co_lat1) * np.sin(co_lat2) * np.cos(delta_lon))
    cos_c = np.clip(cos_c, -1.0, 1.0)
    angular_distance = np.arccos(cos_c)
    return angular_distance * R

# --- 2. Data Loading and Caching (RIS Raw Data) ---

if 'df_zip_data' not in st.session_state:
    st.session_state['df_zip_data'] = pd.DataFrame()

@st.cache_data(show_spinner="Loading ZIP-Lat/Lon Mapping Data...")
def load_raw_data(file_path):
    """Loads the main ZIP data file, targeting the 'RawData' sheet."""
    try:
        # pd.read_excel use karein, targeting 'RawData' sheet
        df_raw = pd.read_excel(file_path, 
                               dtype={'Zip': str}, 
                               sheet_name='RawData') 
                               
    except ValueError as ve:
        raise FileNotFoundError(f"Error reading Excel sheet. Ensure the sheet is named 'RawData' and 'Zip', 'Latitude', 'Longitude' columns are present. Error: {ve}")
    except Exception as e:
        raise Exception(f"Could not read the file {file_path}. Is openpyxl installed? Error: {e}")
    
    required_raw_cols = ['Zip', 'Latitude', 'Longitude']
    if not all(col in df_raw.columns for col in required_raw_cols):
        raise ValueError(f"Raw data file must contain columns: {required_raw_cols}. Found: {list(df_raw.columns)}")
        
    # Ensure Lat/Lon columns are floats (numbers) and remove any NaN rows if present in master data
    df_raw['Latitude'] = pd.to_numeric(df_raw['Latitude'], errors='coerce')
    df_raw['Longitude'] = pd.to_numeric(df_raw['Longitude'], errors='coerce')
    
    df_raw = df_raw[required_raw_cols].set_index('Zip').dropna()
    return df_raw

# --- 3. Main Streamlit Application UI (Execution Start) ---

st.set_page_config(page_title="Bulk RIS Calculator", layout="wide")
st.title("📦 Bulk RIS (Regional In Stock) Distance Calculator")
st.markdown("Upload **Order Data** file to calculate distances. If any Postal Code is missing, you can **manually add** its details below.")

RAW_DATA_PATH = "RIS checker - Rawdata.xlsx"

try:
    df_initial = load_raw_data(RAW_DATA_PATH)
    if st.session_state['df_zip_data'].empty:
        st.session_state['df_zip_data'] = df_initial
        
except Exception as e:
    st.error(f"❌ Critical Error: ZIP-Lat/Lon mapping file ('{RAW_DATA_PATH}') could not be loaded. Please ensure file is named **RIS checker - Rawdata.xlsx**, is in the root, and contains a sheet named **RawData**. Error: {e}")
    st.stop()


# --- File Uploader ---
uploaded_file = st.file_uploader(
    "1. Upload your Order Data File (CSV or Excel) - Must contain: 'Ship From Postal Code' & 'Ship To Postal Code'", 
    type=['csv', 'xlsx']
)

SHIP_FROM_COL = 'Ship From Postal Code'
SHIP_TO_COL = 'Ship To Postal Code'


if uploaded_file is not None:
    # --- Data Reading (Order File) ---
    with st.spinner('Reading file...'):
        try:
            dtype_map = {SHIP_FROM_COL: str, SHIP_TO_COL: str}
            
            if uploaded_file.name.endswith('.csv'):
                df_orders = pd.read_csv(uploaded_file, dtype=dtype_map)
            else:
                df_orders = pd.read_excel(uploaded_file, dtype=dtype_map)

            required_cols = [SHIP_FROM_COL, SHIP_TO_COL]
            if not all(col in df_orders.columns for col in required_cols):
                st.error(f"❌ Error: The uploaded file must contain the columns: {required_cols}. Please check spelling.")
                st.stop()
                
            df_orders[SHIP_FROM_COL] = df_orders[SHIP_FROM_COL].astype(str).str.strip()
            df_orders[SHIP_TO_COL] = df_orders[SHIP_TO_COL].astype(str).str.strip()

        except Exception as e:
            st.error(f"❌ Error reading or validating file: {e}")
            st.stop()
        
    st.subheader(f"2. Calculating RIS Distance for {len(df_orders)} Orders...")

    # --- Calculation Process ---
    df_current_zip_data = st.session_state['df_zip_data']
    
    with st.spinner('Matching Postal Codes and calculating distances...'):
        
        # 1. Merge for Ship From Postal Code (Origin)
        df_temp = df_orders.merge(
            df_current_zip_data, 
            left_on=SHIP_FROM_COL, 
            right_index=True, 
            how='left', 
        ).rename(columns={'Latitude': 'Lat_Origin', 'Longitude': 'Lon_Origin'})

        # 2. Merge for Ship To Postal Code (Destination)
        df_final = df_temp.merge(
            df_current_zip_data, 
            left_on=SHIP_TO_COL, 
            right_index=True, 
            how='left', 
        ).rename(columns={'Latitude': 'Lat_Dest', 'Longitude': 'Lon_Dest'})

        # CRITICAL FIX: Ensure values are not NaN before calling distance function
        df_final['RIS_Distance_KM'] = df_final.apply(lambda row: 
            calculate_distance(
                row['Lat_Origin'], row['Lon_Origin'], 
                row['Lat_Dest'], row['Lon_Dest']
            ) 
            if pd.notna(row['Lat_Origin']) and pd.notna(row['Lon_Origin']) and \
               pd.notna(row['Lat_Dest']) and pd.notna(row['Lon_Dest'])
            else 'PINCODE_NOT_FOUND', axis=1
        )
    
    
    # --- 4. Missing Postal Code Identification and Manual Entry ---
    
    # Identify unique missing codes from Origin and Destination columns
    missing_origin = df_final[df_final['Lat_Origin'].isna()][SHIP_FROM_COL].unique()
    missing_dest = df_final[df_final['Lat_Dest'].isna()][SHIP_TO_COL].unique()
    
    all_missing_pincodes = pd.Series(np.concatenate([missing_origin, missing_dest])).unique()
    # Filter out pincodes that were already in the master data (just in case)
    pincodes_to_add = [p for p in all_missing_pincodes if p not in df_current_zip_data.index] 


    if pincodes_to_add:
        st.error(f"🚨 **{len(pincodes_to_add)}** Unique Postal Codes are **MISSING** from the Raw Data! (e.g., {', '.join(pincodes_to_add[:5])}...)")
        
        with st.expander("➕ **Manually Add Missing Postal Code Details** (Required for Calculation)", expanded=True):
            st.warning("Enter Latitude and Longitude for the missing Postal Codes and click 'Update Data' to re-calculate.")
            
            with st.form("missing_pincode_form"):
                
                new_zip = st.selectbox("Select Postal Code to Add:", options=pincodes_to_add)
                new_lat = st.number_input(f"Latitude for {new_zip}", format="%.6f")
                new_lon = st.number_input(f"Longitude for {new_zip}", format="%.6f")
                
                submitted = st.form_submit_button("Update Data and Re-Calculate")

                if submitted and new_zip and new_lat and new_lon:
                    new_row = pd.DataFrame(
                        [[new_lat, new_lon]],
                        index=[new_zip],
                        columns=['Latitude', 'Longitude']
                    )
                    
                    st.session_state['df_zip_data'] = pd.concat([st.session_state['df_zip_data'], new_row])
                    
                    st.success(f"✅ Postal Code {new_zip} added successfully! Please re-upload your file or click the 'Update Data' button if you did not upload a new file.")
    
    # --- 5. Final Result Display ---
    st.subheader("5. Final Calculated Results")
    
    display_cols = ['RIS_Distance_KM', SHIP_FROM_COL, SHIP_TO_COL] 
    if 'Order ID' in df_final.columns: display_cols.insert(1, 'Order ID')
    if 'ASIN' in df_final.columns: display_cols.insert(1, 'ASIN') 
        
    final_display_df = df_final[display_cols + [col for col in df_final.columns if col not in display_cols and col not in ['Lat_Origin', 'Lon_Origin', 'Lat_Dest', 'Lon_Dest']]]
    
    def highlight_missing(s):
        return ['background-color: #ffcccc' if v == 'PINCODE_NOT_FOUND' else '' for v in s]

    st.dataframe(
        final_display_df.style.apply(highlight_missing, subset=['RIS_Distance_KM']),
        use_container_width=True
    )
    
    # Summary
    not_found_count = (final_display_df['RIS_Distance_KM'] == 'PINCODE_NOT_FOUND').sum()
    if not_found_count > 0:
        st.warning(f"⚠️ **{not_found_count}** Rows still show 'PINCODE_NOT_FOUND'. Please add the missing Postal Codes above.")
    else:
        st.success("🎉 All distances calculated successfully!")
        
    # --- 6. Download Missing Codes List (NEW FEATURE) ---
    st.subheader("6. Download Missing Postal Codes List")
    
    if pincodes_to_add:
        # Create the DataFrame of missing codes
        df_missing_codes = pd.DataFrame(pincodes_to_add, columns=['Missing Postal Code'])
        
        # Create CSV data for download
        csv_missing_export = df_missing_codes.to_csv(index=False).encode('utf-8')
        
        st.warning("The following file contains all unique postal codes that were **not found** in your Raw Data for manual updating.")

        st.download_button(
            label="Download Pincode Note Found File 💾",
            data=csv_missing_export,
            file_name='Missing_Postal_Codes_To_Update.csv',
            mime='text/csv',
        )
    else:
        st.info("No missing postal codes found that require manual update.")
        
    # Download final calculated result
    csv_export = final_display_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Full Results as CSV 💾",
        data=csv_export,
        file_name='RIS_Distance_Calculated_Results.csv',
        mime='text/csv',
    )
