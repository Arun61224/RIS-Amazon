import streamlit as st
import pandas as pd
import numpy as np
import io
import time 

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

# --- 2. Data Loading and Caching (OPTIMIZED MULTI-FORMAT READER) ---

# CRITICAL FIX: Ensure session state maps are dictionaries
if 'zip_lat_map' not in st.session_state or not isinstance(st.session_state['zip_lat_map'], dict):
    st.session_state['zip_lat_map'] = {}
if 'zip_lon_map' not in st.session_state or not isinstance(st.session_state['zip_lon_map'], dict):
    st.session_state['zip_lon_map'] = {}
if 'master_data_loaded' not in st.session_state:
    st.session_state['master_data_loaded'] = False

@st.cache_data(show_spinner="Loading Postal Codes Data...")
def load_raw_data_optimized(file_path, file_data=None):
    """Loads master data supporting multi-format."""
    
    file_extension = file_path.split('.')[-1].lower()
    df_raw = pd.DataFrame() # Initialize an empty DataFrame
    
    try:
        source = file_data if file_data is not None else file_path
        
        if file_extension in ['xlsx', 'xlsm']:
            df_raw = pd.read_excel(source, dtype={'Zip': str}, sheet_name='RawData')
        
        elif file_extension == 'csv':
            try:
                df_raw = pd.read_csv(source, dtype={'Zip': str}, encoding='utf-8')
            except:
                df_raw = pd.read_csv(source, dtype={'Zip': str}, encoding='latin-1')

        elif file_extension == 'txt':
            try:
                # Optimized for CSV and TSV inside TXT
                df_raw = pd.read_table(source, dtype={'Zip': str}, sep=r'[,\t]+', engine='python', skipinitialspace=True, on_bad_lines='skip')
            except:
                df_raw = pd.read_table(source, dtype={'Zip': str}, sep=r'[,\t]+', engine='python', encoding='latin-1', skipinitialspace=True, on_bad_lines='skip')
        
        else:
            raise ValueError("Unsupported file format.")
            
        if df_raw.empty:
            raise ValueError("File is empty or could not be read properly.")
            
    except Exception as e:
        raise Exception(f"Could not read the master data file. Error: {e}")
    
    required_raw_cols = ['Zip', 'Latitude', 'Longitude']
    df_raw.columns = df_raw.columns.str.strip()
    
    if not all(col in df_raw.columns for col in required_raw_cols):
        raise ValueError(f"Master data file must contain columns: {required_raw_cols}. Found: {list(df_raw.columns)}")
        
    df_raw['Latitude'] = pd.to_numeric(df_raw['Latitude'], errors='coerce')
    df_raw['Longitude'] = pd.to_numeric(df_raw['Longitude'], errors='coerce')
    
    df_raw = df_raw.dropna(subset=['Latitude', 'Longitude'])
    
    lat_map = df_raw.set_index('Zip')['Latitude'].to_dict()
    lon_map = df_raw.set_index('Zip')['Longitude'].to_dict()
    
    return lat_map, lon_map

# --- 3. Main Streamlit Application UI (Execution Start) ---

st.set_page_config(page_title="Bulk RIS Calculator", layout="wide")
st.title("📦 Bulk RIS (Regional In Stock) Distance Calculator")
st.markdown("**(Optimized for large datasets)** Use Sections 1 & 2 for calculation. Use the sidebar to update your master Postal Code data.")

# Assuming this file exists locally on the server where Streamlit is running
RAW_DATA_PATH = "RIS checker - Rawdata.xlsx" 

# Initial Load logic
if not st.session_state['master_data_loaded']:
    try:
        lat_map, lon_map = load_raw_data_optimized(RAW_DATA_PATH)
        st.session_state['zip_lat_map'] = lat_map
        st.session_state['zip_lon_map'] = lon_map
        st.session_state['master_data_loaded'] = True
        st.sidebar.success(f"Master Data loaded: {len(lat_map)} Postal Codes")
            
    except Exception as e:
        st.sidebar.error(f"❌ Master Data Load Error: Initial load failed. Please ensure '{RAW_DATA_PATH}' is correct or upload a file in the sidebar. Error: {e}")


# --- Section 7: Master Data Upload/Update (MOVED TO SIDEBAR) ---
def handle_master_update(uploaded_file):
    if uploaded_file is not None:
        st.info("Uploading new Master Data...")
        try:
            file_name = uploaded_file.name
            new_lat_map, new_lon_map = load_raw_data_optimized(file_name, uploaded_file)
            
            st.session_state['zip_lat_map'] = new_lat_map
            st.session_state['zip_lon_map'] = new_lon_map
            st.session_state['master_data_loaded'] = True
            
            st.success(f"✅ Master Data successfully updated! {len(new_lat_map)} Postal Codes loaded.")
            load_raw_data_optimized.clear() 
            time.sleep(1)
            st.rerun()

        except Exception as e:
            st.error(f"❌ ERROR: Updated Master file could not be processed. Error: {e}")

with st.sidebar:
    st.subheader("Master Data Update (Section 7)")
    uploaded_master_file = st.file_uploader(
        f"Upload updated master file (.xlsx, .xlsm, .csv, .txt). Excel needs 'RawData' sheet.",
        type=['xlsx', 'xlsm', 'csv', 'txt'],
        key="master_data_upload_sidebar"
    )
    handle_master_update(uploaded_master_file) 

# Check if data is ready before proceeding to calculation sections
if not st.session_state['master_data_loaded']:
    st.warning("Please resolve the Master Data Load Error (check file/path) or upload an updated file using the sidebar.")
    st.stop()
# ------------------------------------------------------------------------------------------------------------------------------------


st.subheader("1. Upload Order File for RIS Calculation")

# --- NEW: User Input for Column Names ---
col_input1, col_input2 = st.columns(2)

with col_input1:
    ship_from_col_input = st.text_input(
        "Enter **'Ship From'** Postal Code Column Name:",
        value='Ship From Postal Code',
        key="ship_from_col_input"
    ).strip()

with col_input2:
    ship_to_col_input = st.text_input(
        "Enter **'Ship To'** Postal Code Column Name:",
        value='Ship To Postal Code',
        key="ship_to_col_input"
    ).strip()
    
SHIP_FROM_COL = ship_from_col_input
SHIP_TO_COL = ship_to_col_input

# --- File Uploader ---
uploaded_file = st.file_uploader(
    f"Upload your Order Data File (Must contain: '{SHIP_FROM_COL}' & '{SHIP_TO_COL}')", 
    type=['csv', 'xlsx', 'xlsm', 'txt'],
    key="order_file_uploader"
)


if uploaded_file is not None:
    # Validation Check for Column Names
    if not SHIP_FROM_COL or not SHIP_TO_COL:
        st.error("❌ Error: Please enter valid column names for both 'Ship From' and 'Ship To' postal codes.")
        st.stop()

    df_orders = pd.DataFrame() # Initialize df_orders safely
    
    # --- Data Reading and Calculation Process (OPTIMIZED LOOKUP) ---
    with st.spinner('Processing and calculating distances...'):
        try:
            dtype_map = {SHIP_FROM_COL: str, SHIP_TO_COL: str}
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension in ['xlsx', 'xlsm']:
                df_orders = pd.read_excel(uploaded_file, dtype=dtype_map)
            elif file_extension == 'csv':
                df_orders = pd.read_csv(uploaded_file, dtype=dtype_map, encoding='utf-8')
            elif file_extension == 'txt':
                df_orders = pd.read_table(uploaded_file, dtype=dtype_map, sep=r'[,\t]+', engine='python', skipinitialspace=True, on_bad_lines='skip')
            else:
                raise ValueError("Unsupported order file format.")

            required_cols = [SHIP_FROM_COL, SHIP_TO_COL]
            df_orders.columns = df_orders.columns.str.strip() # Strip spaces from column names
            if not all(col in df_orders.columns for col in required_cols):
                st.error(f"❌ Error: The uploaded file must contain the columns: {required_cols}. Found: {list(df_orders.columns)}")
                st.stop()
                
            df_orders[SHIP_FROM_COL] = df_orders[SHIP_FROM_COL].astype(str).str.strip()
            df_orders[SHIP_TO_COL] = df_orders[SHIP_TO_COL].astype(str).str.strip()

        except Exception as e:
            st.error(f"❌ Error reading or validating file: {e}")
            st.stop()
            
        current_lat_map = st.session_state['zip_lat_map']
        current_lon_map = st.session_state['zip_lon_map']

        # Optimized Coordinate Lookup
        df_orders['Lat_Origin'] = df_orders[SHIP_FROM_COL].apply(lambda x: current_lat_map.get(x))
        df_orders['Lon_Origin'] = df_orders[SHIP_FROM_COL].apply(lambda x: current_lon_map.get(x))
        df_orders['Lat_Dest'] = df_orders[SHIP_TO_COL].apply(lambda x: current_lat_map.get(x))
        df_orders['Lon_Dest'] = df_orders[SHIP_TO_COL].apply(lambda x: current_lon_map.get(x))
        
        df_final = df_orders 
        
        # Distance Calculate karna
        df_final['RIS_Distance_KM'] = df_final.apply(lambda row: 
            calculate_distance(
                row['Lat_Origin'], row['Lon_Origin'], 
                row['Lat_Dest'], row['Lon_Dest']
            ) 
            if pd.notna(row['Lat_Origin']) and pd.notna(row['Lon_Origin']) and \
               pd.notna(row['Lat_Dest']) and pd.notna(row['Lon_Dest'])
            else 'PINCODE_NOT_FOUND', axis=1
        )
    
    st.subheader("2. Calculation Results")


    # --- 4. Missing Postal Code Identification (PREPARE DOWNLOAD) ---
    
    missing_origin = df_final[df_final['Lat_Origin'].isna()][SHIP_FROM_COL].unique()
    missing_dest = df_final[df_final['Lat_Dest'].isna()][SHIP_TO_COL].unique()
    
    all_missing_pincodes = pd.Series(np.concatenate([missing_origin, missing_dest])).unique()
    # Filter to only include pincodes not already in the master map
    pincodes_to_add = [p for p in all_missing_pincodes if p not in current_lat_map] 


    # --- 6. Download Missing Codes List & Quick Upload ---
    st.subheader("6. Data Update Workflow")
    
    col_dl1, col_dl2 = st.columns(2)
    
    if pincodes_to_add:
        # Download Missing Codes (Left Column)
        with col_dl1:
            st.error(f"🚨 **{len(pincodes_to_add)}** Missing Codes. Download file to update.")
            
            df_missing_codes_to_update = pd.DataFrame({
                'Zip': pincodes_to_add, 
                'Latitude': [''] * len(pincodes_to_add), 
                'Longitude': [''] * len(pincodes_to_add),
                'Status': ['UPDATE_REQUIRED'] * len(pincodes_to_add)
            })
            
            csv_missing_export = df_missing_codes_to_update.to_csv(index=False).encode('utf-8')

            st.download_button(
                label="Download Missing Pincodes 📥",
                data=csv_missing_export,
                file_name='Missing_Postal_Codes_To_Update.csv',
                mime='text/csv',
                use_container_width=True
            )
            st.caption("Update Latitude/Longitude data in this file.")
        
        # Quick Upload (Right Column)
        with col_dl2:
            st.success("✔ Ready to Upload")
            uploaded_quick_update_file = st.file_uploader(
                "Upload updated **Missing Pincodes** file (.csv)",
                type=['csv'],
                key="quick_update_file_uploader"
            )

            # Handle Quick Upload
            if uploaded_quick_update_file is not None:
                st.info("Merging new Pincodes...")
                try:
                    df_update = pd.read_csv(uploaded_quick_update_file, dtype={'Zip': str})
                    df_update = df_update.dropna(subset=['Latitude', 'Longitude'])
                    
                    new_lat_map = df_update.set_index('Zip')['Latitude'].to_dict()
                    new_lon_map = df_update.set_index('Zip')['Longitude'].to_dict()
                    
                    st.session_state['zip_lat_map'].update(new_lat_map)
                    st.session_state['zip_lon_map'].update(new_lon_map)
                    
                    st.success(f"✅ {len(new_lat_map)} Postal Codes added/updated in the current map! Please re-upload your order file (Section 1).")
                    
                    st.session_state["order_file_uploader"] = None
                    time.sleep(1)
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ ERROR: Quick update failed. Check if file has 'Zip', 'Latitude', 'Longitude' columns filled correctly. Error: {e}")

    else:
        st.info("No missing postal codes found! Calculation is complete.")

    # --- 5. Final Result Display and Download ---
    st.subheader("5. Full Calculated Results Preview")
    
    # --- CHANGE 1: DEFINING COLUMNS WITHOUT COORDS ---
    # NOTE: SHIP_FROM_COL and SHIP_TO_COL now use the user's input names
    display_cols = ['RIS_Distance_KM', SHIP_FROM_COL, SHIP_TO_COL] 
    # Add optional columns if they exist
    if 'Order ID' in df_final.columns: display_cols.insert(1, 'Order ID')
    if 'ASIN' in df_final.columns: display_cols.insert(1, 'ASIN') 
        
    # --- CHANGE 2: EXCLUDING COORD COLUMNS FROM DISPLAY DATAFRAME ---
    # We explicitly exclude all coordinate columns from the final display and the "other columns" list
    COORD_COLS = ['Lat_Origin', 'Lon_Origin', 'Lat_Dest', 'Lon_Dest']
    
    final_display_df = df_final[display_cols + [col for col in df_final.columns if col not in display_cols and col not in COORD_COLS]]
    
    def highlight_missing(s):
        return ['background-color: #ffcccc' if v == 'PINCODE_NOT_FOUND' else '' for v in s]

    st.dataframe(
        final_display_df.style.apply(highlight_missing, subset=['RIS_Distance_KM']),
        use_container_width=True
    )
    
    # Summary
    not_found_count = (final_display_df['RIS_Distance_KM'] == 'PINCODE_NOT_FOUND').sum()
    if not_found_count > 0:
        st.warning(f"⚠️ **{not_found_count}** Rows still show 'PINCODE_NOT_FOUND'. Please use Section 6 to update the missing codes.")
    else:
        st.success("🎉 All distances calculated successfully!")
        
    # --- Download Buttons ---
    
    # ----------------------------------------------------
    # LOGIC: Filter data for distances >= 300 KM
    
    # 1. Filter out rows where distance is not a valid number (i.e., 'PINCODE_NOT_FOUND')
    df_calculated = df_final[df_final['RIS_Distance_KM'] != 'PINCODE_NOT_FOUND'].copy()
    
    # 2. Convert the filtered distance column to numeric type for comparison
    df_calculated.loc[:, 'RIS_Distance_KM'] = pd.to_numeric(df_calculated['RIS_Distance_KM'])

    # 3. Filter for distances >= 300 KM
    df_300km_plus = df_calculated[df_calculated['RIS_Distance_KM'] >= 300].copy()
    
    # --- CHANGE 3: EXCLUDING COORD COLUMNS FROM DOWNLOAD DATAFRAME ---
    # Ensure the 300km+ dataframe only contains the necessary columns for export
    all_export_cols = [col for col in df_300km_plus.columns if col not in COORD_COLS]
    df_300km_plus_export = df_300km_plus[all_export_cols]

    # Create CSV export for 300km+ data
    csv_export_300km = df_300km_plus_export.to_csv(index=False).encode('utf-8')
    # ----------------------------------------------------
    
    # 1. Create columns for side-by-side buttons
    col_final_dl, col_final_merge = st.columns(2)
    
    with col_final_dl:
        # Download Button for 300 KM+ data
        st.download_button(
            label=f"Download Results (>= 300 KM) ({len(df_300km_plus)} rows) 📥",
            data=csv_export_300km,
            file_name='RIS_Distance_300KM_Plus_Results.csv', 
            mime='text/csv',
            use_container_width=True
        )
        
    with col_final_merge:
        # "Merge with MTR" button (Placeholder remains)
        if st.button(
            label="Merge this Data with MTR",
            key="merge_with_mtr_button",
            use_container_width=True,
            help="This button is a placeholder. You would implement your custom logic here to merge RIS results with your MTR data."
        ):
            st.info("Merge with MTR function triggered! Please implement your specific data merging logic in the code.")
            
    # --------------------------------------------------------

else:
    # If no file is uploaded in Section 1, prompt the user.
    st.info("👆 Please upload your Order Data File in Section 1 to start the calculation.")
