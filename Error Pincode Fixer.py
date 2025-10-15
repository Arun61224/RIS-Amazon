import pandas as pd
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# Geocoding Engine चुनें (Nominatim/OpenStreetMap)
geolocator = Nominatim(user_agent="Pincode_Locator_Tool")
# Rate Limiter: API rate limits का ध्यान रखने के लिए
# Nominatim के लिए यह 1.5 सेकंड का ब्रेक ज़रूरी है ताकि Rate Limit क्रॉस न हो।
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1.5)

# मान्य Pincode कॉलम नाम
# 'Postal Code' को आपकी फाइल के लिए जोड़ा गया है
PINCODE_COLUMNS = ['Postal Code', 'Pincode', 'PostalCod', 'PostalCode', 'ZIP'] 

def get_lat_long_from_pincode(pincode):
    """दिए गए Pincode के लिए Latitude और Longitude प्राप्त करता है।"""
    # Invalid Pincode types को हैंडल करें
    try:
        # Pincode को string में बदलें (जैसे 795001)
        pincode_str = str(int(pincode))
    except (ValueError, TypeError):
        return None, None # Invalid Pincode होने पर None return करें
        
    try:
        # Pincode के साथ 'India' जोड़ें
        location = geocode(f"{pincode_str}, India")
        if location:
            return location.latitude, location.longitude
        else:
            return None, None
    except Exception:
        # Rate limit या connection error को हैंडल करें
        return None, None

def bulk_geocoder_logic(input_file_path):
    """फाइल को लोड और प्रोसेस करता है।"""
    
    # 1. इनपुट फ़ाइल पढ़ें
    if input_file_path.endswith('.csv'):
        df = pd.read_csv(input_file_path)
    elif input_file_path.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(input_file_path)
    else:
        return False, "Invalid file format. Please use CSV or Excel."

    # Pincode कॉलम का सही नाम खोजें
    pincode_col = next((col for col in df.columns if col in PINCODE_COLUMNS), None)

    # सुनिश्चित करें कि Pincode कॉलम मौजूद है
    if not pincode_col:
        return False, f"Error: The input file must have one of these columns: {', '.join(PINCODE_COLUMNS)}."

    # 2. Geocoding शुरू करने से पहले पॉपअप दिखाएँ
    messagebox.showinfo(
        "Processing Start", 
        f"Processing {len(df)} entries from column '{pincode_col}'.\n\n"
        "This will take approx. 1.5 seconds per Pincode due to API rate limits. Please wait, the tool will notify you when finished."
    )

    # 3. Geocoding लागू करें
    results = df[pincode_col].apply(lambda x: get_lat_long_from_pincode(x))

    # 4. Lat/Long को DataFrame में जोड़ें
    df['Latitude'] = [r[0] for r in results]
    df['Longitude'] = [r[1] for r in results]

    # 5. आउटपुट फ़ाइल सेव करें
    base, ext = os.path.splitext(input_file_path)
    # Output हमेशा CSV में होगा
    output_file_path = f"{base}_LatLong_Output.csv"
    
    df.to_csv(output_file_path, index=False)
    
    return True, output_file_path

def open_file_dialog():
    """GUI फाइल सेलेक्शन विंडो खोलता है।"""
    # Root विंडो छुपाएँ
    root = tk.Tk()
    root.withdraw()

    # फाइल चुनने के लिए डायलॉग खोलें
    file_path = filedialog.askopenfilename(
        title="Select your Pincode File (CSV or Excel)",
        filetypes=(("CSV/Excel files", "*.csv *.xlsx *.xls"), ("All files", "*.*"))
    )

    if file_path:
        # Geocoding शुरू करें
        success, result_info = bulk_geocoder_logic(file_path)
        
        if success:
            messagebox.showinfo("Success 🎉", f"Geocoding complete!\nResults saved to:\n{result_info}")
        else:
            messagebox.showerror("Error ❌", result_info)
    else:
        messagebox.showinfo("Cancelled", "File selection cancelled.")

# --- कोड चलाएं ---
if __name__ == "__main__":
    open_file_dialog()
