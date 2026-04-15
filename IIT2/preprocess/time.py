import pandas as pd
import glob
import os

def process_mumbai_data(input_folder, output_folder):
    """
    Reads cleaned CSVs, filters for Mumbai, converts Unix timestamps to IST,
    adds explicit Date and Time columns, and saves the result.
    """
    # 1. Define Mumbai's Bounding Box
    MIN_LAT = 18.8900
    MAX_LAT = 19.3000
    MIN_LON = 72.7500
    MAX_LON = 73.0000

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in '{input_folder}'.")
        return

    print(f"Found {len(csv_files)} files. Starting processing and time conversion...\n")

    for file_path in csv_files:
        filename = os.path.basename(file_path)
        output_path = os.path.join(output_folder, f"mumbai_ist_{filename}")
        
        print(f"Processing: {filename}...")
        
        try:
            # Load the file
            df = pd.read_csv(file_path)
            
            # 2. Filter for Mumbai FIRST (This makes the time conversion much faster 
            # because it only has to calculate the time for the points we are keeping)
            df = df[
                (df['latitude'] >= MIN_LAT) & 
                (df['latitude'] <= MAX_LAT) & 
                (df['longitude'] >= MIN_LON) & 
                (df['longitude'] <= MAX_LON)
            ].copy() # .copy() prevents warnings when adding new columns
            
            if len(df) > 0:
                # 3. TIME CONVERSION LOGIC
                # A. Convert Unix to UTC Datetime
                df['datetime_utc'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
                
                # B. Add 5 Hours and 30 Minutes for IST
                df['datetime_ist'] = df['datetime_utc'] + pd.Timedelta(hours=5, minutes=30)
                
                # C. Extract Date and Time into their own clean columns
                # '%Y-%m-%d' gives you e.g., "2023-06-01"
                df['date_ist'] = df['datetime_ist'].dt.strftime('%Y-%m-%d')
                
                # '%I:%M:%S %p' gives you e.g., "05:30:01 AM"
                df['time_ist'] = df['datetime_ist'].dt.strftime('%I:%M:%S %p')
                
                # Optional: Drop the intermediate datetime columns if you want to keep the file size small
                df = df.drop(columns=['datetime_utc', 'datetime_ist'])
                
                # 4. Save the file
                df.to_csv(output_path, index=False)
                print(f"  -> Saved {len(df):,} rows with Date & Time columns to {output_path}")
            else:
                print("  -> No Mumbai points found. Skipping.")
                
        except Exception as e:
            print(f"  [ERROR] Failed to process {filename}. Error: {e}")

    print("\nProcessing Complete! All your files now have 'date_ist' and 'time_ist' columns.")

# ==========================================
# HOW TO USE
# ==========================================
# 1. Folder with your India-wide clean files
INPUT_DIR = '../mumbai_data/02_mumbai' 

# 2. New folder where ONLY Mumbai data WITH time columns will be saved
OUTPUT_DIR = './mumbai_time/02_time'

# Run it!
process_mumbai_data(INPUT_DIR, OUTPUT_DIR)