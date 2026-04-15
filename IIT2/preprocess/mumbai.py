import pandas as pd
import glob
import os

def extract_mumbai_data(input_folder, output_folder):
    """
    Reads all CSVs in the input folder, removes all rows outside of Mumbai,
    and saves the filtered data into the output folder.
    """
    # 1. Define Mumbai's Bounding Box
    MIN_LAT = 18.8900
    MAX_LAT = 19.3000
    MIN_LON = 72.7500
    MAX_LON = 73.0000

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    # Find all CSV files
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in '{input_folder}'.")
        return

    print(f"Found {len(csv_files)} files. Starting Mumbai extraction...\n")

    total_india_rows = 0
    total_mumbai_rows = 0

    # Process each file
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        output_path = os.path.join(output_folder, f"mumbai_only_{filename}")
        
        print(f"Processing: {filename}...")
        
        try:
            # Load the whole file
            df = pd.read_csv(file_path)
            original_count = len(df)
            total_india_rows += original_count
            
            # 2. APPLY THE BOUNDING BOX FILTER
            # This keeps ONLY the rows where the latitude and longitude are inside our box
            mumbai_df = df[
                (df['latitude'] >= MIN_LAT) & 
                (df['latitude'] <= MAX_LAT) & 
                (df['longitude'] >= MIN_LON) & 
                (df['longitude'] <= MAX_LON)
            ]
            
            mumbai_count = len(mumbai_df)
            total_mumbai_rows += mumbai_count
            
            print(f"  -> Kept {mumbai_count:,} out of {original_count:,} rows.")
            
            # 3. Save the filtered data if it actually found points in Mumbai
            if mumbai_count > 0:
                mumbai_df.to_csv(output_path, index=False)
                print(f"  -> Saved to {output_path}")
            else:
                print("  -> No Mumbai points found in this file. Skipping save.")
                
        except Exception as e:
            print(f"  [ERROR] Failed to process {filename}. Error: {e}")

    # Final Summary
    print("\n" + "="*40)
    print("EXTRACTION COMPLETE")
    print("="*40)
    print(f"Total Rows Scanned (India): {total_india_rows:,}")
    print(f"Total Rows Kept (Mumbai):   {total_mumbai_rows:,}")
    
    if total_india_rows > 0:
        percentage = (total_mumbai_rows / total_india_rows) * 100
        print(f"Mumbai represents {percentage:.2f}% of your dataset.")

# ==========================================
# HOW TO USE
# ==========================================
# 1. Folder with your India-wide clean files
INPUT_DIR = 'E://clean_data/02_clean' 

# 2. New folder where ONLY Mumbai data will be saved
OUTPUT_DIR = '../mumbai_data/02_mumbai'

# Run the extraction!
extract_mumbai_data(INPUT_DIR, OUTPUT_DIR)