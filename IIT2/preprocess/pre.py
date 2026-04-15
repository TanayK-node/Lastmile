import os
import glob
import pandas as pd

def format_all_csvs_in_folder(input_folder, output_folder):
    """
    Reads all tab-separated files in the input_folder and 
    saves them as properly formatted comma-separated CSVs in the output_folder.
    """
    # 1. Create the output folder if it doesn't already exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    # 2. Get a list of all CSV files in the input folder
    search_pattern = os.path.join(input_folder, "*.csv")
    csv_files = glob.glob(search_pattern)
    
    if not csv_files:
        print(f"No CSV files found in '{input_folder}'.")
        return

    print(f"Found {len(csv_files)} files. Starting conversion...")

    # 3. Loop through each file and reformat it
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        output_path = os.path.join(output_folder, f"formatted_{filename}")
        
        try:
            # Read the unformatted file using tab separation ('\t')
            df = pd.read_csv(file_path, sep='\t')
            
            # Save the file out using standard commas (',')
            df.to_csv(output_path, index=False, sep=',')
            
            print(f"  [SUCCESS] Formatted: {filename}")
        except Exception as e:
            print(f"  [ERROR] Failed to format {filename}. Error: {e}")
            
    print("All files processed!")

# ==========================================
# HOW TO USE THE SCRIPT
# ==========================================
# 1. Change INPUT_DIR to the folder containing your unformatted files.
# 2. Change OUTPUT_DIR to the folder where you want the clean files saved.

INPUT_DIR = 'E://gps_data/02'
OUTPUT_DIR = 'E://clean_data/02_clean'

# Run the function
format_all_csvs_in_folder(INPUT_DIR, OUTPUT_DIR)